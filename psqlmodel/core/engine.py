from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, ContextManager, AsyncContextManager, Callable
import threading
import queue
import os
import importlib.util
import asyncio
import concurrent.futures
from collections import deque, defaultdict
import datetime

import psycopg
import asyncpg

from ..orm.model import PSQLModel
from ..orm.column import Column  # puede ser útil para typings internos


# ============================================================
# Custom Exceptions
# ============================================================

class PSQLModelError(Exception):
    """Base exception for all PSQLModel errors."""
    pass


class DatabaseNotFoundError(PSQLModelError):
    """Raised when attempting to connect to a non-existent database."""
    
    def __init__(self, database_name: str, original_error: Exception):
        self.database_name = database_name
        self.original_error = original_error
        
        message = (
            f"\n\n{'=' * 70}\n"
            f"❌ DATABASE NOT FOUND: '{database_name}'\n"
            f"{'=' * 70}\n\n"
            f"The database '{database_name}' does not exist on the server.\n\n"
            f"💡 SOLUTION:\n"
            f"   Enable automatic database creation by setting:\n\n"
            f"   PSQLModel.init(\n"
            f"       'your_connection_string',\n"
            f"       ensure_database=True  # ← Set this to True\n"
            f"   )\n\n"
            f"   Or create the database manually:\n"
            f"   CREATE DATABASE {database_name};\n\n"
            f"{'=' * 70}\n"
        )
        super().__init__(message)


class ConnectionError(PSQLModelError):
    """Raised when connection to database fails."""
    pass


@dataclass
class EngineConfig:
    dsn: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: Optional[str] = None
    async_: bool = False
    pool_size: int = 20
    auto_adjust_pool_size: bool = False
    max_pool_size: Optional[int] = None
    connection_timeout: Optional[float] = None
    ensure_database: bool = True
    ensure_tables: bool = True
    ensure_migrations: bool = False  # Auto-run migrations on startup
    check_schema_drift: bool = True  # Warn if models differ from DB
    migrations_path: Optional[str] = None  # Path for migrations directory
    models_path: Optional[str] = None  # ruta opcional a archivo o paquete con todos los modelos
    debug: bool = False  # si True, imprime todas las sentencias SQL ejecutadas
    logger: Optional[Callable[[str], None]] = None  # logger opcional para debug
    # Health-check / pool repair
    health_check_enabled: bool = False
    health_check_interval: float = 30.0
    health_check_retries: int = 1
    health_check_timeout: float = 5.0
    # Connection lifecycle
    pool_pre_ping: bool = False
    pool_recycle: Optional[float] = None  # seconds
    max_retries: int = 0
    retry_delay: float = 0.0
    # --- NUEVO: métricas / tracer / logging estructurado ---
    enable_metrics: bool = True
    enable_query_tracer: bool = True
    query_trace_size: int = 200
    enable_structured_logging: bool = True

    def __post_init__(self):
        """Parse DSN if provided and populate connection fields."""
        if self.dsn:
            from urllib.parse import urlparse, unquote
            
            # Try to parse as URL
            try:
                parsed = urlparse(self.dsn)
                
                # Extract username if not already set
                if parsed.username and not self.username:
                    self.username = unquote(parsed.username)
                
                # Extract password if not already set
                if parsed.password and not self.password:
                    self.password = unquote(parsed.password)
                
                # Extract host if not already set (and not default)
                if parsed.hostname and self.host == "localhost":
                    self.host = parsed.hostname
                
                # Extract port if not already set (and not default)
                if parsed.port and self.port == 5432:
                    self.port = parsed.port
                
                # Extract database if not already set
                if parsed.path and len(parsed.path) > 1 and not self.database:
                    # Remove leading slash
                    self.database = parsed.path.lstrip('/')
                    
            except Exception:
                # If URL parsing fails, keep original DSN as-is
                pass


class Engine:
    """Motor principal del ORM: pooling sync/async, ejecución, DDL y triggers."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        # Pooles separados para sync / async.
        self._pool: Optional[queue.Queue] = None  # sync (psycopg)
        self._async_pool: Optional[asyncpg.pool.Pool] = None  # async (asyncpg)
        self._pool_lock = threading.Lock()
        self._pool_size = 0
        # Hooks for execution pipeline
        self._hooks: dict[str, list] = {
            "before_execute": [],
            "after_execute": [],
        }
        # Middlewares: list of (priority, func, timeout)
        self._middlewares_sync: list[tuple[int, Any, Optional[float]]] = []
        self._middlewares_async: list[tuple[int, Any, Optional[float]]] = []
        # Health monitor handles
        self._health_thread: Optional[threading.Thread] = None
        self._health_thread_stop: threading.Event = threading.Event()
        self._health_task: Optional[asyncio.Task] = None
        # Track async pool sizing for auto-adjust logic
        self._async_pool_max_size: Optional[int] = None
        # Lifecycle tracking
        self._conn_last_used_sync: dict[Any, float] = {}
        self._async_pool_last_recreate: float = 0.0

        # ====================================================
        # NUEVO: estado interno para métricas + tracer
        # ====================================================
        self._metrics_lock = threading.Lock()
        self._metrics_enabled = bool(self.config.enable_metrics)
        self._query_tracer_enabled = bool(self.config.enable_query_tracer)
        self._structured_logging_enabled = bool(self.config.enable_structured_logging)

        # Estructura de métricas:
        #   - total_queries / total_errors
        #   - by_statement: SELECT / INSERT / UPDATE / DELETE / etc.
        #   - by_table: public.users, public.posts, etc.
        self._metrics = {
            "total_queries": 0,
            "total_errors": 0,
            "by_statement": defaultdict(
                lambda: {
                    "count": 0,
                    "errors": 0,
                    "total_duration_ms": 0.0,
                    "total_rows": 0,
                }
            ),
            "by_table": defaultdict(
                lambda: {
                    "count": 0,
                    "errors": 0,
                    "total_duration_ms": 0.0,
                    "total_rows": 0,
                }
            ),
        }

        # Query tracer circular
        self._query_trace_size = int(self.config.query_trace_size or 0)
        if self._query_tracer_enabled and self._query_trace_size > 0:
            self._query_trace: Optional[deque] = deque(
                maxlen=self._query_trace_size
            )
        else:
            self._query_trace = None

        # Flag para no registrar dos veces el middleware interno
        self._logging_middlewares_installed: bool = False
        if (
            self._structured_logging_enabled
            or self._metrics_enabled
            or self._query_tracer_enabled
        ):
            self._install_internal_logging_middlewares()

    # ============================================================
    # Debug helper
    # ============================================================
    def _debug(self, msg: str, *args: Any) -> None:
        """Internal debug printing respecting config.debug and optional logger."""
        if not self.config.debug:
            return
        if args:
            msg = msg.format(*args)
        if self.config.logger:
            try:
                self.config.logger(msg)
            except Exception:
                # Fallback a print si el logger falla
                print(msg)
        else:
            print(msg)

    # ============================================================
    # Helpers privados para logging/metrics/tracer
    # ============================================================
    def _extract_statement_type(self, sql: str) -> str:
        """Intentar extraer el tipo de sentencia: SELECT/INSERT/UPDATE/DELETE/DDL/etc."""
        if not sql:
            return "UNKNOWN"
        first = sql.strip().split(None, 1)[0].upper()
        # Normalización rápida
        if first in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
            return first
        if first in {"CREATE", "ALTER", "DROP"}:
            return "DDL"
        if first in {"BEGIN", "COMMIT", "ROLLBACK"}:
            return "TX"
        return first or "UNKNOWN"

    def _extract_table_name(self, sql: str) -> Optional[str]:
        """Heurística simple para extraer el nombre de tabla principal."""
        if not sql:
            return None
        s = sql.upper()
        # Buscamos FROM / INTO / UPDATE
        # Devolvemos la primera coincidencia razonable
        import re
        patterns = [
            r"\bFROM\s+([^\s;,]+)",
            r"\bINTO\s+([^\s;,]+)",
            r"\bUPDATE\s+([^\s;,]+)",
            r"\bJOIN\s+([^\s;,]+)",
        ]
        candidates = []
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                raw = m.group(1)
                # Eliminar alias rápido: "schema.table AS t" -> "schema.table"
                raw = raw.split("AS", 1)[0].strip()
                # Volver al case original usando length
                start = s.find(raw)
                if start != -1:
                    candidates.append(sql[start:start + len(raw)])
                else:
                    candidates.append(raw)
        return candidates[0] if candidates else None

    def _update_metrics_internal(
        self,
        statement: Optional[str],
        table: Optional[str],
        duration_ms: Optional[float],
        rows_count: Optional[int],
        error: bool,
    ) -> None:
        """Actualizar contadores de métricas internas."""
        if not self._metrics_enabled:
            return
        with self._metrics_lock:
            self._metrics["total_queries"] += 1
            if error:
                self._metrics["total_errors"] += 1

            if statement:
                st = self._metrics["by_statement"][statement]
                st["count"] += 1
                if error:
                    st["errors"] += 1
                if duration_ms is not None:
                    st["total_duration_ms"] += float(duration_ms)
                if rows_count is not None:
                    st["total_rows"] += int(rows_count or 0)

            if table:
                tb = self._metrics["by_table"][table]
                tb["count"] += 1
                if error:
                    tb["errors"] += 1
                if duration_ms is not None:
                    tb["total_duration_ms"] += float(duration_ms)
                if rows_count is not None:
                    tb["total_rows"] += int(rows_count or 0)

    def _append_query_trace_internal(
        self,
        sql: str,
        params: Any,
        duration_ms: Optional[float],
        rows_count: Optional[int],
        statement: Optional[str],
        table: Optional[str],
        error: Optional[BaseException],
        started_at: datetime.datetime,
        finished_at: datetime.datetime,
    ) -> None:
        """Guardar entrada en el tracer de últimas N queries."""
        if not (self._query_tracer_enabled and self._query_trace is not None):
            return
        entry = {
            "sql": sql,
            "params": params,
            "statement": statement,
            "table": table,
            "duration_ms": duration_ms,
            "rows": rows_count,
            "error": repr(error) if error else None,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }
        with self._metrics_lock:
            self._query_trace.append(entry)

    def _install_internal_logging_middlewares(self) -> None:
        """Registrar los middlewares internos de logging/metrics/tracer."""
        if self._logging_middlewares_installed:
            return
        self._logging_middlewares_installed = True

        # PRIORIDAD ALTA para envolver toda la pipeline (incluyendo otros middlewares)
        high_priority = 10_000

        def _logging_middleware_sync(sql, params, call_next):
            import time
            start_ts = datetime.datetime.now(datetime.timezone.utc)
            t0 = time.perf_counter()
            err: Optional[BaseException] = None
            rows = None
            try:
                rows = call_next(sql, params)
                return rows
            except Exception as e:
                err = e
                raise
            finally:
                t1 = time.perf_counter()
                duration_ms = (t1 - t0) * 1000.0
                # Contar filas si es posible
                rows_count: Optional[int] = None
                try:
                    if rows is not None and hasattr(rows, "__len__"):
                        rows_count = len(rows)  # type: ignore[arg-type]
                except Exception:
                    rows_count = None

                statement = self._extract_statement_type(sql)
                table = self._extract_table_name(sql)
                finished_ts = datetime.datetime.now(datetime.timezone.utc)

                # Actualizar métricas y tracer
                self._update_metrics_internal(
                    statement=statement,
                    table=table,
                    duration_ms=duration_ms,
                    rows_count=rows_count,
                    error=err is not None,
                )
                self._append_query_trace_internal(
                    sql=sql,
                    params=params,
                    duration_ms=duration_ms,
                    rows_count=rows_count,
                    statement=statement,
                    table=table,
                    error=err,
                    started_at=start_ts,
                    finished_at=finished_ts,
                )

                # Logging estructurado (solo si debug=True)
                if self._structured_logging_enabled:
                    # Mostrar SQL completo si debug está habilitado
                    if self.config.debug:
                        self._debug("[ENGINE] SQL: {} {}", sql, params or [])
                    
                    self._debug(
                        "[ENGINE|QUERY] stmt={} table={} duration_ms={:.2f} rows={} error={}",
                        statement,
                        table,
                        duration_ms,
                        rows_count,
                        bool(err),
                    )

        async def _logging_middleware_async(sql, params, call_next):
            import time
            start_ts = datetime.datetime.now(datetime.timezone.utc)
            t0 = time.perf_counter()
            err: Optional[BaseException] = None
            rows = None
            try:
                rows = await call_next(sql, params)
                return rows
            except Exception as e:
                err = e
                raise
            finally:
                t1 = time.perf_counter()
                duration_ms = (t1 - t0) * 1000.0
                rows_count: Optional[int] = None
                try:
                    if rows is not None and hasattr(rows, "__len__"):
                        rows_count = len(rows)  # type: ignore[arg-type]
                except Exception:
                    rows_count = None

                statement = self._extract_statement_type(sql)
                table = self._extract_table_name(sql)
                finished_ts = datetime.datetime.now(datetime.timezone.utc)

                self._update_metrics_internal(
                    statement=statement,
                    table=table,
                    duration_ms=duration_ms,
                    rows_count=rows_count,
                    error=err is not None,
                )
                self._append_query_trace_internal(
                    sql=sql,
                    params=params,
                    duration_ms=duration_ms,
                    rows_count=rows_count,
                    statement=statement,
                    table=table,
                    error=err,
                    started_at=start_ts,
                    finished_at=finished_ts,
                )

                if self._structured_logging_enabled:
                    # Mostrar SQL completo si debug está habilitado
                    if self.config.debug:
                        self._debug("[ENGINE] SQL: {} {}", sql, params or [])
                    
                    self._debug(
                        "[ENGINE|QUERY] stmt={} table={} duration_ms={:.2f} rows={} error={}",
                        statement,
                        table,
                        duration_ms,
                        rows_count,
                        bool(err),
                    )

        # Registrar como middlewares normales (siguen siendo "internos")
        self.add_middleware_sync(_logging_middleware_sync, priority=high_priority)
        self.add_middleware_async(_logging_middleware_async, priority=high_priority)

    # ============================================================
    # API pública de métricas y tracer
    # ============================================================
    def get_query_metrics(self) -> dict[str, Any]:
        """Devolver snapshot de métricas (copias de los contadores internos)."""
        if not self._metrics_enabled:
            return {
                "enabled": False,
                "total_queries": 0,
                "total_errors": 0,
                "by_statement": {},
                "by_table": {},
            }
        with self._metrics_lock:
            by_stmt = {
                k: dict(v) for k, v in self._metrics["by_statement"].items()
            }
            by_table = {
                k: dict(v) for k, v in self._metrics["by_table"].items()
            }
            return {
                "enabled": True,
                "total_queries": self._metrics["total_queries"],
                "total_errors": self._metrics["total_errors"],
                "by_statement": by_stmt,
                "by_table": by_table,
            }

    def reset_query_metrics(self) -> None:
        """Resetear contadores de métricas (útil en tests o en desarrollo)."""
        with self._metrics_lock:
            self._metrics["total_queries"] = 0
            self._metrics["total_errors"] = 0
            self._metrics["by_statement"].clear()
            self._metrics["by_table"].clear()

    def get_query_trace(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Devolver las últimas N queries registradas por el tracer."""
        if not (self._query_tracer_enabled and self._query_trace is not None):
            return []
        with self._metrics_lock:
            data = list(self._query_trace)
        if limit is not None and limit >= 0:
            return data[-limit:]
        return data

    # ============================================================
    # DSN helpers (eliminan duplicación)
    # ============================================================
    def _build_sync_dsn(self, *, database_override: Optional[str] = None, admin: bool = False) -> str:
        """Construir DSN estilo psycopg."""
        if self.config.dsn and not admin:
            return self.config.dsn
        dbname = (
            "postgres"
            if admin
            else (database_override or self.config.database or "postgres")
        )
        return (
            f"dbname={dbname} user={self.config.username or ''} "
            f"password={self.config.password or ''} host={self.config.host} port={self.config.port}"
        )

    def _build_async_dsn(self, *, database_override: Optional[str] = None) -> str:
        """Construir DSN estilo asyncpg."""
        # BUGFIX: respetar database_override incluso cuando se pasa un DSN completo.
        if self.config.dsn:
            if database_override:
                from urllib.parse import urlparse, urlunparse
                try:
                    parsed = urlparse(self.config.dsn)
                    new_path = f"/{database_override}"
                    parsed = parsed._replace(path=new_path)
                    return urlunparse(parsed)
                except Exception:
                    # Si algo falla al parsear, mantener el DSN original
                    return self.config.dsn
            return self.config.dsn
        dbname = database_override or self.config.database or "postgres"
        return (
            f"postgresql://{self.config.username or ''}:{self.config.password or ''}"
            f"@{self.config.host}:{self.config.port}/{dbname}"
        )

    # ============================================================
    # Gestión de conexiones sync
    # ============================================================
    def _init_sync_pool(self) -> None:
        """Inicializar el pool síncrono con pool_size conexiones."""
        if self._pool is not None:
            return
        self._pool = queue.Queue()
        for _ in range(self.config.pool_size):
            conn = self._create_sync_connection()
            self._pool.put(conn)
            self._pool_size += 1
        self._debug("[ENGINE] sync pool initialized with size={}", self._pool_size)

    def _repair_sync_pool(self) -> None:
        """Verificar todas las conexiones del pool y recrear las que estén cerradas."""
        if self._pool is None:
            return
        with self._pool_lock:
            conns = []
            while not self._pool.empty():
                try:
                    conns.append(self._pool.get_nowait())
                except queue.Empty:
                    break
            repaired = 0
            for i, conn in enumerate(conns):
                try:
                    closed = getattr(conn, "closed", 1)
                except Exception:
                    closed = 1
                if closed:
                    try:
                        conns[i] = self._create_sync_connection()
                        repaired += 1
                    except Exception:
                        # leave original if cannot recreate
                        conns[i] = conn
            for conn in conns:
                self._pool.put(conn)
            if repaired and self.config.debug:
                self._debug("[ENGINE] sync pool repaired: {} connections recreated", repaired)

    def _create_sync_connection(self):
        """Crear una conexión psycopg para el pool sync."""
        dsn = self._build_sync_dsn()
        self._debug("[ENGINE] CONNECT (pool sync) DSN={}", dsn)
        try:
            return psycopg.connect(dsn)
        except psycopg.OperationalError as e:
            error_msg = str(e).lower()
            if "database" in error_msg and "does not exist" in error_msg:
                db_name = self.config.database or "unknown"
                raise DatabaseNotFoundError(db_name, e) from e
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    def acquire_sync(self):
        """Obtener una conexión del pool (versión síncrona)."""
        if self.config.async_:
            raise RuntimeError("Engine configurado en modo async; use acquire()")
        if self._pool is None:
            self._init_sync_pool()

        timeout = self.config.connection_timeout
        try:
            conn = self._pool.get(timeout=timeout)
        except queue.Empty:
            # auto-ajuste si está habilitado
            if self.config.auto_adjust_pool_size:
                with self._pool_lock:
                    max_pool = self.config.max_pool_size or (self.config.pool_size * 4)
                    if self._pool_size < max_pool:
                        conn = self._create_sync_connection()
                        self._pool_size += 1
                        self._debug(
                            "[ENGINE] sync pool resize: new_size={} max={}",
                            self._pool_size,
                            max_pool,
                        )
                        return conn
            raise TimeoutError("Timeout acquiring connection from pool")

        # Pre-ping y recycle
        import time
        now = time.time()
        if self.config.pool_recycle and self.config.pool_recycle > 0:
            last = self._conn_last_used_sync.get(conn, 0.0)
            if (now - last) >= float(self.config.pool_recycle):
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._create_sync_connection()
        if self.config.pool_pre_ping:
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._create_sync_connection()

        return conn

    def release_sync(self, conn) -> None:
        """Devolver una conexión al pool (versión síncrona)."""
        if self.config.async_:
            raise RuntimeError("Engine configurado en modo async; use release()")
        if self._pool is None:
            return
        import time
        self._conn_last_used_sync[conn] = time.time()
        self._pool.put(conn)

    # ============================================================
    # Gestión de conexiones async
    # ============================================================
    async def acquire(self):
        """Obtener una conexión del pool (versión asíncrona, asyncpg)."""
        if not self.config.async_:
            raise RuntimeError("Engine configurado en modo sync; use acquire_sync()")

        if self._async_pool is None:
            await self._init_async_pool()

        assert self._async_pool is not None
        # Async recycle
        if self.config.pool_recycle and self.config.pool_recycle > 0:
            import time
            if (time.time() - self._async_pool_last_recreate) >= float(self.config.pool_recycle):
                await self._recreate_async_pool(
                    self._async_pool_max_size or (self.config.max_pool_size or self.config.pool_size)
                )

        timeout = self.config.connection_timeout
        try:
            if timeout and timeout > 0:
                conn = await asyncio.wait_for(self._async_pool.acquire(), timeout=timeout)
            else:
                conn = await self._async_pool.acquire()
            # Pre-ping
            if self.config.pool_pre_ping:
                try:
                    await asyncio.wait_for(
                        conn.fetchrow("SELECT 1"),
                        timeout=self.config.health_check_timeout or 5.0,
                    )
                except Exception:
                    try:
                        await self._async_pool.release(conn)
                    except Exception:
                        try:
                            await conn.close()
                        except Exception:
                            pass
                    if timeout and timeout > 0:
                        conn = await asyncio.wait_for(self._async_pool.acquire(), timeout=timeout)
                    else:
                        conn = await self._async_pool.acquire()
            return conn
        except asyncio.TimeoutError:
            if self.config.auto_adjust_pool_size:
                # BUGFIX: ahora devolvemos el resultado
                return await self._async_auto_adjust_and_retry_acquire()
            raise TimeoutError("Timeout acquiring connection from async pool")

    async def release(self, conn) -> None:
        """Devolver una conexión al pool (versión asíncrona)."""
        if not self.config.async_:
            raise RuntimeError("Engine configurado en modo sync; use release_sync()")
        if self._async_pool is None:
            return
        try:
            await self._async_pool.release(conn)
        except Exception:
            try:
                await conn.close()
            except Exception:
                pass

    async def _init_async_pool(self) -> None:
        """Inicializar el pool asyncpg para modo asíncrono."""
        if self._async_pool is not None:
            return

        dsn = self._build_async_dsn()
        self._debug("[ENGINE] ASYNC POOL CREATE DSN={}", dsn)
        try:
            max_size = self.config.max_pool_size or self.config.pool_size
            self._async_pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=max_size,
                timeout=self.config.connection_timeout,
            )
            self._async_pool_max_size = max_size
            import time
            self._async_pool_last_recreate = time.time()
        except asyncpg.InvalidCatalogNameError as e:
            db_name = self.config.database or "unknown"
            raise DatabaseNotFoundError(db_name, e) from e
        except Exception as e:
            raise ConnectionError(f"Failed to create async pool: {e}") from e

    async def _async_auto_adjust_and_retry_acquire(self):
        """Attempt to auto-increase async pool size up to max_pool_size and retry acquire once."""
        current_max = self._async_pool_max_size or (self.config.max_pool_size or self.config.pool_size)
        hard_max = self.config.max_pool_size or (self.config.pool_size * 4)
        if current_max >= hard_max:
            raise TimeoutError("Timeout acquiring connection from async pool (max size reached)")

        new_max = min(current_max + 1, hard_max)
        try:
            await self._recreate_async_pool(new_max)
        except Exception:
            raise TimeoutError("Timeout acquiring connection from async pool (resize failed)")

        assert self._async_pool is not None
        timeout = self.config.connection_timeout
        try:
            if timeout and timeout > 0:
                return await asyncio.wait_for(self._async_pool.acquire(), timeout=timeout)
            else:
                return await self._async_pool.acquire()
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout acquiring connection from async pool")

    async def _recreate_async_pool(self, new_max_size: int) -> None:
        """Close current async pool and create a new one with the provided max_size."""
        if self._async_pool is not None:
            try:
                await self._async_pool.close()
            except Exception:
                pass
            self._async_pool = None

        dsn = self._build_async_dsn()
        self._async_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=new_max_size,
            timeout=self.config.connection_timeout,
        )
        self._async_pool_max_size = new_max_size
        self._debug("[ENGINE] async pool resize: new_max_size={}", new_max_size)
        import time
        self._async_pool_last_recreate = time.time()

    # ============================================================
    # Context managers
    # ============================================================
    def connection(self) -> ContextManager[Any]:
        """Context manager síncrono para adquirir/liberar conexión."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            conn = self.acquire_sync()
            try:
                yield conn
            finally:
                self.release_sync(conn)

        return _cm()

    def connection_async(self) -> AsyncContextManager[Any]:
        """Context manager asíncrono para adquirir/liberar conexión."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acm():
            conn = await self.acquire()
            try:
                yield conn
            finally:
                await self.release(conn)

        return _acm()

    # ============================================================
    # Health monitor helpers
    # ============================================================
    def _health_monitor_thread(self) -> None:
        """Background thread that periodically inspects the sync pool."""
        interval = max(0.1, float(self.config.health_check_interval or 0.0))
        stop_evt = self._health_thread_stop
        while not stop_evt.wait(interval):
            try:
                self._repair_sync_pool()
            except Exception:
                self._debug("[ENGINE] health monitor error (sync)")

    def start_health_monitor(self) -> None:
        """Start health monitor for sync engines."""
        if not self.config.health_check_enabled or self.config.async_:
            return
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_thread_stop.clear()
        t = threading.Thread(target=self._health_monitor_thread, daemon=True)
        self._health_thread = t
        t.start()

    async def start_health_monitor_async(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start async health-monitor task for async engines."""
        if not self.config.health_check_enabled or not self.config.async_:
            return
        if self._health_task and not self._health_task.done():
            return
        loop = loop or asyncio.get_running_loop()

        async def _task():
            interval = max(0.1, float(self.config.health_check_interval or 0.0))
            while True:
                try:
                    await self._repair_async_pool()
                except Exception:
                    self._debug("[ENGINE] health monitor error (async)")
                await asyncio.sleep(interval)

        self._health_task = loop.create_task(_task())

    async def _repair_async_pool(self) -> None:
        """Try to ping the async pool; if broken, recreate it."""
        if self._async_pool is None:
            return
        try:
            async with self._async_pool.acquire() as conn:
                try:
                    await asyncio.wait_for(
                        conn.fetchrow("SELECT 1"),
                        timeout=self.config.health_check_timeout,
                    )
                except asyncio.TimeoutError:
                    raise
        except Exception:
            try:
                await self._async_pool.close()
            except Exception:
                pass
            self._async_pool = None
            try:
                await self._init_async_pool()
            except Exception:
                self._debug("[ENGINE] async pool repair failed")

    def stop_health_monitor(self) -> None:
        """Stop the sync health monitor thread if running."""
        if self._health_thread:
            self._health_thread_stop.set()
            try:
                self._health_thread.join(timeout=1.0)
            except Exception:
                pass
            self._health_thread = None

    def dispose(self) -> None:
        """Gracefully close all pool connections (sync)."""
        # stop health monitor
        self.stop_health_monitor()
        # Close sync pool connections
        if self._pool is not None:
            conns = []
            try:
                while not self._pool.empty():
                    conns.append(self._pool.get_nowait())
            except Exception:
                pass
            for c in conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._pool = None
            self._pool_size = 0
            self._debug("[ENGINE] sync pool disposed")

    async def stop_health_monitor_async(self) -> None:
        """Stop the async health monitor task if running and close async pool."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except Exception:
                pass
            self._health_task = None
        if self._async_pool is not None:
            try:
                await self._async_pool.close()
            except Exception:
                pass
            self._async_pool = None
            self._debug("[ENGINE] async pool disposed")

    async def dispose_async(self) -> None:
        """Gracefully dispose async resources."""
        await self.stop_health_monitor_async()

    # ============================================================
    # Health metrics
    # ============================================================
    def health_check(self) -> dict[str, Any]:
        """Return basic health metrics for the engine and pools (sync-safe)."""
        data: dict[str, Any] = {
            "mode": "async" if self.config.async_ else "sync",
            "database": self.config.database,
        }
        if not self.config.async_:
            current_size = self._pool_size
            max_size = self.config.max_pool_size or (self.config.pool_size * 4)
            if self._pool is not None:
                try:
                    idle = self._pool.qsize()
                except Exception:
                    idle = None
                active = current_size - (idle or 0)
            else:
                idle = 0
                active = 0
            data.update(
                {
                    "pool_size": current_size,
                    "max_pool_size": max_size,
                    "idle": idle,
                    "active": active,
                }
            )
        else:
            data.update(
                {
                    "max_pool_size": self._async_pool_max_size
                    or (self.config.max_pool_size or self.config.pool_size),
                    "details": "use health_check_async() for async pool metrics",
                }
            )
        return data

    async def health_check_async(self) -> dict[str, Any]:
        """Return detailed async pool metrics (requires async engine)."""
        if not self.config.async_:
            raise RuntimeError("Engine configurado en modo sync; use health_check()")
        if self._async_pool is None:
            await self._init_async_pool()
        assert self._async_pool is not None
        data: dict[str, Any] = {
            "mode": "async",
            "database": self.config.database,
        }
        try:
            size = self._async_pool.get_size()  # type: ignore[attr-defined]
            max_size = self._async_pool.get_max_size()  # type: ignore[attr-defined]
            idle = self._async_pool.get_idle_count()  # type: ignore[attr-defined]
        except Exception:
            size = None
            max_size = self._async_pool_max_size or (
                self.config.max_pool_size or self.config.pool_size
            )
            idle = None
        data.update({"pool_size": size, "max_pool_size": max_size, "idle": idle})
        if isinstance(size, int) and isinstance(idle, int):
            data["active"] = max(0, size - idle)
        return data

    # Metrics alias + logger
    def metrics(self) -> dict[str, Any]:
        return self.health_check()

    async def metrics_async(self) -> dict[str, Any]:
        return await self.health_check_async()

    def start_metrics_logger(self, interval: float = 30.0) -> None:
        """Start a background thread that prints health metrics periodically (debug-style)."""
        def _worker():
            import time
            while True:
                try:
                    m = self.health_check()
                    print(f"[ENGINE] metrics: {m}")
                except Exception:
                    pass
                time.sleep(max(0.1, float(interval or 0.0)))

        if getattr(self, "_metrics_thread", None) and getattr(
            self._metrics_thread, "is_alive", lambda: False
        )():
            return
        t = threading.Thread(target=_worker, daemon=True)
        self._metrics_thread = t  # type: ignore[attr-defined]
        t.start()

    async def start_metrics_logger_async(self, interval: float = 30.0) -> None:
        """Start a background asyncio task that prints health metrics periodically (async)."""
        if getattr(self, "_metrics_task", None) and not self._metrics_task.done():  # type: ignore
            return

        async def _worker():
            while True:
                try:
                    m = await self.health_check_async()
                    print(f"[ENGINE] metrics (async): {m}")
                except Exception:
                    pass
                await asyncio.sleep(max(0.1, float(interval or 0.0)))
        
        loop = asyncio.get_running_loop()
        self._metrics_task = loop.create_task(_worker())  # type: ignore

    # ============================================================
    # Ejecución sync
    # ============================================================
    def execute(self, query_or_sql: Any, *params: Any, **kwargs: Any) -> Any:
        """Ejecutar una operación en modo síncrono, usando parametrización segura."""
        if self.config.async_:
            raise RuntimeError("Engine configurado en modo async; use execute_async()")
        if hasattr(query_or_sql, "to_sql_params"):
            sql, query_params = query_or_sql.to_sql_params()
        elif hasattr(query_or_sql, "to_sql"):
            sql = query_or_sql.to_sql()
            query_params = params
        else:
            sql = str(query_or_sql)
            query_params = params
        return self._run_sync_pipeline(sql, query_params)

    def _execute_core_sync(self, sql: str, query_params: Any):
        """Core synchronous executor."""
        self._debug("[ENGINE] SQL: {} {}", sql, query_params)

        # Hooks before_execute
        for h in list(self._hooks.get("before_execute", [])):
            try:
                h(sql, query_params)
            except Exception:
                self._debug("[ENGINE] before_execute hook error")

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, query_params or None)
            try:
                rows = cur.fetchall()
            except psycopg.ProgrammingError:
                rows = None
            cur.close()
            # after_execute hooks
            for h in list(self._hooks.get("after_execute", [])):
                try:
                    h(sql, rows)
                except Exception:
                    self._debug("[ENGINE] after_execute hook error")
            return rows

    # ============================================================
    # NUEVO: helper raw para Relation (sync only)
    # ============================================================
    def execute_raw(self, sql: str, params: Optional[list[Any]] = None) -> list[tuple]:
        """
        Ejecutar SQL crudo y devolver siempre una lista de filas (list[tuple]).

        Usado internamente por el sistema de relaciones (many_to_many) para
        consultar tablas de unión sin pasar por el QueryBuilder.

        - Solo soporta modo sync (Engine.async_ == False).
        - Usa el pool existente a través de self.connection().
        """
        if self.config.async_:
            raise RuntimeError("execute_raw solo está disponible en Engine síncrono")

        self._debug("[ENGINE] RAW SQL: {} {}", sql, params)

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params or None)
            try:
                rows = cur.fetchall()
            except psycopg.ProgrammingError:
                rows = []
            conn.commit()
            cur.close()
            return rows

    async def execute_raw_async(self, sql: str, params: Optional[list[Any]] = None) -> list[tuple]:
        """
        Ejecutar SQL crudo y devolver siempre una lista de filas (async).
        
        - Solo soporta modo async (Engine.async_ == True).
        """
        if not self.config.async_:
            raise RuntimeError("execute_raw_async solo está disponible en Engine asíncrono")

        self._debug("[ENGINE] RAW SQL (async): {} {}", sql, params)

        if self._async_pool is None:
            await self._init_async_pool()

        # Convert placeholders %s -> $n
        if "%s" in sql:
            idx = 1
            while "%s" in sql:
                sql = sql.replace("%s", f"${idx}", 1)
                idx += 1

        assert self._async_pool is not None
        async with self._async_pool.acquire() as conn:
            # asyncpg .fetch returns Record objects (dict-like)
            records = await conn.fetch(sql, *params) if params else await conn.fetch(sql)
            return [tuple(r.values()) for r in records]

    def _run_sync_pipeline(self, sql: str, query_params: Any):
        """Run sync middlewares + core executor."""
        import signal

        def final(s, p):
            return self._execute_core_sync(s, p)

        chain = final
        # BUGFIX: prioridad más alta envuelve a las demás (se ejecuta primero).
        sorted_mws = sorted(self._middlewares_sync, key=lambda x: x[0])
        for priority, mw_func, timeout in sorted_mws:
            prev = chain

            def make_mw(mw, prev_func, mw_timeout):
                def _wrapped(s, p):
                    # BUGFIX: evitar señales en hilos secundarios o plataformas sin setitimer.
                    use_alarm = (
                        mw_timeout is not None
                        and mw_timeout > 0
                        and hasattr(signal, "setitimer")
                        and threading.current_thread() is threading.main_thread()
                    )
                    if use_alarm:
                        def _timeout_handler(signum, frame):
                            raise TimeoutError(
                                f"Middleware timed out after {mw_timeout}s"
                            )
                        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                        try:
                            signal.setitimer(signal.ITIMER_REAL, mw_timeout)
                            return mw(s, p, prev_func)
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, old_handler)
                    else:
                        return mw(s, p, prev_func)

                return _wrapped

            chain = make_mw(mw_func, prev, timeout)

        # Retry wrapper
        def _retry_run():
            attempts = max(0, int(self.config.max_retries or 0))
            delay = float(self.config.retry_delay or 0.0)
            last_exc = None
            for i in range(attempts + 1):
                try:
                    return chain(sql, query_params)
                except Exception as e:
                    last_exc = e
                    if i < attempts and delay > 0:
                        import time
                        time.sleep(delay)
                        continue
                    raise last_exc

        return _retry_run()

    # Paralelismo sync
    def parallel_execute(self, tasks: list[Any], *, max_workers: Optional[int] = None) -> list[Any]:
        """Ejecuta múltiples consultas en paralelo (modo sync) usando hilos."""
        if self.config.async_:
            raise RuntimeError("Engine async; use parallel_execute_async()")
        if not tasks:
            return []

        def _unpack(task):
            if isinstance(task, tuple):
                return task[0], task[1:]
            return task, ()

        def _run(task):
            q, params = _unpack(task)
            return self.execute(q, *params)

        workers = max_workers or min(len(tasks), self.config.pool_size or len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run, t) for t in tasks]
            return [f.result() for f in futures]

    # ============================================================
    # Ejecución async
    # ============================================================
    async def execute_async(self, query_or_sql: Any, *params: Any, **kwargs: Any) -> Any:
        """Ejecutar una operación en modo asíncrono (asyncpg), usando parametrización segura."""
        if not self.config.async_:
            raise RuntimeError("Engine configurado en modo sync; use execute()")

        if hasattr(query_or_sql, "to_sql_params"):
            sql, query_params = query_or_sql.to_sql_params()
        elif hasattr(query_or_sql, "to_sql"):
            sql = query_or_sql.to_sql()
            query_params = params
        else:
            sql = str(query_or_sql)
            query_params = params

        self._debug("[ENGINE] SQL (async): {} {}", sql, query_params)
        return await self._run_async_pipeline(sql, query_params)

    async def _execute_core_async(self, sql: str, query_params: Any):
        """Core async executor."""
        # before_execute hooks (sync o async)
        for h in list(self._hooks.get("before_execute", [])):
            try:
                res = h(sql, query_params)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                self._debug("[ENGINE] before_execute hook error")

        if self._async_pool is None:
            await self._init_async_pool()

        # Convert psycopg-style %s placeholders to asyncpg-style $1, $2, $3
        # This is necessary because query builders generate %s but asyncpg expects $N
        if "%s" in sql:
            idx = 1
            while "%s" in sql:
                sql = sql.replace("%s", f"${idx}", 1)
                idx += 1

        assert self._async_pool is not None
        async with self._async_pool.acquire() as conn:
            stmt = await conn.prepare(sql)
            rows = await stmt.fetch(*query_params) if query_params else await stmt.fetch()
            # after_execute hooks
            for h in list(self._hooks.get("after_execute", [])):
                try:
                    res = h(sql, rows)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    self._debug("[ENGINE] after_execute hook error")
            return rows

    async def _run_async_pipeline(self, sql: str, query_params: Any):
        """Run async middlewares + core executor."""
        async def final(s, p):
            return await self._execute_core_async(s, p)

        chain = final
        # BUGFIX: prioridad más alta envuelve a las demás (se ejecuta primero).
        sorted_mws = sorted(self._middlewares_async, key=lambda x: x[0])
        for priority, mw_func, timeout in sorted_mws:
            prev = chain

            def make_mw(mw, prev_func, mw_timeout):
                async def _wrapped(s, p):
                    if mw_timeout is not None and mw_timeout > 0:
                        try:
                            return await asyncio.wait_for(
                                mw(s, p, prev_func), timeout=mw_timeout
                            )
                        except asyncio.TimeoutError:
                            raise TimeoutError(
                                f"Async middleware timed out after {mw_timeout}s"
                            )
                    else:
                        return await mw(s, p, prev_func)

                return _wrapped

            chain = make_mw(mw_func, prev, timeout)

        attempts = max(0, int(self.config.max_retries or 0))
        delay = float(self.config.retry_delay or 0.0)
        last_exc = None
        for i in range(attempts + 1):
            try:
                return await chain(sql, query_params)
            except Exception as e:
                last_exc = e
                if i < attempts and delay > 0:
                    await asyncio.sleep(delay)
                    continue
                raise last_exc

    async def parallel_execute_async(self, tasks: list[Any], *, max_concurrency: Optional[int] = None) -> list[Any]:
        """Ejecuta múltiples consultas en paralelo (modo async) usando asyncio.gather."""
        if not self.config.async_:
            raise RuntimeError("Engine sync; use parallel_execute()")
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def _run(task):
            q, params = (task[0], task[1:]) if isinstance(task, tuple) else (task, ())
            if semaphore:
                async with semaphore:
                    return await self.execute_async(q, *params)
            return await self.execute_async(q, *params)

        return await asyncio.gather(*[_run(t) for t in tasks])

    # ============================================================
    # Middleware registration
    # ============================================================
    def add_middleware_sync(self, func, *, priority: int = 100, timeout: Optional[float] = None) -> None:
        self._middlewares_sync.append((priority, func, timeout))

    def add_middleware_async(self, func, *, priority: int = 100, timeout: Optional[float] = None) -> None:
        self._middlewares_async.append((priority, func, timeout))

    def remove_middleware_sync(self, func) -> None:
        self._middlewares_sync = [(p, f, t) for p, f, t in self._middlewares_sync if f is not func]

    def remove_middleware_async(self, func) -> None:
        self._middlewares_async = [(p, f, t) for p, f, t in self._middlewares_async if f is not func]

    def clear_middlewares_sync(self) -> None:
        self._middlewares_sync.clear()

    def clear_middlewares_async(self) -> None:
        self._middlewares_async.clear()

    # ============================================================
    # Hooks management
    # ============================================================
    def add_hook(self, name: str, func) -> None:
        if name not in self._hooks:
            raise ValueError(f"Unknown hook '{name}'")
        self._hooks[name].append(func)

    def remove_hook(self, name: str, func) -> None:
        if name not in self._hooks:
            return
        try:
            self._hooks[name].remove(func)
        except ValueError:
            pass

    # ============================================================
    # Transactions
    # ============================================================
    def transaction(self):
        from .transactions import Transaction
        return Transaction(self)

    async def transaction_async(self):
        from .transactions import Transaction
        return Transaction(self)

    # ============================================================
    # Auto-setup: database, tables, triggers
    # ============================================================
    def ensure_database(self) -> None:
        """Crear la base de datos si no existe."""
        dbname = self.config.database
        if not dbname:
            return

        admin_dsn = self._build_sync_dsn(admin=True)
        self._debug("[ENGINE] CONNECT (admin) DSN={}", admin_dsn)

        conn = psycopg.connect(admin_dsn, autocommit=True)
        try:
            cur = conn.cursor()
            self._debug(
                "[ENGINE] SQL: {} {}",
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (dbname,),
            )
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone() is not None
            if not exists:
                sql = f"CREATE DATABASE {dbname}"
                self._debug("[ENGINE] SQL: {}", sql)
                cur.execute(sql)
            cur.close()
        finally:
            conn.close()

    async def ensure_database_async(self) -> None:
        """Crear la base de datos si no existe (async)."""
        dbname = self.config.database
        if not dbname:
            return
             
        admin_dsn = self._build_async_dsn(database_override="postgres")
        self._debug("[ENGINE] CONNECT (admin async) DSN={}", admin_dsn)
        
        try:
            conn = await asyncpg.connect(admin_dsn)
            try:
                exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", dbname)
                if not exists:
                    sql = f"CREATE DATABASE {dbname}"
                    self._debug("[ENGINE] SQL (async): {}", sql)
                    await conn.execute(sql)
            finally:
                await conn.close()
        except Exception as e:
            self._debug("[ENGINE] ensure_database_async error: {}", e)
            # Fallback or re-raise? Logging for now to match safe behavior
            pass

    def ensure_tables(self) -> None:
        """Lanzar EnsureDatabaseTables + DDL + triggers (modo sync)."""
        EnsureDatabaseTables(self)
        self.apply_ddl_plan()
        self._ensure_triggers_sync()

    async def ensure_tables_async(self) -> None:
        """Lanzar EnsureDatabaseTables + DDL + triggers (modo async)."""
        EnsureDatabaseTables(self)
        await self.apply_ddl_plan_async()
        await self._ensure_triggers_async()

    def ensure_migrations(self) -> None:
        """
        Auto-run migrations on startup.
        
        1. Initialize migrations if not already done
        2. Auto-generate migration if schema drift detected  
        3. Apply pending migrations
        """
        from ..migrations import MigrationManager, MigrationConfig
        
        migrations_path = self.config.migrations_path or "./migrations"
        
        config = MigrationConfig(
            migrations_path=migrations_path,
            debug=self.config.debug,
            logger=self.config.logger,
        )
        manager = MigrationManager(self, config)
        
        # Initialize if needed
        status = manager.status()
        if not status.initialized:
            manager.init()
            self._debug("[ENGINE] Migrations initialized at {}", migrations_path)
        
        # Check for drift and auto-generate
        try:
            diff = manager._compute_diff()
            if diff.has_changes:
                migration = manager.autogenerate("Auto-generated migration")
                if migration:
                    self._debug("[ENGINE] Auto-generated migration: {}", migration.version)
        except Exception as e:
            self._debug("[ENGINE] Autogenerate skipped: {}", e)
        
        # Apply pending
        count = manager.upgrade()
        if count > 0:
            self._debug("[ENGINE] Applied {} migration(s)", count)

    def check_schema_drift(self) -> None:
        """Check for differences between models and database, emit warning if found.
        
        Note: Only works for sync engines. Async engines skip this check.
        """
        # Skip for async engines - migrations use sync operations
        if self.config.async_:
            self._debug("[ENGINE] Schema drift check skipped for async engine")
            return
        
        import warnings
        from psqlmodel.migrations import MigrationManager, MigrationConfig
        
        migrations_path = self.config.migrations_path or "./migrations"
        
        config = MigrationConfig(
            migrations_path=migrations_path,
            debug=False,  # Suppress debug for this check
            auto_detect_changes=True,
        )
        
        try:
            manager = MigrationManager(self, config)
            diff = manager._compute_diff()
            
            if diff.has_changes:
                # Build detailed list of changes
                changes = []
                for d in diff.new_tables:
                    changes.append(f"+{d.object_name}")
                for d in diff.removed_tables:
                    changes.append(f"-{d.object_name}")
                for d in diff.modified_tables:
                    # Include column details if available
                    col_changes = d.details.get("column_changes", [])
                    if col_changes:
                        col_details = []
                        for c in col_changes[:3]:  # Limit to 3 columns
                            if c["change"] == "type_changed":
                                col_details.append(f"{c['column']}:{c['from']}->{c['to']}")
                            elif c["change"] == "added":
                                col_details.append(f"+{c['column']}")
                            elif c["change"] == "removed":
                                col_details.append(f"-{c['column']}")
                            else:
                                col_details.append(c['column'])
                        changes.append(f"~{d.object_name}({', '.join(col_details)})")
                    else:
                        changes.append(f"~{d.object_name}")
                
                # Limit to first 5 for readability
                if len(changes) > 5:
                    changes_str = ", ".join(changes[:5]) + f" (+{len(changes)-5} more)"
                else:
                    changes_str = ", ".join(changes)
                
                warnings.warn(
                    f"[PSQLMODEL] Schema drift detected: {changes_str}. "
                    f"Run 'python -m psqlmodel migrate autogenerate \"msg\"' to create migration.",
                    UserWarning,
                    stacklevel=3
                )
                self._debug("[ENGINE] Schema drift: {}", changes_str)
        except Exception as e:
            # Silently ignore if migrations not initialized
            self._debug("[ENGINE] Schema drift check skipped: {}", e)


    def _ensure_triggers_sync(self) -> None:
        """Crear triggers para todos los modelos (sync)."""
        discovered_models = getattr(self, "_discovered_models", [])
        plpython_available = self._check_plpython_available_sync()

        if not plpython_available:
            self._debug(
                "[ENGINE] plpython3u not available, using PL/pgSQL fallback for triggers"
            )
            self._debug(
                "[ENGINE] To enable Python execution in triggers, install: sudo apt-get install postgresql-plpython3-16"
            )

        for model_cls, _ in discovered_models:
            if not hasattr(model_cls, "__triggers__"):
                continue
            for trigger in model_cls.__triggers__:
                try:
                    trigger._use_plpython = plpython_available
                    func_sql, trigger_sql = trigger.to_sql()
                    self._execute_ddl_sync(func_sql)
                    for cmd in trigger_sql.split(";"):
                        cmd = cmd.strip()
                        if cmd:
                            self._execute_ddl_sync(cmd + ";")
                    lang = "plpython3u" if plpython_available else "plpgsql"
                    self._debug(
                        "[ENGINE] Created trigger: {} ({})",
                        trigger.trigger_name,
                        lang,
                    )
                except Exception as e:
                    self._debug(
                        "[ENGINE] Warning: Failed to create trigger {}: {}",
                        getattr(trigger, "trigger_name", "?"),
                        e,
                    )

    def _check_plpython_available_sync(self) -> bool:
        """Check if plpython3u extension is available (sync)."""
        try:
            dsn = self._build_sync_dsn()
            conn = psycopg.connect(dsn)
            try:
                cur = conn.cursor()
                cur.execute("CREATE EXTENSION IF NOT EXISTS plpython3u;")
                conn.commit()
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'plpython3u';"
                )
                result = cur.fetchone()
                cur.close()
                return result is not None
            finally:
                conn.close()
        except Exception:
            return False

    def _execute_ddl_sync(self, sql: str) -> None:
        """Ejecutar un comando DDL (sync)."""
        dsn = self._build_sync_dsn()
        conn = psycopg.connect(dsn)
        try:
            cur = conn.cursor()
            self._debug("[ENGINE] SQL DDL: {}...", sql[:100])
            cur.execute(sql)
            conn.commit()
            cur.close()
        finally:
            conn.close()

    async def _ensure_triggers_async(self) -> None:
        """Crear triggers para todos los modelos (async)."""
        discovered_models = getattr(self, "_discovered_models", [])
        plpython_available = await self._check_plpython_available_async()

        if not plpython_available:
            self._debug(
                "[ENGINE] plpython3u not available, using PL/pgSQL fallback for triggers"
            )
            self._debug(
                "[ENGINE] To enable Python execution in triggers, install: sudo apt-get install postgresql-plpython3-16"
            )

        for model_cls, _ in discovered_models:
            if not hasattr(model_cls, "__triggers__"):
                continue
            for trigger in model_cls.__triggers__:
                try:
                    trigger._use_plpython = plpython_available
                    func_sql, trigger_sql = trigger.to_sql()
                    await self._execute_ddl_async(func_sql)
                    for cmd in trigger_sql.split(";"):
                        cmd = cmd.strip()
                        if cmd:
                            await self._execute_ddl_async(cmd + ";")
                    lang = "plpython3u" if plpython_available else "plpgsql"
                    self._debug(
                        "[ENGINE] Created trigger: {} ({})",
                        trigger.trigger_name,
                        lang,
                    )
                except Exception as e:
                    self._debug(
                        "[ENGINE] Warning: Failed to create trigger {}: {}",
                        getattr(trigger, "trigger_name", "?"),
                        e,
                    )

    async def _check_plpython_available_async(self) -> bool:
        """Check if plpython3u extension is available (async)."""
        try:
            conn = await self._get_async_connection()
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS plpython3u;")
                result = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'plpython3u';"
                )
                return result is not None
            finally:
                await conn.close()
        except Exception:
            return False

    async def _execute_ddl_async(self, sql: str) -> None:
        """Ejecutar un comando DDL (async)."""
        conn = await self._get_async_connection()
        try:
            self._debug("[ENGINE] SQL DDL (async): {}...", sql[:100])
            await conn.execute(sql)
        finally:
            await conn.close()

    async def _get_async_connection(self):
        """Obtener una conexión asyncpg para operaciones DDL."""
        dsn = self._build_async_dsn()
        return await asyncpg.connect(dsn)

    # ============================================================
    # Aplicar plan DDL
    # ============================================================
    def apply_ddl_plan(self) -> None:
        """Ejecutar el plan DDL almacenado en _last_ddl_plan."""
        ddl_plan = getattr(self, "_last_ddl_plan", None)
        if not ddl_plan:
            return

        dsn = self._build_sync_dsn()
        self._debug("[ENGINE] CONNECT (DDL) DSN={}", dsn)
        conn = psycopg.connect(dsn)
        try:
            cur = conn.cursor()
            for sql_statement in ddl_plan:
                self._debug("[ENGINE] SQL DDL PLAN: {}", sql_statement)
                cur.execute(sql_statement)
            conn.commit()
            cur.close()
        finally:
            conn.close()

    async def apply_ddl_plan_async(self) -> None:
        """Ejecutar el plan DDL almacenado en _last_ddl_plan (async)."""
        ddl_plan = getattr(self, "_last_ddl_plan", None)
        if not ddl_plan:
            return
        
        conn = await self._get_async_connection()
        try:
            for sql_statement in ddl_plan:
                self._debug("[ENGINE] SQL DDL PLAN (async): {}", sql_statement)
                await conn.execute(sql_statement)
        finally:
            await conn.close()


# ============================================================
# EnsureDatabaseTables – detección de modelos @table y DDL
# (igual que tu versión, sólo reutiliza generate_table_ddl)
# ============================================================

def _iter_python_files(root_dir: str):
    IGNORE_DIRS = {
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "env",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        "*.egg-info",
        "site-packages",
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [
            d for d in dirnames if d not in IGNORE_DIRS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            if name.endswith(".py") and not name.startswith("."):
                yield os.path.join(dirpath, name)


def _import_module_from_path(path: str):
    import sys
    main_file = os.path.abspath(sys.argv[0]) if sys.argv else None
    if main_file and os.path.abspath(path) == main_file:
        return sys.modules.get("__main__")

    abs_path = os.path.abspath(path)
    for mod_name, mod in sys.modules.items():
        if hasattr(mod, "__file__") and mod.__file__:
            if os.path.abspath(mod.__file__) == abs_path:
                return mod

    module_name = os.path.splitext(os.path.basename(path))[0]
    unique_name = f"_psqlmodel_import_{abs(hash(abs_path))}"

    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore
        except Exception:
            return None
        return module
    return None


def _iter_models_in_module(module):
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, type) and issubclass(obj, PSQLModel):
            if hasattr(obj, "__tablename__") and hasattr(obj, "__columns__"):
                yield obj


def EnsureDatabaseTables(engine: Engine) -> None:
    from ..orm.table import generate_table_ddl
    import sys
    import re

    main_file = os.path.abspath(sys.argv[0]) if sys.argv[0] else None
    models = []
    table_registry: dict = {}

    if engine.config.models_path:
        path = engine.config.models_path
        if path in ("__main__", ".", "__file__"):
            if main_file and os.path.isfile(main_file):
                module = _import_module_from_path(main_file)
                if module:
                    for model_cls in _iter_models_in_module(module):
                        models.append((model_cls, main_file))
        elif os.path.isdir(path):
            for fpath in _iter_python_files(path):
                module = _import_module_from_path(fpath)
                if not module:
                    continue
                for model_cls in _iter_models_in_module(module):
                    models.append((model_cls, fpath))
        else:
            module = _import_module_from_path(path)
            if module:
                for model_cls in _iter_models_in_module(module):
                    models.append((model_cls, path))
    else:
        project_root = os.getcwd()
        for fpath in _iter_python_files(project_root):
            module = _import_module_from_path(fpath)
            if not module:
                continue
            for model_cls in _iter_models_in_module(module):
                models.append((model_cls, fpath))

    # Conflictos
    for model_cls, fpath in models:
        schema = getattr(model_cls, "__schema__", "public") or "public"
        table_name = getattr(model_cls, "__tablename__", None)
        if table_name:
            key = (schema, table_name)
            table_registry.setdefault(key, []).append((model_cls, fpath))

    conflicts = {k: v for k, v in table_registry.items() if len(v) > 1}
    if conflicts:
        error_lines = [
            "ERROR: Multiple models with the same table name detected.",
            "This can cause unexpected behavior when creating tables.",
            "",
        ]
        for (schema, table_name), model_list in conflicts.items():
            error_lines.append(f"  Table '{schema}.{table_name}' defined in:")
            for model_cls, fpath in model_list:
                error_lines.append(f"    - {model_cls.__name__} in {fpath}")
        error_lines.append("")
        error_lines.append("SOLUTION: Use the 'models_path' parameter in create_engine()")
        error_lines.append("          to specify the directory or file containing your models.")
        raise ValueError("\n".join(error_lines))

    # Schemas
    schemas = set()
    for model_cls, _ in models:
        schema = getattr(model_cls, "__schema__", "public") or "public"
        schemas.add(schema)

    ddl_statements = [f"CREATE SCHEMA IF NOT EXISTS {schema};" for schema in schemas]

    # Dependencias
    table_keys = []
    table_ddls = {}
    table_deps = {}
    for model_cls, fpath in models:
        schema = getattr(model_cls, "__schema__", "public") or "public"
        table_name = getattr(model_cls, "__tablename__", None)
        key = (schema, table_name)
        table_keys.append(key)
        try:
            statements = generate_table_ddl(model_cls)
            statements = [s for s in statements if not s.startswith("CREATE SCHEMA")]
            deps = set()
            for stmt in statements:
                if stmt.startswith("CREATE TABLE"):
                    for m in re.finditer(r"REFERENCES ([\w]+)\.([\w]+)", stmt):
                        dep_schema, dep_table = m.groups()
                        deps.add((dep_schema, dep_table))
            table_ddls[key] = statements
            table_deps[key] = deps
        except Exception:
            table_ddls[key] = []
            table_deps[key] = set()

    sorted_keys = []
    visited = set()

    def visit(key):
        if key in visited:
            return
        for dep in table_deps.get(key, set()):
            if dep in table_ddls:
                visit(dep)
        visited.add(key)
        sorted_keys.append(key)

    for key in table_keys:
        visit(key)

    for key in sorted_keys:
        ddl_statements.extend(table_ddls.get(key, []))

    engine._last_ddl_plan = ddl_statements  # type: ignore[attr-defined]
    engine._discovered_models = models  # type: ignore[attr-defined]

    junction_tables = _generate_junction_tables(models)
    ddl_statements.extend(junction_tables)


def _generate_junction_tables(models: list) -> list[str]:
    junction_ddl = []
    seen_junctions = set()

    for model_cls, _ in models:
        if not hasattr(model_cls, "__relations__"):
            continue

        for rel_name, relationship in model_cls.__relations__.items():
            if not hasattr(relationship, "secondary") or not relationship.secondary:
                continue

            junction_table = relationship.secondary
            if junction_table in seen_junctions:
                continue
            seen_junctions.add(junction_table)

            relationship._detect_relationship_type()
            if relationship._relationship_type != "many_to_many":
                continue

            owner_table = model_cls.__tablename__
            target_model = relationship._resolve_target()
            if not target_model:
                continue
            target_table = target_model.__tablename__

            # BUGFIX: respetar schema en las FKs de tablas de unión
            owner_schema = getattr(model_cls, "__schema__", "public") or "public"
            target_schema = getattr(target_model, "__schema__", "public") or "public"
            owner_full_table = f"{owner_schema}.{owner_table}"
            target_full_table = f"{target_schema}.{target_table}"

            owner_pk_col = list(model_cls.__columns__.keys())[0]
            target_pk_col = list(target_model.__columns__.keys())[0]

            owner_pk_type = model_cls.__columns__[owner_pk_col].type_hint
            target_pk_type = target_model.__columns__[target_pk_col].type_hint

            owner_sql_type = _python_type_to_sql(owner_pk_type)
            target_sql_type = _python_type_to_sql(target_pk_type)

            owner_fk = f"{_to_singular(owner_table)}_id"
            target_fk = f"{_to_singular(target_table)}_id"

            ddl = f"""CREATE TABLE IF NOT EXISTS {junction_table} (
    {owner_fk} {owner_sql_type} NOT NULL,
    {target_fk} {target_sql_type} NOT NULL,
    PRIMARY KEY ({owner_fk}, {target_fk}),
    FOREIGN KEY ({owner_fk}) REFERENCES {owner_full_table}({owner_pk_col}) ON DELETE CASCADE,
    FOREIGN KEY ({target_fk}) REFERENCES {target_full_table}({target_pk_col}) ON DELETE CASCADE
);"""
            junction_ddl.append(ddl)

    return junction_ddl


def _python_type_to_sql(python_type) -> str:
    if python_type is None:
        return "INTEGER"

    # BUGFIX: detección robusta del nombre de tipo + case-insensitive
    if hasattr(python_type, "__name__"):
        # Clases y tipos (p.ej. uuid.UUID, int)
        type_name = python_type.__name__
    elif hasattr(python_type, "__class__"):
        # Instancias de tipos (p.ej. UUID(), Integer())
        type_name = python_type.__class__.__name__
    else:
        type_name = str(python_type)

    type_name = type_name.lower()

    type_map = {
        "uuid": "UUID",
        "integer": "INTEGER",
        "bigint": "BIGINT",
        "smallint": "SMALLINT",
        "serial": "SERIAL",
        "bigserial": "BIGSERIAL",
        "varchar": "VARCHAR",
        "text": "TEXT",
        "boolean": "BOOLEAN",
        "timestamptz": "TIMESTAMP WITH TIME ZONE",
    }

    return type_map.get(type_name, "INTEGER")


def _to_singular(table_name: str) -> str:
    if table_name.endswith("s"):
        return table_name[:-1]
    return table_name


# ============================================================
# create_engine – punto de entrada unificado
# ============================================================

def create_engine(
    dsn: Optional[str] = None,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    host: str = "localhost",
    port: int = 5432,
    database: Optional[str] = None,
    async_: bool = False,
    pool_size: int = 20,
    auto_adjust_pool_size: bool = False,
    max_pool_size: Optional[int] = None,
    connection_timeout: Optional[float] = None,
    ensure_database: bool = True,
    ensure_tables: bool = True,
    ensure_migrations: bool = False,
    check_schema_drift: bool = True,
    migrations_path: Optional[str] = None,
    models_path: Optional[str] = None,
    debug: bool = False,
    # Health-check options
    health_check_enabled: bool = False,
    health_check_interval: float = 30.0,
    health_check_retries: int = 1,
    health_check_timeout: float = 5.0,
    # Lifecycle options
    pool_pre_ping: bool = False,
    pool_recycle: Optional[float] = None,
    max_retries: int = 0,
    retry_delay: float = 0.0,
    # Logger opcional
    logger: Optional[Callable[[str], None]] = None,
    # NUEVO: toggles de métricas / tracer / logging estructurado
    enable_metrics: bool = True,
    enable_query_tracer: bool = True,
    query_trace_size: int = 200,
    enable_structured_logging: bool = True,
) -> Engine:
    """Crear e inicializar un Engine."""
    config = EngineConfig(
        dsn=dsn,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        async_=async_,
        pool_size=pool_size,
        auto_adjust_pool_size=auto_adjust_pool_size,
        max_pool_size=max_pool_size,
        connection_timeout=connection_timeout,
        ensure_database=ensure_database,
        ensure_tables=ensure_tables,
        ensure_migrations=ensure_migrations,
        check_schema_drift=check_schema_drift,
        migrations_path=migrations_path,
        models_path=models_path,
        debug=debug,
        logger=logger,
        health_check_enabled=health_check_enabled,
        health_check_interval=health_check_interval,
        health_check_retries=health_check_retries,
        health_check_timeout=health_check_timeout,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        max_retries=max_retries,
        retry_delay=retry_delay,
        enable_metrics=enable_metrics,
        enable_query_tracer=enable_query_tracer,
        query_trace_size=query_trace_size,
        enable_structured_logging=enable_structured_logging,
    )

    engine = Engine(config=config)

    # 1. Crear base de datos si no existe
    if ensure_database:
        engine.ensure_database()

    # 2. Inicializar pool
    if not async_:
        engine._init_sync_pool()
        if engine.config.health_check_enabled:
            try:
                engine.start_health_monitor()
            except Exception:
                engine._debug("[ENGINE] failed to start sync health monitor")
    else:
        if engine.config.health_check_enabled:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(engine.start_health_monitor_async(loop))
                else:
                    loop.run_until_complete(engine.start_health_monitor_async(loop))
            except Exception:
                engine._debug("[ENGINE] failed to start async health monitor")

    # 3. Crear tablas
    if ensure_tables:
        engine.ensure_tables()

    # 4. Auto-run migrations if enabled
    if ensure_migrations:
        engine.ensure_migrations()

    # 5. Check for schema drift (warning only)
    if config.check_schema_drift:
        engine.check_schema_drift()

    return engine
