from __future__ import annotations

"""Session and AsyncSession wrappers around Transaction.

Integran Engine + Transaction + PSQLModel con:
- Context managers sync/async para FastAPI / frameworks.
- Identity map (cache por PK).
- Carga diferida y eager (Include).
- API tipo ORM: add, get, exec, exec_one, exec_scalar, refresh, delete, bulk_*.
- SessionManager con contextvars para que el QueryBuilder use siempre
  la sesión/transacción actual sin pasarla a mano.
"""

from typing import TYPE_CHECKING, Any, List, Optional, TypeVar, Generic, Awaitable
from contextvars import ContextVar, Token

from .transactions import Transaction

if TYPE_CHECKING:  # avoid circular imports at runtime
    from ..orm.model import PSQLModel
    from .engine import Engine

T = TypeVar("T")


# ============================================================
# SessionManager basado en contextvars
# ============================================================

_current_session: ContextVar["Session | None"] = ContextVar(
    "psqlmodel_current_session", default=None
)
_current_async_session: ContextVar["AsyncSession | None"] = ContextVar(
    "psqlmodel_current_async_session", default=None
)


class SessionManager:
    """Punto central para obtener la Session/AsyncSession actual.

    Uso típico dentro del QueryBuilder:

        from psqlmodel.session import SessionManager

        def all(self):
            session = SessionManager.require_current()
            return session.exec(self).all()
    """

    # ---- Sync ----
    @staticmethod
    def current() -> "Session | None":
        return _current_session.get()

    @staticmethod
    def require_current() -> "Session":
        session = _current_session.get()
        if session is None:
            raise RuntimeError(
                "No hay Session activa. Usa 'with Session(engine) as session:' "
                "o una dependencia de FastAPI que cree la sesión."
            )
        return session

    # ---- Async ----
    @staticmethod
    def current_async() -> "AsyncSession | None":
        return _current_async_session.get()

    @staticmethod
    def require_current_async() -> "AsyncSession":
        session = _current_async_session.get()
        if session is None:
            raise RuntimeError(
                "No hay AsyncSession activa. Usa 'async with AsyncSession(engine) as session:' "
                "o una dependencia async que cree la sesión."
            )
        return session


# ============================================================
# Result wrappers
# ============================================================

class QueryResult(Generic[T]):
    """Result wrapper that behaves like a read-only list + helpers."""

    def __init__(self, data: List[T], model_cls: Optional[type] = None):
        self._data = data
        self._model_cls = model_cls

    # List-like behaviour
    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self) -> str:
        return f"QueryResult(len={len(self._data)}, model={getattr(self._model_cls, '__name__', None)})"

    # Helpers
    def all(self) -> List[T]:
        """Return all results as a list."""
        return self._data

    def first(self) -> Optional[T]:
        """Return the first result or None if empty."""
        return self._data[0] if self._data else None

    def one(self) -> T:
        """Return exactly one result or raise an error."""
        if not self._data:
            raise ValueError("No results found")
        if len(self._data) > 1:
            raise ValueError(f"Expected 1 result, got {len(self._data)}")
        return self._data[0]

    def one_or_none(self) -> Optional[T]:
        """Return exactly one result, None if empty, or raise if multiple."""
        if not self._data:
            return None
        if len(self._data) > 1:
            raise ValueError(f"Expected 1 result, got {len(self._data)}")
        return self._data[0]


class AsyncQueryResult(Generic[T]):
    """Async result wrapper that provides .all() / .first() / .one() / .one_or_none()."""

    def __init__(self, coro: Awaitable[List[T]]):
        self._coro = coro
        self._data: Optional[List[T]] = None

    async def _ensure_data(self) -> List[T]:
        """Await the coroutine if not already done."""
        if self._data is None:
            self._data = await self._coro
        return self._data

    async def all(self) -> List[T]:
        """Return all results as a list."""
        return await self._ensure_data()

    async def first(self) -> Optional[T]:
        """Return the first result or None if empty."""
        data = await self._ensure_data()
        return data[0] if data else None

    async def one(self) -> T:
        """Return exactly one result or raise an error."""
        data = await self._ensure_data()
        if not data:
            raise ValueError("No results found")
        if len(data) > 1:
            raise ValueError(f"Expected 1 result, got {len(data)}")
        return data[0]

    async def one_or_none(self) -> Optional[T]:
        """Return exactly one result, None if empty, or raise if multiple."""
        data = await self._ensure_data()
        if not data:
            return None
        if len(data) > 1:
            raise ValueError(f"Expected 1 result, got {len(data)}")
        return data[0]

    def __await__(self):
        """Allow awaiting the result directly (equivalent to .all())."""
        return self._ensure_data().__await__()


# ============================================================
# Synchronous Session
# ============================================================

class Session:
    """Synchronous session wrapper around Transaction.

    - Usa el pool del Engine y una Transaction interna.
    - Soporta identity map.
    - Se publica en un ContextVar vía SessionManager.

    auto_commit / auto_rollback:
    - auto_commit=True: si sales del 'with' sin excepción y no hiciste commit
      manual, hace COMMIT automático.
    - auto_rollback=True: si sales del 'with' con excepción o con auto_commit=False
      y hay TX pendiente, hace ROLLBACK automático.
    - atomic: bool = False.
      Si True, commit() cierra la sesión.
      Si False (default), commit() cierra la transacción pero mantiene la sesión abierta
      para nuevas operaciones (que abrirán una nueva transacción automáticamente).
    """

    # ----------------------------------
    # Métodos de alto nivel (flush/tx)
    # ----------------------------------
    def _ensure_transaction(self) -> None:
        """Asegura que haya una transacción activa. Si no, arranca una nueva."""
        if self._closed:
            raise RuntimeError("Session is closed.")
        
        if self._tx is None:
            if self._atomic:
                 # En modo atomic, si no hay tx es porque se cerró (commit/rollback)
                 # y no se debe reabrir. Pero self._closed debería haberlo atrapado antes.
                 # Por seguridad:
                 raise RuntimeError("Session is not active (atomic mode).")
            
            # Modo no-atomic: abrir nueva transacción
            self._tx = Transaction(self.engine)
            # Heredar hooks de la sesión anterior o engine? 
            # Transaction toma engine. 
            # Los hooks de session no existen, solo en engine o transaction.
            self._tx.__enter__() # BEGIN
    def flush(self) -> None:
        """Flush explícito de modelos registrados en la Transaction actual.

        No hace COMMIT ni ROLLBACK, solo sincroniza los cambios pendientes
        con la base de datos dentro de la misma transacción.
        """
        self._ensure_transaction()
        if self._tx is None or self._tx._conn is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")

        # Reaprovechamos la lógica de Transaction._flush_model,
        # respetando el orden: inserts, updates, deletes.
        self._tx._run_hooks("before_flush")  # type: ignore[attr-defined]
        for model in list(getattr(self._tx, "_inserts", [])):
            self._tx._flush_model(None, model, op="insert")
        for model in list(getattr(self._tx, "_updates", [])):
            self._tx._flush_model(None, model, op="update")
        for model in list(getattr(self._tx, "_deletes", [])):
            self._tx._flush_model(None, model, op="delete")
        self._tx._run_hooks("after_flush")  # type: ignore[attr-defined]

        # Limpiamos colas de la TX para evitar re-flush duplicado
        self._tx._inserts.clear()
        self._tx._updates.clear()
        self._tx._deletes.clear()

    def commit(self) -> None:
        """Commit explícito de la transacción actual.

        - Usa Transaction.__exit__ internamente.
        - Si atomic=True: Cierra la sesión (comportamiento legacy).
        - Si atomic=False (default): Cierra sólo la transacción; la sesión sigue abierta.
        """
        if self._tx is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")

        # Simula salida del context manager sin error → COMMIT
        self._tx.__exit__(None, None, None)
        self._tx = None
        
        if self._atomic:
            self._closed = True

    def rollback(self) -> None:
        """Rollback explícito de la transacción actual.
        
        - Si atomic=True: Cierra la sesión.
        - Si atomic=False (default): Cierra sólo la transacción; la sesión sigue abierta.
        """
        if self._tx is None:
             raise RuntimeError("Session is not active. Use it inside a 'with' block.")

        class _SessionRollback(Exception):
            pass

        self._tx.__exit__(_SessionRollback, _SessionRollback("manual rollback"), None)
        self._tx = None
        
        if self._atomic:
            self._closed = True

    def refresh(self, model: "PSQLModel") -> None:
        """Vuelve a cargar un modelo desde la BD por su PK, dentro de la misma TX."""
        self._ensure_transaction()
        if self._tx is None or self._tx._conn is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")

        pk_name = None
        cols = getattr(model.__class__, "__columns__", {})
        for name, col in cols.items():
            if getattr(col, "primary_key", False):
                pk_name = name
                break
        if pk_name is None:
            raise ValueError("No primary key defined")

        pk_value = getattr(model, pk_name)
        table = getattr(model, "__tablename__")
        schema = getattr(model, "__schema__", "public") or "public"

        sql = f"SELECT * FROM {schema}.{table} WHERE {pk_name} = %s"
        rows = self._tx._execute_sql_in_tx_sync(sql, [pk_value])  # type: ignore[attr-defined]
        if rows:
            row = rows[0]
            # Asumimos orden de columnas igual al de __columns__
            col_names = list(cols.keys())
            for idx, name in enumerate(col_names):
                setattr(model, name, row[idx])

    def delete(self, model: "PSQLModel") -> None:
        """Marca un modelo para DELETE inmediato (no solo register)."""
        self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")
        self._tx.register(model, op="delete")
        # Flush inmediato del delete (opcional, pero coherente con behaviour anterior)
        self._tx._flush_model(None, model, op="delete")

    def bulk_insert(self, models: list) -> None:
        self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")
        self._tx.bulk_insert(models)

    def bulk_update(self, models: list) -> None:
        self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")
        self._tx.bulk_update(models)

    # ----------------------------------
    # Constructor / context manager
    # ----------------------------------
    def __init__(
        self,
        engine: "Engine",
        auto_commit: bool = False,
        auto_rollback: bool = True,
        atomic: bool = False,
    ):
        self.engine = engine
        self._auto_commit = auto_commit
        self._auto_rollback = auto_rollback
        self._atomic = atomic
        
        self._tx: Optional[Transaction] = None
        self._closed = False
        
        # Identity Map: (modelo_cls, pk_value) -> instancia
        self._identity_map: dict = {}

        # Token para contextvar
        self._ctx_token: Optional[Token] = None

    def __enter__(self) -> "Session":
        if self.engine.config.async_:
            raise RuntimeError("Engine async; use AsyncSession(engine) en su lugar.")
        if self._closed:
            raise RuntimeError("Session is already closed.")
        if self._tx is not None:
            raise RuntimeError("Session is already active.")

        self._tx = Transaction(self.engine)
        self._tx.__enter__()  # Acquire conn + BEGIN

        # Publicar en contextvar
        self._ctx_token = _current_session.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Siempre limpiar el contextvar
        if self._ctx_token is not None:
            _current_session.reset(self._ctx_token)
            self._ctx_token = None

        if self._tx is None:
            self._closed = True
            return

        # Si ya hicimos commit/rollback manual, _tx debería ser None.
        # Por seguridad, comprobamos:
        if self._closed:
            self._tx = None
            return

        # Lógica de auto_commit / auto_rollback
        if exc_type is not None:
            # Hay excepción dentro del bloque
            if self._auto_rollback:
                self._tx.__exit__(exc_type, exc, tb)
            else:
                # Seguridad: aunque auto_rollback=False, es peligroso no hacer rollback.
                # Hacemos rollback "silencioso" para no dejar la conexión en estado raro.
                self._tx.__exit__(exc_type, exc, tb)
        else:
            # No hay excepción
            if self._auto_commit:
                # COMMIT normal
                self._tx.__exit__(None, None, None)
            elif self._auto_rollback:
                # Sin commit explícito → rollback implícito
                class _AutoRollback(Exception):
                    pass

                self._tx.__exit__(_AutoRollback, _AutoRollback("auto rollback end-of-context"), None)
            else:
                # auto_commit=False y auto_rollback=False
                # Para no dejar la conexión sucia, hacemos rollback de seguridad.
                class _SafeRollback(Exception):
                    pass

                self._tx.__exit__(_SafeRollback, _SafeRollback("safe rollback (auto_commit=False, auto_rollback=False)"), None)

        self._tx = None
        self._closed = True

    # ----------------------------------
    # API de trabajo con modelos
    # ----------------------------------
    def add(self, model: "PSQLModel") -> None:
        """Registra un modelo en la UoW de la transacción."""
        self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")
        self._tx.register(model)
        try:
            setattr(model, "__session__", self)
        except Exception:
            pass
        self._add_to_cache(model)

    def get(self, model_cls: type, pk_value: Any):
        """Carga por primary key con caché (similar a SQLAlchemy Session.get)."""
        if pk_value is None:
            return None
        key = (model_cls, pk_value)
        if key in self._identity_map:
            return self._identity_map[key]

        # Resolver PK
        pk_name = None
        for name, col in getattr(model_cls, "__columns__", {}).items():
            if getattr(col, "primary_key", False):
                pk_name = name
                break
        if pk_name is None:
            raise ValueError(f"Modelo {model_cls.__name__} no tiene primary key definida.")

        from ..query.builder import Select  # import local para evitar ciclos
        pk_col = getattr(model_cls, pk_name)
        res = self.exec(Select(model_cls).Where(pk_col == pk_value)).first()
        if res:
            self._add_to_cache(res)
        return res

    # ----------------------------------
    # Exec de queries (sync)
    # ----------------------------------
    def exec(self, query: Any, *, params: Sequence[Any] | None = None) -> QueryResult:
        """Execute a query builder (Select, Insert, Update, Delete, etc.) or raw SQL."""
        self._ensure_transaction()
        if self._tx is None or self._tx._conn is None:
            raise RuntimeError("Session is not active. Use it inside a 'with' block.")

        sql: str
        query_params: Sequence[Any]

        # Get SQL and params from query
        if isinstance(query, str):
            # Raw SQL string
            sql = query
            query_params = params or []
        elif hasattr(query, "to_sql_params"):
            sql, query_params = query.to_sql_params()
        elif hasattr(query, "to_sql"):
            sql = query.to_sql()
            query_params = []
        else:
            raise ValueError("Query must be a string or have to_sql_params() or to_sql() method")

        from psqlmodel.query.builder import SelectQuery

        # Caso 1: SELECT * FROM Modelo → usamos pipeline del Engine vía Transaction.
        # Caso 1: SELECT * FROM Modelo → usamos pipeline del Engine vía Transaction.
        if isinstance(query, SelectQuery) and getattr(query, "select_all", False):
            rows = self._tx._execute_sql_in_tx_sync(sql, list(query_params))  # type: ignore[attr-defined]
            rows = rows or []

            result_list: List[Any] = []
            base_model = query.base_model
            col_names = list(getattr(base_model, "__columns__", {}).keys())

            pk_name = None
            for name, col in getattr(base_model, "__columns__", {}).items():
                if getattr(col, "primary_key", False):
                    pk_name = name
                    break

            for row in rows:
                row_dict = dict(zip(col_names, row))
                if pk_name and pk_name in row_dict:
                    key = (base_model, row_dict[pk_name])
                    if key in self._identity_map:
                        result_list.append(self._identity_map[key])
                        continue
                instance = base_model(**row_dict)
                try:
                    setattr(instance, "__session__", self)
                except Exception:
                    pass
                self._add_to_cache(instance)
                result_list.append(instance)

            if getattr(query, "includes", None):
                self._load_includes_sync(result_list, query.includes)

            return QueryResult(result_list, model_cls=base_model)

        # Caso 2: cualquier otra query → usamos cursor directo.
        # FIX: Manual debug log since we are bypassing Transaction wrapper for raw queries
        # to preserve access to cursor.description (Transaction wrapper doesn't return metadata).
        self.engine._debug("[ENGINE] SQL: {} {}", sql, list(query_params) if query_params else [])

        cur = self._tx._conn.cursor()
        try:
            cur.execute(sql, list(query_params) if query_params else None)
            if not cur.description:
                result: List[Any] = []
            else:
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                result = [dict(zip(col_names, row)) for row in rows]
        finally:
            cur.close()

        return QueryResult(result, model_cls=None)

    # ----------------------------------
    # Helpers de relaciones (sync)
    # ----------------------------------
    def _load_includes_sync(self, instances, includes):
        """Carga relaciones declaradas vía Include() para consultas sync."""
        if not instances or not includes:
            return

        from psqlmodel.query.builder import Select, SelectQuery
        from psqlmodel.orm.column import Column

        owner_model = type(instances[0])

        for include_target in includes:
            target_model = None
            custom_query = None
            select_columns = None

            if isinstance(include_target, SelectQuery):
                custom_query = include_target
                if include_target.select_all and hasattr(include_target, "_from_model"):
                    target_model = include_target._from_model
                elif include_target.columns:
                    first_col = include_target.columns[0]
                    expr = first_col.expr if hasattr(first_col, "expr") else first_col
                    if hasattr(expr, "model"):
                        target_model = expr.model
            elif isinstance(include_target, type):
                target_model = include_target
            elif isinstance(include_target, Column):
                target_model = include_target.model
                select_columns = [include_target]
            else:
                continue

            if not target_model:
                continue

            rel_info = self._detect_relationship(owner_model, target_model)
            if not rel_info:
                continue

            rel_type = rel_info["type"]
            fk_name = rel_info["fk_name"]
            attr_name = rel_info["attr_name"]

            # ONE-TO-MANY
            if rel_type == "one_to_many":
                owner_pk = None
                for name, col in getattr(owner_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        owner_pk = name
                        break
                if owner_pk is None:
                    continue
                owner_ids = [getattr(inst, owner_pk) for inst in instances]
                fk_col = getattr(target_model, fk_name, None)
                if fk_col is None:
                    continue

                if custom_query:
                    query = custom_query.Where(fk_col).In(owner_ids)
                else:
                    if select_columns:
                        cols_to_select = list(select_columns)
                        fk_sql = fk_col.to_sql()
                        has_fk = any(getattr(c, "to_sql", lambda: "")() == fk_sql for c in cols_to_select)
                        if not has_fk:
                            cols_to_select.append(fk_col)
                        query = Select(*cols_to_select).Where(fk_col).In(owner_ids)
                    else:
                        query = Select(target_model).Where(fk_col).In(owner_ids)

                related_items = self.exec(query).all()

                grouped: dict[Any, list[Any]] = {}
                for item in related_items:
                    fk_value = None
                    extracted_value = item
                    if isinstance(item, dict):
                        fk_value = item.get(fk_name)
                        if len(item) == 2:
                            for k, v in item.items():
                                if k != fk_name:
                                    extracted_value = v
                                    break
                    else:
                        fk_value = getattr(item, fk_name, None)

                    if fk_value is not None:
                        grouped.setdefault(fk_value, []).append(extracted_value)

                for inst in instances:
                    pk_value = getattr(inst, owner_pk, None)
                    try:
                        setattr(inst, attr_name, grouped.get(pk_value, []))
                    except Exception:
                        pass

            # MANY-TO-ONE / ONE-TO-ONE
            elif rel_type in ("many_to_one", "one_to_one"):
                fk_values = [getattr(inst, fk_name, None) for inst in instances if hasattr(inst, fk_name)]
                fk_values = [v for v in fk_values if v is not None]
                if not fk_values:
                    continue

                target_pk = None
                for name, col in getattr(target_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        target_pk = name
                        break
                if target_pk is None:
                    continue
                pk_col = getattr(target_model, target_pk, None)
                if pk_col is None:
                    continue

                from psqlmodel.query.builder import Select

                if custom_query:
                    query = custom_query.Where(pk_col).In(fk_values)
                else:
                    if select_columns:
                        query = Select(*select_columns).Where(pk_col).In(fk_values)
                    else:
                        query = Select(target_model).Where(pk_col).In(fk_values)

                related_items = self.exec(query).all()

                items_map: dict[Any, Any] = {}
                for item in related_items:
                    pk_value = item.get(target_pk) if isinstance(item, dict) else getattr(item, target_pk, None)
                    if pk_value is not None:
                        items_map[pk_value] = item

                for inst in instances:
                    if hasattr(inst, fk_name):
                        fk_value = getattr(inst, fk_name)
                        try:
                            setattr(inst, attr_name, items_map.get(fk_value))
                        except Exception:
                            pass

            # MANY-TO-MANY
            elif rel_type == "many_to_many":
                owner_pk = None
                for name, col in getattr(owner_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        owner_pk = name
                        break
                if owner_pk is None:
                    continue
                owner_ids = [getattr(inst, owner_pk, None) for inst in instances]
                owner_ids = [v for v in owner_ids if v is not None]
                if not owner_ids:
                    continue

                relations = getattr(owner_model, "__relations__", {})
                rel_obj = None
                for attr_name2, rel_candidate in relations.items():
                    try:
                        target = rel_candidate._resolve_target()
                    except Exception:
                        target = None
                    if target is target_model:
                        rel_obj = rel_candidate
                        break
                junction = getattr(rel_obj, "secondary", None) if rel_obj else None
                if not junction:
                    continue

                def singular(name: str) -> str:
                    return name[:-1] if name.endswith("s") else name

                owner_table = getattr(owner_model, "__tablename__", owner_model.__name__.lower())
                target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())
                owner_fk = f"{singular(owner_table)}_id"
                target_fk = f"{singular(target_table)}_id"

                schema = getattr(owner_model, "__schema__", "public") or "public"
                junction_full = junction if "." in junction else f"{schema}.{junction}"

                cur = self._tx._conn.cursor()
                try:
                    cur.execute(
                        f"SELECT {owner_fk}, {target_fk} FROM {junction_full} WHERE {owner_fk} = ANY(%s)",
                        (owner_ids,),
                    )
                    rows = cur.fetchall()
                finally:
                    cur.close()

                by_owner: dict[Any, list[Any]] = {}
                target_ids = set()
                for row_owner, row_target in rows:
                    by_owner.setdefault(row_owner, []).append(row_target)
                    if row_target is not None:
                        target_ids.add(row_target)

                if not target_ids:
                    for inst in instances:
                        setattr(inst, attr_name, [])
                    continue

                target_pk = None
                for name, col in getattr(target_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        target_pk = name
                        break
                if target_pk is None:
                    continue
                pk_col = getattr(target_model, target_pk, None)
                if pk_col is None:
                    continue

                from psqlmodel.query.builder import Select

                if select_columns:
                    cols_to_select = list(select_columns)
                    fk_sql = pk_col.to_sql()
                    has_pk = any(getattr(c, "to_sql", lambda: "")() == fk_sql for c in cols_to_select)
                    if not has_pk:
                        cols_to_select.append(pk_col)
                    query = Select(*cols_to_select).Where(pk_col).In(target_ids)
                else:
                    query = Select(target_model).Where(pk_col).In(target_ids)

                related_items = self.exec(query).all()

                items_map: dict[Any, Any] = {}
                for item in related_items:
                    pk_value = item.get(target_pk) if isinstance(item, dict) else getattr(item, target_pk, None)
                    if pk_value is not None:
                        items_map[pk_value] = item

                for inst in instances:
                    pk_value = getattr(inst, owner_pk, None)
                    ids_for_owner = by_owner.get(pk_value, [])
                    setattr(inst, attr_name, [items_map[i] for i in ids_for_owner if i in items_map])

    def _detect_relationship(self, owner_model, target_model):
        """Auto-detecta la relación entre dos modelos usando metadatos y heurística."""
        try:
            relations = getattr(owner_model, "__relations__", {})
        except Exception:
            relations = {}

        for attr_name, rel in relations.items():
            try:
                target = rel._resolve_target()
            except Exception:
                target = None
            if target is not target_model:
                continue
            try:
                rel._detect_relationship_type()
            except Exception:
                pass
            rel_type = getattr(rel, "_relationship_type", None)
            fk_name = getattr(rel, "_foreign_key", None)

            owner_table = getattr(owner_model, "__tablename__", owner_model.__name__.lower())
            target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())

            def to_singular(name: str) -> str:
                return name[:-1] if name.endswith("s") else name

            if not fk_name:
                if rel_type in ("many_to_one", "one_to_one"):
                    fk_name = f"{to_singular(target_table)}_id"
                elif rel_type == "one_to_many":
                    fk_name = f"{to_singular(owner_table)}_id"

            return {
                "type": rel_type or "many_to_one",
                "fk_name": fk_name,
                "attr_name": attr_name,
            }

        # Heurística de respaldo
        owner_table = owner_model.__tablename__
        target_table = target_model.__tablename__

        def to_singular(name):
            return name[:-1] if name.endswith("s") else name

        owner_fk_name = f"{to_singular(target_table)}_id"
        if owner_fk_name in getattr(owner_model, "__columns__", {}):
            col = owner_model.__columns__[owner_fk_name]
            is_unique = bool(getattr(col, "unique", False))
            return {
                "type": "one_to_one" if is_unique else "many_to_one",
                "fk_name": owner_fk_name,
                "attr_name": to_singular(target_table),
            }

        target_fk_name = f"{to_singular(owner_table)}_id"
        if target_fk_name in getattr(target_model, "__columns__", {}):
            return {
                "type": "one_to_many",
                "fk_name": target_fk_name,
                "attr_name": target_table,
            }

        return None

    def _add_to_cache(self, model: "PSQLModel") -> None:
        """Guarda la instancia en el identity map si tiene PK."""
        pk_name = None
        for name, col in getattr(model.__class__, "__columns__", {}).items():
            if getattr(col, "primary_key", False):
                pk_name = name
                break
        if pk_name is None:
            return
        pk_val = getattr(model, pk_name, None)
        if pk_val is None:
            return
        key = (model.__class__, pk_val)
        self._identity_map[key] = model

    # ----------------------------------
    # Atajos exec_* y paralelismo
    # ----------------------------------
    def exec_one(self, query: Any) -> Any:
        """Execute a query and return only the first result (or None)."""
        result = self.exec(query)
        return result.first()

    def exec_scalar(self, query: Any) -> Any:
        """Execute a query and return the first column of the first row."""
        first_row = self.exec(query).first()
        if first_row is None:
            return None
        if isinstance(first_row, dict):
            return next(iter(first_row.values())) if first_row else None
        cols = getattr(first_row.__class__, "__columns__", {})
        if cols:
            first_name = next(iter(cols.keys()))
            return getattr(first_row, first_name, None)
        return first_row

    def parallel_exec(self, tasks: list[Any], *, max_workers: int | None = None) -> list[Any]:
        """Ejecuta múltiples consultas en paralelo usando el Engine (sync).

        IMPORTANTE:
        - Usa el pool del Engine, no la conexión de esta Session.
        - Por tanto, NO forma parte de la misma transacción que la Session.
        - Útil para lecturas en paralelo (reporting, dashboards, etc.).
        """
        return self.engine.parallel_execute(tasks, max_workers=max_workers)


# ============================================================
# Asynchronous Session
# ============================================================

class AsyncSession:
    """Asynchronous session wrapper around Transaction.

    Pensada para usar con Engine async:

        async def get_async_session():
            async with AsyncSession(engine) as session:
                yield session
    """

    # ----------------------------------
    # Métodos de alto nivel (flush/tx)
    # ----------------------------------
    async def flush(self) -> None:
        """Flush explícito en modo async."""
        if self._tx is None or self._tx._conn is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")

        await self._tx._run_hooks_async("before_flush")  # type: ignore[attr-defined]
        for model in list(getattr(self._tx, "_inserts", [])):
            await self._tx._flush_model_async(self._tx._conn, model, op="insert")
        for model in list(getattr(self._tx, "_updates", [])):
            await self._tx._flush_model_async(self._tx._conn, model, op="update")
        for model in list(getattr(self._tx, "_deletes", [])):
            await self._tx._flush_model_async(self._tx._conn, model, op="delete")
        await self._tx._run_hooks_async("after_flush")  # type: ignore[attr-defined]

        self._tx._inserts.clear()
        self._tx._updates.clear()
        self._tx._deletes.clear()

    async def commit(self) -> None:
        """Commit explicito (async).

        - Si atomic=True: Cierra la sesión.
        - Si atomic=False (default): Cierra sólo la transacción; la sesión sigue abierta.
        """
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")

        await self._tx.__aexit__(None, None, None)
        self._tx = None
        
        if self._atomic:
            self._closed = True

    async def rollback(self) -> None:
        """Rollback explicito (async).

        - Si atomic=True: Cierra la sesión.
        - Si atomic=False (default): Cierra sólo la transacción; la sesión sigue abierta.
        """
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")

        class _AsyncSessionRollback(Exception):
            pass

        await self._tx.__aexit__(_AsyncSessionRollback, _AsyncSessionRollback("manual rollback"), None)
        self._tx = None
        
        if self._atomic:
            self._closed = True

    async def refresh(self, model: "PSQLModel") -> None:
        """Reload model from DB in same TX (async)."""
        await self._ensure_transaction()
        if self._tx is None or self._tx._conn is None:
            raise RuntimeError("AsyncSession is not active.")

        pk_name = None
        cols = getattr(model.__class__, "__columns__", {})
        for name, col in cols.items():
            if getattr(col, "primary_key", False):
                pk_name = name
                break
        if pk_name is None:
            raise ValueError("No primary key defined")

        pk_value = getattr(model, pk_name)
        table = getattr(model, "__tablename__")
        schema = getattr(model, "__schema__", "public") or "public"
        sql = f"SELECT * FROM {schema}.{table} WHERE {pk_name} = $1"

        rows = await self._tx._execute_sql_in_tx_async(sql, [pk_value])  # type: ignore[attr-defined]
        if rows:
            row = rows[0]
            row_dict = dict(row)
            for name in cols.keys():
                if name in row_dict:
                    setattr(model, name, row_dict[name])

    async def delete(self, model: "PSQLModel") -> None:
        await self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")
        self._tx.register(model, op="delete")
        await self._tx._flush_model_async(self._tx._conn, model, op="delete")

    async def bulk_insert(self, models: list) -> None:
        await self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")
        await self._tx.bulk_insert_async(models)

    async def bulk_update(self, models: list) -> None:
        await self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")
        await self._tx.bulk_update_async(models)

    # ----------------------------------
    # Constructor / context manager
    # ----------------------------------
    def __init__(
        self,
        engine: "Engine",
        auto_commit: bool = False,
        auto_rollback: bool = True,
        atomic: bool = False,
    ):  
        self.engine = engine
        self._tx: Optional[Transaction] = None
        self._closed: bool = False

        self._auto_commit = auto_commit
        self._auto_rollback = auto_rollback
        self._atomic = atomic

        # Identity Map: (modelo_cls, pk_value) -> instancia
        self._identity_map: dict = {}

        self._ctx_token: Optional[Token] = None

    async def __aenter__(self) -> "AsyncSession":
        if not self.engine.config.async_:
            raise RuntimeError("Engine sync; use Session(engine) en su lugar.")
        if self._closed:
            raise RuntimeError("AsyncSession is already closed.")
        if self._tx is not None:
            raise RuntimeError("AsyncSession is already active.")

        self._tx = Transaction(self.engine)
        await self._tx.__aenter__()

        self._ctx_token = _current_async_session.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Siempre limpiar el contextvar
        if self._ctx_token is not None:
            _current_async_session.reset(self._ctx_token)
            self._ctx_token = None

        if self._tx is None:
            self._closed = True
            return

        if self._closed:
            # If closed, just ensure tx is cleaned up
             self._tx = None
             return

        # Lógica auto_commit / auto_rollback
        if exc_type is not None:
            if self._auto_rollback:
                await self._tx.__aexit__(exc_type, exc, tb)
            else:
                # Seguridad: rollback igualmente
                await self._tx.__aexit__(exc_type, exc, tb)
        else:
            if self._auto_commit:
                await self._tx.__aexit__(None, None, None)
            elif self._auto_rollback:
                class _AutoRollback(Exception):
                    pass

                await self._tx.__aexit__(
                    _AutoRollback,
                    _AutoRollback("auto rollback end-of-context"),
                    None,
                )
            else:
                class _SafeRollback(Exception):
                    pass

                await self._tx.__aexit__(
                    _SafeRollback,
                    _SafeRollback("safe rollback (auto_commit=False, auto_rollback=False)"),
                    None,
                )

        self._tx = None
        
        if self._atomic:
            self._closed = True

    async def _ensure_transaction(self) -> None:
        """Asegura que haya una transacción activa (async)."""
        if self._closed:
            raise RuntimeError("AsyncSession is closed.")
        
        if self._tx is None:
            if self._atomic:
                 raise RuntimeError("AsyncSession is not active (atomic mode).")
            
            # Modo no-atomic: abrir nueva transacción
            self._tx = Transaction(self.engine)
            await self._tx.__aenter__()

    # ----------------------------------
    # API de modelos (async)
    # ----------------------------------
    async def add(self, model: "PSQLModel") -> None:
        await self._ensure_transaction()
        if self._tx is None:
            raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")
        self._tx.register(model)
        try:
            setattr(model, "__session__", self)
        except Exception:
            pass
        self._add_to_cache(model)

    async def get(self, model_cls: type, pk_value: Any):
        """Get model by primary key with identity map caching (async)."""
        if pk_value is None:
            return None
        key = (model_cls, pk_value)
        if key in self._identity_map:
            return self._identity_map[key]

        # Resolve PK
        pk_name = None
        for name, col in getattr(model_cls, "__columns__", {}).items():
            if getattr(col, "primary_key", False):
                pk_name = name
                break
        if pk_name is None:
            raise ValueError(f"Modelo {model_cls.__name__} no tiene primary key definida.")

        from ..query.builder import Select
        pk_col = getattr(model_cls, pk_name)
        res = await (await self.exec(Select(model_cls).Where(pk_col == pk_value))).first()
        if res:
            self._add_to_cache(res)
        return res

    # ----------------------------------
    # Helper methods
    # ----------------------------------
    def _add_to_cache(self, model: "PSQLModel") -> None:
        """Add model to identity map."""
        try:
            pk_name = None
            for name, col in getattr(model.__class__, "__columns__", {}).items():
                if getattr(col, "primary_key", False):
                    pk_name = name
                    break
            if pk_name:
                pk_value = getattr(model, pk_name, None)
                if pk_value is not None:
                    key = (model.__class__, pk_value)
                    self._identity_map[key] = model
        except Exception:
            pass

    # ----------------------------------
    # Exec de queries (async)
    # ----------------------------------
    def exec(self, query: Any, *, params: Sequence[Any] | None = None) -> AsyncQueryResult:
        """Execute a query builder (Select, Insert, Update, Delete, etc.) or raw SQL asynchronously."""

        async def _execute() -> List[Any]:
            await self._ensure_transaction()
            if self._tx is None or self._tx._conn is None:
                raise RuntimeError("AsyncSession is not active. Use it inside an 'async with' block.")

            sql: str
            query_params: Sequence[Any]

            if isinstance(query, str):
                # Raw SQL string
                sql = query
                query_params = params or []
            elif hasattr(query, "to_sql_params"):
                sql, query_params = query.to_sql_params()
            elif hasattr(query, "to_sql"):
                sql = query.to_sql()
                query_params = []
            else:
                raise ValueError("Query must be a string or have to_sql_params() or to_sql() method")

            rows = await self._tx._execute_sql_in_tx_async(sql, list(query_params))  # type: ignore[attr-defined]
            rows = rows or []

            from psqlmodel.query.builder import SelectQuery

            if isinstance(query, SelectQuery) and getattr(query, "select_all", False):
                result: List[Any] = []
                seen = set()

                pk_name = None
                for name, col in getattr(query.base_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        pk_name = name
                        break

                for row in rows:
                    row_dict = dict(row)
                    if pk_name and pk_name in row_dict:
                        # Check identity map first
                        key = (query.base_model, row_dict[pk_name])
                        if key in self._identity_map:
                            result.append(self._identity_map[key])
                            continue
                    instance = query.base_model(**row_dict)
                    try:
                        setattr(instance, "__session__", self)
                    except Exception:
                        pass
                    self._add_to_cache(instance)
                    result.append(instance)

                if getattr(query, "includes", None) and result:
                    await self._load_includes_async(result, query.includes)

                return result

            return [dict(r) for r in rows] if rows else []

        return AsyncQueryResult(_execute())

    async def _load_includes_async(self, instances, includes):
        """Load related objects for Include() - auto-detects relationships."""
        if not instances:
            return

        from psqlmodel.query.builder import Select, SelectQuery
        from psqlmodel.orm.column import Column

        owner_model = type(instances[0])

        for include_target in includes:
            target_model = None
            custom_query = None
            select_columns = None

            if isinstance(include_target, SelectQuery):
                custom_query = include_target
                if include_target.select_all and hasattr(include_target, "_from_model"):
                    target_model = include_target._from_model
                elif include_target.columns:
                    first_col = include_target.columns[0]
                    if hasattr(first_col, "model"):
                        target_model = first_col.model
            elif isinstance(include_target, type):
                target_model = include_target
            elif isinstance(include_target, Column):
                target_model = include_target.model
                select_columns = [include_target]
            else:
                continue

            if not target_model:
                continue

            rel_info = self._detect_relationship(owner_model, target_model)
            if not rel_info:
                continue

            rel_type = rel_info["type"]
            fk_name = rel_info["fk_name"]
            attr_name = rel_info["attr_name"]

            # ONE-TO-MANY
            if rel_type == "one_to_many":
                owner_pk = None
                for name, col in getattr(owner_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        owner_pk = name
                        break
                if owner_pk is None:
                    continue
                owner_ids = [getattr(inst, owner_pk) for inst in instances]


                fk_col = getattr(target_model, fk_name)

                if custom_query:
                    # Ensure FK is in the query if it's a multi-column select
                    if hasattr(custom_query, 'columns') and custom_query.columns and not custom_query.select_all:
                        # Check if FK is already in columns
                        fk_sql = fk_col.to_sql()
                        has_fk = any(getattr(c, "to_sql", lambda: "")() == fk_sql for c in custom_query.columns)
                        if not has_fk:
                            # Clone the query with FK added to columns
                            cols_with_fk = list(custom_query.columns) + [fk_col]
                            query = Select(*cols_with_fk)
                            
                            # Clone WHERE clauses
                            if hasattr(custom_query, 'where') and custom_query.where:
                                query.where = list(custom_query.where)
                            # Add FK IN clause
                            query = query.Where(fk_col).In(owner_ids)
                            
                            # Clone ORDER BY
                            if hasattr(custom_query, 'order_by') and custom_query.order_by:
                                query.order_by = list(custom_query.order_by)
                            
                            # Clone LIMIT
                            if hasattr(custom_query, 'limit') and custom_query.limit:
                                query.limit = custom_query.limit
                                
                            # Clone OFFSET
                            if hasattr(custom_query, 'offset') and custom_query.offset:
                                query.offset = custom_query.offset
                        else:
                            query = custom_query.Where(fk_col).In(owner_ids)
                    else:
                        query = custom_query.Where(fk_col).In(owner_ids)
                else:
                    if select_columns:
                        cols_to_select = list(select_columns)
                        fk_sql = fk_col.to_sql()
                        has_fk = any(getattr(c, "to_sql", lambda: "")() == fk_sql for c in cols_to_select)
                        if not has_fk:
                            cols_to_select.append(fk_col)
                        query = Select(*cols_to_select).Where(fk_col).In(owner_ids)
                    else:
                        query = Select(target_model).Where(fk_col).In(owner_ids)

                related_items = await self.exec(query).all()

                grouped: dict[Any, list[Any]] = {}
                for item in related_items:
                    fk_value = None
                    extracted_value = item

                    if isinstance(item, dict):
                        fk_value = item.get(fk_name)
                        if len(item) == 2:
                            for k, v in item.items():
                                if k != fk_name:
                                    extracted_value = v
                                    break
                    else:
                        fk_value = getattr(item, fk_name, None)

                    if fk_value is not None:
                        grouped.setdefault(fk_value, []).append(extracted_value)

                for inst in instances:
                    pk_value = getattr(inst, owner_pk)
                    setattr(inst, attr_name, grouped.get(pk_value, []))


            # MANY-TO-ONE / ONE-TO-ONE
            elif rel_type in ("many_to_one", "one_to_one"):
                fk_values = [getattr(inst, fk_name) for inst in instances if hasattr(inst, fk_name)]
                fk_values = [v for v in fk_values if v is not None]
                if not fk_values:
                    continue

                target_pk = None
                for name, col in getattr(target_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        target_pk = name
                        break
                if target_pk is None:
                    continue
                pk_col = getattr(target_model, target_pk)

                if custom_query:
                    query = custom_query.Where(pk_col).In(fk_values)
                else:
                    if select_columns:
                        query = Select(*select_columns).Where(pk_col).In(fk_values)
                    else:
                        query = Select(target_model).Where(pk_col).In(fk_values)

                related_items = await self.exec(query).all()

                items_map: dict[Any, Any] = {}
                for item in related_items:
                    pk_value = item.get(target_pk) if isinstance(item, dict) else getattr(item, target_pk, None)
                    if pk_value is not None:
                        items_map[pk_value] = item

                for inst in instances:
                    if hasattr(inst, fk_name):
                        fk_value = getattr(inst, fk_name)
                        setattr(inst, attr_name, items_map.get(fk_value))

            # MANY-TO-MANY
            elif rel_type == "many_to_many":
                owner_pk = None
                for name, col in getattr(owner_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        owner_pk = name
                        break
                if owner_pk is None:
                    continue
                owner_ids = [getattr(inst, owner_pk, None) for inst in instances]
                owner_ids = [v for v in owner_ids if v is not None]
                if not owner_ids:
                    continue

                relations = getattr(owner_model, "__relations__", {})
                rel_obj = None
                for attr_name2, rel_candidate in relations.items():
                    try:
                        target = rel_candidate._resolve_target()
                    except Exception:
                        target = None
                    if target is target_model:
                        rel_obj = rel_candidate
                        break
                junction = getattr(rel_obj, "secondary", None) if rel_obj else None
                if not junction:
                    continue

                def singular(name: str) -> str:
                    return name[:-1] if name.endswith("s") else name

                owner_table = getattr(owner_model, "__tablename__", owner_model.__name__.lower())
                target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())
                owner_fk = f"{singular(owner_table)}_id"
                target_fk = f"{singular(target_table)}_id"

                schema = getattr(owner_model, "__schema__", "public") or "public"
                junction_full = junction if "." in junction else f"{schema}.{junction}"

                rows = await self._tx._conn.fetch(  # type: ignore[union-attr]
                    f"SELECT {owner_fk}, {target_fk} FROM {junction_full} WHERE {owner_fk} = ANY($1)",
                    owner_ids,
                )

                by_owner: dict[Any, list[Any]] = {}
                target_ids = set()
                for row in rows:
                    row_owner = row[owner_fk]
                    row_target = row[target_fk]
                    by_owner.setdefault(row_owner, []).append(row_target)
                    if row_target is not None:
                        target_ids.add(row_target)

                if not target_ids:
                    for inst in instances:
                        setattr(inst, attr_name, [])
                    continue

                target_pk = None
                for name, col in getattr(target_model, "__columns__", {}).items():
                    if getattr(col, "primary_key", False):
                        target_pk = name
                        break
                if target_pk is None:
                    continue
                pk_col = getattr(target_model, target_pk, None)
                if pk_col is None:
                    continue

                if select_columns:
                    cols_to_select = list(select_columns)
                    fk_sql = pk_col.to_sql()
                    has_pk = any(getattr(c, "to_sql", lambda: "")() == fk_sql for c in cols_to_select)
                    if not has_pk:
                        cols_to_select.append(pk_col)
                    query = Select(*cols_to_select).Where(pk_col).In(target_ids)
                else:
                    query = Select(target_model).Where(pk_col).In(target_ids)

                related_items = await self.exec(query).all()

                items_map: dict[Any, Any] = {}
                for item in related_items:
                    pk_value = item.get(target_pk) if isinstance(item, dict) else getattr(item, target_pk, None)
                    if pk_value is not None:
                        items_map[pk_value] = item

                for inst in instances:
                    pk_value = getattr(inst, owner_pk, None)
                    ids_for_owner = by_owner.get(pk_value, [])
                    setattr(inst, attr_name, [items_map[i] for i in ids_for_owner if i in items_map])

    def _detect_relationship(self, owner_model, target_model):
        """Auto-detect relationship type between two models."""
        try:
            relations = getattr(owner_model, "__relations__", {})
        except Exception:
            relations = {}

        for attr_name, rel in relations.items():
            try:
                target = rel._resolve_target()
            except Exception:
                target = None
            
            if target is not target_model:
                continue
            
            try:
                rel._detect_relationship_type()
            except Exception:
                pass
            rel_type = getattr(rel, "_relationship_type", None)
            fk_name = getattr(rel, "_foreign_key", None)

            owner_table = getattr(owner_model, "__tablename__", owner_model.__name__.lower())
            target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())

            def to_singular(name: str) -> str:
                return name[:-1] if name.endswith("s") else name

            if not fk_name:
                if rel_type in ("many_to_one", "one_to_one"):
                    fk_name = f"{to_singular(target_table)}_id"
                elif rel_type == "one_to_many":
                    fk_name = f"{to_singular(owner_table)}_id"

            return {
                "type": rel_type or "many_to_one",
                "fk_name": fk_name,
                "attr_name": attr_name,
            }

        owner_table = owner_model.__tablename__
        target_table = target_model.__tablename__

        def to_singular(name):
            return name[:-1] if name.endswith("s") else name

        owner_fk_name = f"{to_singular(target_table)}_id"
        if owner_fk_name in owner_model.__columns__:
            col = owner_model.__columns__[owner_fk_name]
            is_unique = bool(getattr(col, "unique", False))
            return {
                "type": "one_to_one" if is_unique else "many_to_one",
                "fk_name": owner_fk_name,
                "attr_name": to_singular(target_table),
            }

        target_fk_name = f"{to_singular(owner_table)}_id"
        # BUG FIX: Should check target_model.__columns__, not owner_model.__columns__
        if target_fk_name in target_model.__columns__:
            return {
                "type": "one_to_many",
                "fk_name": target_fk_name,
                "attr_name": target_table,
            }

        return None

    # ----------------------------------
    # Atajos exec_* y paralelismo
    # ----------------------------------
    async def exec_one(self, query: Any) -> Any:
        """Execute a query and return only the first result (or None)."""
        result = await self.exec(query).first()
        return result

    async def exec_scalar(self, query: Any) -> Any:
        """Execute a query and return the first column of the first row."""
        first_row = await self.exec(query).first()
        if first_row is None:
            return None
        if isinstance(first_row, dict):
            return next(iter(first_row.values())) if first_row else None
        cols = getattr(first_row.__class__, "__columns__", {})
        if cols:
            first_name = next(iter(cols.keys()))
            return getattr(first_row, first_name, None)
        return first_row

    async def parallel_exec(self, tasks: list[Any], *, max_concurrency: int | None = None) -> list[Any]:
        """Ejecuta múltiples consultas en paralelo usando el Engine (async).

        Igual que en Session.parallel_exec:
        - Usa el pool del Engine, no la conexión de esta AsyncSession.
        - No forma parte de la misma transacción.
        """
        return await self.engine.parallel_execute_async(tasks, max_concurrency=max_concurrency)
