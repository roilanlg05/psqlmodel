
# ============================================================
# subscriber.py — Subscribe 4.0 - Engine Compatible
# ============================================================
# - 100% Engine integration: uses engine pool, middleware, hooks, debug
# - Own connection pool when use_engine_pool=False
# - All parameters in snake_case
# - Leverages transactions.py for DDL
# - Corrected factory API
# ============================================================

import threading
import asyncio
import psycopg
import asyncpg
import select
import json
from uuid import uuid4
from typing import Any, Callable, List, Optional, Dict
from dataclasses import dataclass
import atexit
import inspect

from ..orm.column import Column
from ..core.transactions import Transaction, AsyncTransaction


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SubscriberConfig:
    """Configuration for Subscribe instances."""
    use_engine_pool: bool = True          # Use engine's connection pool
    auto_delete_trigger: bool = True      # Auto-cleanup triggers on Stop()
    daemon_mode: bool = False             # Thread daemon mode
    listen_timeout: float = 1.0           # Timeout for select() in seconds
    connection_retry: int = 3             # Connection retry attempts
    retry_delay: float = 0.5              # Delay between retries (seconds)
    parallel_ddl: bool = False            # Create triggers in parallel (async only)
    max_ddl_concurrency: int = 5          # Max concurrent DDL operations
    own_pool_size: int = 5                # Pool size when use_engine_pool=False


# ============================================================
# EXCEPCIONES
# ============================================================

class ColumnNotFoundError(Exception):
    pass

class TriggerCreationError(Exception):
    pass


# ============================================================
# SUBSCRIBE
# ============================================================

class Subscribe:
    """
    Subscribe 4.0 — Real-time subscription with full Engine integration.
    
    Features:
    - Uses engine's connection pool by default
    - Own pool when use_engine_pool=False
    - Leverages engine middleware/hooks
    - Supports engine debug mode
    - Parallel DDL creation
    - Full async/sync support
    
    Usage:
        sub = Subscribe.engine(engine, use_engine_pool=True)
        sub(User).OnEvent("insert", "delete").Exec(callback).Start()
    """

    @classmethod
    def engine(cls, engine, use_engine_pool: bool = True, 
               auto_delete_trigger: bool = True, daemon_mode: bool = False,
               listen_timeout: float = 1.0, connection_retry: int = 3,
               retry_delay: float = 0.5, parallel_ddl: bool = False,
               max_ddl_concurrency: int = 5, own_pool_size: int = 5,
               config: Optional[SubscriberConfig] = None,
               # Backward compat (deprecated):
               UseEnginePool: Optional[bool] = None,
               AutoDeleteTrigger: Optional[bool] = None):
        """
        Create a Subscribe factory for the given engine.
        
        Args:
            engine: PSQLModel Engine instance
            use_engine_pool: Use engine's pool (True) or own pool (False)
            auto_delete_trigger: Auto-cleanup triggers on Stop()
            daemon_mode: Run listener thread as daemon
            listen_timeout: Select timeout for LISTEN loop
            connection_retry: Number of connection retry attempts
            retry_delay: Delay between retries
            parallel_ddl: Create triggers in parallel (async only)
            max_ddl_concurrency: Max concurrent DDL operations
            own_pool_size: Pool size when use_engine_pool=False
            config: SubscriberConfig object (overrides individual params)
            
            # Deprecated (backward compat):
            UseEnginePool: Use use_engine_pool instead
            AutoDeleteTrigger: Use auto_delete_trigger instead
        
        Returns:
            Subscribe factory instance (call with model to create subscription)
        
        Example:
            sub = Subscribe.engine(engine, use_engine_pool=True)
            sub(User).OnEvent("insert", "delete").Exec(callback).Start()
        """
        if config is None:
            config = SubscriberConfig(
                use_engine_pool=use_engine_pool,
                auto_delete_trigger=auto_delete_trigger,
                daemon_mode=daemon_mode,
                listen_timeout=listen_timeout,
                connection_retry=connection_retry,
                retry_delay=retry_delay,
                parallel_ddl=parallel_ddl,
                max_ddl_concurrency=max_ddl_concurrency,
                own_pool_size=own_pool_size
            )
        
        # Backward compatibility with old PascalCase params
        if UseEnginePool is not None:
            config.use_engine_pool = UseEnginePool
        if AutoDeleteTrigger is not None:
            config.auto_delete_trigger = AutoDeleteTrigger
        
        if not engine:
            raise ValueError("Must provide a valid engine.")
        
        return cls(items=None, engine=engine, config=config, _factory=True)

    def __init__(self, items, engine=None, config: Optional[SubscriberConfig] = None, 
                 _factory: bool = False,
                 # Backward compat (deprecated):
                 _use_engine_pool: Optional[bool] = None,
                 _autodelete_trigger: Optional[bool] = None):
        # Apply config
        if config is None:
            config = SubscriberConfig()
        
        # Backward compat
        if _use_engine_pool is not None:
            config.use_engine_pool = _use_engine_pool
        if _autodelete_trigger is not None:
            config.auto_delete_trigger = _autodelete_trigger
        
        if _factory:
            self._is_factory = True
            self.engine = engine
            self.config = config
            self._engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
            
            # Own pool (when use_engine_pool=False)
            self._own_pool_sync: Optional[Any] = None
            self._own_pool_async: Optional[Any] = None
            
            # Factory state
            self._callback = None
            self._custom_channel = None
            self._event = None
            self._events: List[str] = []
            self._channel = None
            self._items = []
            self._thread = None
            self._running = False
            self._created_triggers = []
            self._listener_conn = None
            return

        if engine is None:
            raise ValueError("Must provide an engine. Use Subscribe.engine(engine).")

        self._is_factory = False
        self.engine = engine
        self.config = config
        self._engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))

        # Own pool (when use_engine_pool=False)
        self._own_pool_sync: Optional[Any] = None
        self._own_pool_async: Optional[Any] = None

        if items is None:
            self._items = []
        elif isinstance(items, (list, tuple)):
            self._items = list(items)
        else:
            self._items = [items]

        self._event: Optional[str] = None
        self._events: List[str] = []
        self._callback: Optional[Callable[[dict], Any]] = None
        self._custom_channel: Optional[str] = None
        self._channel: str = self._generate_channel()
        self._thread = None
        self._running = False
        self._created_triggers: List[tuple] = []
        self._listener_conn = None

        # Register cleanup
        atexit.register(self._cleanup_on_exit)

        # Validate columns
        for itm in self._items:
            if isinstance(itm, Column):
                self._validate_column_exists(itm)

    # --------------------------------------------------------
    # API pública
    # --------------------------------------------------------
    def __call__(self, *model_or_column):
        """
        Factory call to create subscription instance.
        
        Usage:
            sub = Subscribe.engine(engine)
            sub(User).OnEvent("insert").Exec(callback).Start()
        """
        if not self._is_factory:
            raise TypeError("This instance is not a factory. Use Subscribe.engine(engine) first.")
        
        if not model_or_column:
            return Subscribe(items=None, engine=self.engine, config=self.config)
        
        items = list(model_or_column)
        payload = items if len(items) > 1 else items[0]
        return Subscribe(items=payload, engine=self.engine, config=self.config)

    def OnEvent(self, *event_names: str):
        if self._is_factory:
            raise ValueError("Instancia de fábrica: llama primero Subscribe.engine(engine)(Model).")

        # Soportar OnEvent("update", "delete") o OnEvent(["update", "delete"])
        if len(event_names) == 1 and isinstance(event_names[0], (list, tuple, set)):
            events = list(event_names[0])
        else:
            events = list(event_names)

        events = [str(e).lower().strip() for e in events if e]
        if not events:
            raise ValueError("Debe indicar al menos un evento.")

        if "change" in events:
            if len(events) > 1:
                raise ValueError("El evento 'change' no se puede combinar con otros.")
            resolved = ["insert", "update", "delete"]
        else:
            valid = {"insert", "update", "delete"}
            resolved = []
            for ev in events:
                if ev not in valid:
                    raise ValueError(f"Evento no soportado: {ev}")
                if ev not in resolved:
                    resolved.append(ev)

        self._events = resolved
        # mantener compat con código previo
        self._event = resolved[0] if len(resolved) == 1 else ",".join(resolved)
        return self

    def Exec(self, callback: Callable[[dict], Any]):
        if not callable(callback):
            raise ValueError("Callback inválido.")
        self._callback = self._wrap_payload_callback(callback)
        return self

    def OnChannel(self, channel_name: str):
        if not channel_name:
            raise ValueError("channel_name inválido.")
        self._custom_channel = channel_name.strip()
        return self

    def Start(self):
        """Start listening thread and create triggers."""
        if self._is_factory:
            raise ValueError("Factory instance: call Subscribe.engine(engine)(Model) first.")

        # Default callback for debugging
        if self._custom_channel and not self._callback:
            self._callback = lambda p: self.engine._debug("[SUBSCRIBER] NOTIFY: {}", p)

        if not self._callback:
            raise ValueError("Callback must be set.")

        if not self._custom_channel:
            if not self._events:
                raise ValueError("Event must be set when not using custom channel.")
            if not self._items:
                raise ValueError("Must provide model/column or use OnChannel().")
            
            try:
                # Cleanup old triggers and create new ones
                if self._engine_async:
                    # For async engines, we need to run in the current async context
                    # This will be called from the thread running asyncio.run()
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self._drop_existing_sub_triggers_async())
                    loop.run_until_complete(self._ensure_triggers_async(self._channel))
                else:
                    # Sync engine
                    self._drop_existing_sub_triggers()
                    self._ensure_triggers(self._channel)
            except Exception:
                if self.config.auto_delete_trigger:
                    self._drop_created_triggers()
                raise

        channels = [self._custom_channel or self._channel]
        self._running = True

        daemon = self.config.daemon_mode

        if self._engine_async:
            self._thread = threading.Thread(
                target=lambda: asyncio.run(self._listen_loop_async(channels)),
                daemon=daemon
            )
        else:
            self._thread = threading.Thread(
                target=self._listen_loop_sync,
                args=(channels,),
                daemon=daemon
            )

        try:
            self._thread.start()
            self.engine._debug("[SUBSCRIBER] Started listening on channels: {}", channels)
        except Exception:
            if self.config.auto_delete_trigger:
                self._drop_created_triggers()
            raise
        
        return self

    async def StartAsync(self):
        """
        Start listening (Async version).
        
        Creates triggers asynchronously and runs the listener loop as an asyncio Task
        in the current loop (avoiding threads).
        """
        if self._is_factory:
            raise ValueError("Factory instance: call Subscribe.engine(engine)(Model) first.")

        if self._custom_channel and not self._callback:
             self._callback = lambda p: self.engine._debug("[SUBSCRIBER] NOTIFY: {}", p)

        if not self._callback:
            raise ValueError("Callback must be set.")

        if not self._custom_channel:
            if not self._events:
                raise ValueError("Event must be set when not using custom channel.")
            if not self._items:
                raise ValueError("Must provide model/column or use OnChannel().")
            
            try:
                # Cleanup old and create new triggers (async)
                if not self._engine_async:
                     raise RuntimeError("StartAsync requires an async Engine.")
                
                await self._drop_existing_sub_triggers_async()
                await self._ensure_triggers_async(self._channel)
            except Exception:
                if self.config.auto_delete_trigger:
                    await self._drop_created_triggers_async()
                raise

        channels = [self._custom_channel or self._channel]
        self._running = True

        # Use asyncio Task instead of thread
        self._listening_task = asyncio.create_task(self._listen_loop_async(channels))
        self.engine._debug("[SUBSCRIBER] Started listening task on channels: {}", channels)
        
        return self

    def Stop(self):
        """Stop listening and cleanup."""
        self._running = False
        if self._thread:
            try:
                self._thread.join(timeout=5.0)
            except Exception:
                pass
        
        if self.config.auto_delete_trigger:
            self._drop_created_triggers()
        
        # Cleanup own pool
        if not self.config.use_engine_pool:
            self._cleanup_own_pool()
        
        # Close listener connection
        if not self._engine_async and self._listener_conn:
            try:
                if self.config.use_engine_pool:
                    self.engine.release_sync(self._listener_conn)
                else:
                    self._listener_conn.close()
            except:
                pass
            self._listener_conn = None

    async def StopAsync(self):
        """Stop listening and cleanup (Async version)."""
        self._running = False
        
        # Stop task
        task = getattr(self, "_listening_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._listening_task = None
        
        if self.config.auto_delete_trigger:
            await self._drop_created_triggers_async()
        
        # Cleanup own pool
        if not self.config.use_engine_pool:
            self._cleanup_own_pool()

    # --------------------------------------------------------
    # Helpers de configuración
    # --------------------------------------------------------
    def _cleanup_on_exit(self):
        """Cleanup method called by atexit."""
        if self.config.auto_delete_trigger:
            self._drop_created_triggers()

    # --------------------------------------------------------
    # Own Pool Management
    # --------------------------------------------------------
    def _init_own_pool(self):
        """Initialize own sync connection pool when use_engine_pool=False."""
        if self.config.use_engine_pool or self._own_pool_sync is not None:
            return
        
        import queue as queue_module
        self._own_pool_sync = queue_module.Queue(maxsize=self.config.own_pool_size)
        
        for _ in range(self.config.own_pool_size):
            try:
                conn = self._create_direct_connection_sync()
                self._own_pool_sync.put(conn)
            except Exception as e:
                self.engine._debug("[SUBSCRIBER] Failed to create pool connection: {}", e)
                raise
        
        self.engine._debug("[SUBSCRIBER] Created own sync pool: {} connections", 
                         self.config.own_pool_size)

    async def _init_own_pool_async(self):
        """Initialize async pool when use_engine_pool=False."""
        if self.config.use_engine_pool or self._own_pool_async is not None:
            return
        
        dsn = self._get_dsn()
        try:
            self._own_pool_async = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=self.config.own_pool_size
            )
            self.engine._debug("[SUBSCRIBER] Created own async pool")
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Failed to create async pool: {}", e)
            raise

    def _cleanup_own_pool(self):
        """Cleanup own connection pools."""
        # Sync pool
        if self._own_pool_sync is not None:
            while not self._own_pool_sync.empty():
                try:
                    conn = self._own_pool_sync.get_nowait()
                    try:
                        conn.close()
                    except:
                        pass
                except:
                    break
            self._own_pool_sync = None
            self.engine._debug("[SUBSCRIBER] Cleaned up own sync pool")
        
        # Async pool
        if self._own_pool_async is not None:
            try:
                # Try async cleanup
                asyncio.run(self._own_pool_async.close())
            except:
                # Fallback: just set to None
                pass
            self._own_pool_async = None
            self.engine._debug("[SUBSCRIBER] Cleaned up own async pool")

    def _create_direct_connection_sync(self):
        """Create direct psycopg connection (for own pool)."""
        dsn = self._get_dsn()
        return psycopg.connect(dsn)

    def _get_dsn(self) -> str:
        """Get DSN from engine config."""
        cfg = self.engine.config
        if cfg.dsn:
            return cfg.dsn
        
        dbname = cfg.database or "postgres"
        user = cfg.username or ""
        password = cfg.password or ""
        host = cfg.host or "localhost"
        port = cfg.port or 5432
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    # --------------------------------------------------------
    # Connection Acquisition/Release
    # --------------------------------------------------------
    def _acquire_connection_sync(self):
        """Acquire sync connection from engine pool or own pool with retry."""
        import time
        
        for attempt in range(self.config.connection_retry):
            try:
                if self.config.use_engine_pool:
                    # Use engine's pool
                    conn = self.engine.acquire_sync()
                    self.engine._debug("[SUBSCRIBER] Acquired connection from engine pool")
                    return conn
                else:
                    # Use own pool
                    if self._own_pool_sync is None:
                        self._init_own_pool()
                    conn = self._own_pool_sync.get(timeout=self.config.listen_timeout)
                    self.engine._debug("[SUBSCRIBER] Acquired connection from own pool")
                    return conn
            except Exception as e:
                if attempt < self.config.connection_retry - 1:
                    self.engine._debug("[SUBSCRIBER] Connection acquisition failed (attempt {}/{}): {}", 
                                     attempt + 1, self.config.connection_retry, e)
                    time.sleep(self.config.retry_delay)
                    continue
                self.engine._debug("[SUBSCRIBER] Connection acquisition failed after {} attempts", 
                                 self.config.connection_retry)
                raise

    async def _acquire_connection_async(self):
        """Acquire async connection from engine pool or own pool with retry."""
        for attempt in range(self.config.connection_retry):
            try:
                if self.config.use_engine_pool:
                    # Use engine's pool
                    conn = await self.engine.acquire_async()
                    self.engine._debug("[SUBSCRIBER] Acquired async connection from engine pool")
                    return conn
                else:
                    # Use own pool
                    if self._own_pool_async is None:
                        await self._init_own_pool_async()
                    conn = await self._own_pool_async.acquire()
                    self.engine._debug("[SUBSCRIBER] Acquired async connection from own pool")
                    return conn
            except Exception as e:
                if attempt < self.config.connection_retry - 1:
                    self.engine._debug("[SUBSCRIBER] Async connection acquisition failed (attempt {}/{}): {}", 
                                     attempt + 1, self.config.connection_retry, e)
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                self.engine._debug("[SUBSCRIBER] Async connection acquisition failed after {} attempts", 
                                 self.config.connection_retry)
                raise

    def _release_connection_sync(self, conn):
        """Release sync connection back to pool."""
        if conn is None:
            return
        
        try:
            if self.config.use_engine_pool:
                self.engine.release_sync(conn)
                self.engine._debug("[SUBSCRIBER] Released connection to engine pool")
            else:
                if self._own_pool_sync is not None:
                    self._own_pool_sync.put(conn)
                    self.engine._debug("[SUBSCRIBER] Released connection to own pool")
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error releasing connection: {}", e)

    async def _release_connection_async(self, conn):
        """Release async connection back to pool."""
        if conn is None:
            return
        
        try:
            if self.config.use_engine_pool:
                await self.engine.release_async(conn)
                self.engine._debug("[SUBSCRIBER] Released async connection to engine pool")
            else:
                if self._own_pool_async is not None:
                    await self._own_pool_async.release(conn)
                    self.engine._debug("[SUBSCRIBER] Released async connection to own pool")
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error releasing async connection: {}", e)

    # --------------------------------------------------------
    # Helpers de configuración
    # --------------------------------------------------------
    def _generate_channel(self) -> str:
        return f"__sub_{uuid4().hex}"

    def _resolve_events(self) -> List[str]:
        if self._events:
            return list(self._events)
        ev = (self._event or '').lower().strip()
        if ev == 'change':
            return ['insert', 'update', 'delete']
        return [ev]

    def _group_items(self) -> List[Dict[str, Any]]:
        groups: Dict[tuple, Dict[str, Any]] = {}
        for item in self._items:
            if isinstance(item, Column):
                base_model = getattr(item, 'model', None)
                col_name = getattr(item, 'name', None) or getattr(item, 'attr_name', None)
                if not base_model or not col_name:
                    raise ColumnNotFoundError("Columna inválida para suscripción.")
                schema = getattr(base_model, '__schema__', 'public') or 'public'
                table = getattr(base_model, '__tablename__', base_model.__name__.lower())
                key = (schema, table, base_model)
                entry = groups.setdefault(key, {
                    'schema': schema,
                    'table': table,
                    'model': base_model,
                    'columns': set(),
                    'table_level': False,
                })
                entry['columns'].add(col_name)
            else:
                model_cls = item
                schema = getattr(model_cls, '__schema__', 'public') or 'public'
                table = getattr(model_cls, '__tablename__', model_cls.__name__.lower())
                key = (schema, table, model_cls)
                entry = groups.setdefault(key, {
                    'schema': schema,
                    'table': table,
                    'model': model_cls,
                    'columns': set(),
                    'table_level': False,
                })
                entry['table_level'] = True
        return list(groups.values())

    def _validate_column_exists(self, column: Column):
        base_model = getattr(column, 'model', None)
        name = getattr(column, 'name', None) or getattr(column, 'attr_name', None)
        cols = getattr(base_model, '__columns__', {}) if base_model else {}
        if name not in cols:
            raise ColumnNotFoundError(f"La columna '{name}' no existe en {getattr(base_model, '__name__', base_model)}")

    # --------------------------------------------------------
    # TRIGGERS
    # --------------------------------------------------------
    def _ensure_triggers(self, channel: str):
        """Create triggers using sync Transaction."""
        if self._engine_async:
            raise RuntimeError("Use _ensure_triggers_async for async engines")
        
        groups = self._group_items()
        events = self._resolve_events()
        for grp in groups:
            pk = self._primary_key_name(grp['model'])
            for ev in events:
                if grp['table_level']:
                    self._create_table_trigger(grp['schema'], grp['table'], pk, ev, channel)
                else:
                    cols = sorted(grp['columns'])
                    self._create_column_trigger(grp['schema'], grp['table'], pk, cols, ev, channel)

    async def _ensure_triggers_async(self, channel: str):
        """
        Create triggers using AsyncTransaction.
        
        Supports parallel execution when config.parallel_ddl=True, using
        engine.parallel_execute_async() to create multiple triggers concurrently.
        """
        groups = self._group_items()
        events = self._resolve_events()
        
        # Build list of all trigger creation tasks
        trigger_tasks = []
        for grp in groups:
            pk = self._primary_key_name(grp['model'])
            for ev in events:
                if grp['table_level']:
                    trigger_tasks.append((
                        'table',
                        grp['schema'], 
                        grp['table'], 
                        pk, 
                        ev, 
                        channel
                    ))
                else:
                    cols = sorted(grp['columns'])
                    trigger_tasks.append((
                        'column',
                        grp['schema'],
                        grp['table'],
                        pk,
                        cols,
                        ev,
                        channel
                    ))
        
        if not trigger_tasks:
            return
        
        # Execute in parallel or sequential based on config
        if self.config.parallel_ddl and len(trigger_tasks) > 1:
            # Parallel execution using engine.parallel_execute_async()
            self.engine._debug("[SUBSCRIBER] Creating {} triggers in parallel (max concurrency: {})", 
                             len(trigger_tasks), self.config.max_ddl_concurrency)
            
            async def create_trigger_task(task_info):
                """Wrapper to create a single trigger."""
                if task_info[0] == 'table':
                    _, schema, table, pk, ev, ch = task_info
                    await self._create_table_trigger_async(schema, table, pk, ev, ch)
                else:  # column
                    _, schema, table, pk, cols, ev, ch = task_info
                    await self._create_column_trigger_async(schema, table, pk, cols, ev, ch)
            
            # Use engine.parallel_execute_async if available, otherwise asyncio.gather
            if hasattr(self.engine, 'parallel_execute_async'):
                # Engine has parallel_execute_async - use it
                tasks = [create_trigger_task(t) for t in trigger_tasks]
                await self.engine.parallel_execute_async(
                    tasks, 
                    max_concurrency=self.config.max_ddl_concurrency
                )
            else:
                # Fallback: use asyncio.gather with semaphore for concurrency control
                import asyncio
                semaphore = asyncio.Semaphore(self.config.max_ddl_concurrency)
                
                async def limited_task(task_info):
                    async with semaphore:
                        await create_trigger_task(task_info)
                
                await asyncio.gather(*[limited_task(t) for t in trigger_tasks])
            
            self.engine._debug("[SUBSCRIBER] Completed parallel trigger creation")
        else:
            # Sequential execution
            for task_info in trigger_tasks:
                if task_info[0] == 'table':
                    _, schema, table, pk, ev, ch = task_info
                    await self._create_table_trigger_async(schema, table, pk, ev, ch)
                else:  # column
                    _, schema, table, pk, cols, ev, ch = task_info
                    await self._create_column_trigger_async(schema, table, pk, cols, ev, ch)

    def _primary_key_name(self, model_cls) -> Optional[str]:
        cols = getattr(model_cls, '__columns__', {}) if model_cls else {}
        for name, col in cols.items():
            if getattr(col, 'primary_key', False):
                return name
        return None

    @staticmethod
    def _qident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _create_table_trigger(self, schema: str, table: str, pk: Optional[str], event: str, channel: str):
        trig_base = f"__sub_{table}_{event}_{self._channel[-6:]}"
        trigger_name = trig_base
        function_name = f"{trig_base}_fn"
        ch = channel

        payload = f"json_build_object('event', lower(TG_OP), 'schema', TG_TABLE_SCHEMA, 'table', TG_TABLE_NAME, 'pk_name', '{pk}', 'old', CASE WHEN TG_OP IN ('UPDATE','DELETE') THEN row_to_json(OLD) END, 'new', CASE WHEN TG_OP IN ('UPDATE','INSERT') THEN row_to_json(NEW) END)::text"

        body = (
            f"PERFORM pg_notify('{ch}', {payload});\n"
            f"IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;"
        )

        qs, qt, qtrg, qfn = map(self._qident, [schema, table, trigger_name, function_name])
        sql = f"""
        CREATE OR REPLACE FUNCTION {qfn}() RETURNS trigger AS $$
        BEGIN
            {body}
        END; $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {qtrg} ON {qs}.{qt};
        CREATE TRIGGER {qtrg} AFTER {event.upper()} ON {qs}.{qt} FOR EACH ROW EXECUTE FUNCTION {qfn}();
        """
        self._execute_ddl(sql, schema, table, trigger_name, function_name)

    async def _create_table_trigger_async(self, schema: str, table: str, pk: Optional[str], 
                                          event: str, channel: str):
        """Async version of _create_table_trigger."""
        trig_base = f"__sub_{table}_{event}_{self._channel[-6:]}"
        trigger_name = trig_base
        function_name = f"{trig_base}_fn"
        ch = channel

        payload = f"json_build_object('event', lower(TG_OP), 'schema', TG_TABLE_SCHEMA, 'table', TG_TABLE_NAME, 'pk_name', '{pk}', 'old', CASE WHEN TG_OP IN ('UPDATE','DELETE') THEN row_to_json(OLD) END, 'new', CASE WHEN TG_OP IN ('UPDATE','INSERT') THEN row_to_json(NEW) END)::text"

        body = (
            f"PERFORM pg_notify('{ch}', {payload});\n"
            f"IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;"
        )

        qs, qt, qtrg, qfn = map(self._qident, [schema, table, trigger_name, function_name])
        sql_list = [
            f"CREATE OR REPLACE FUNCTION {qfn}() RETURNS trigger AS $$\nBEGIN\n{body}\nEND; $$ LANGUAGE plpgsql;",
            f"DROP TRIGGER IF EXISTS {qtrg} ON {qs}.{qt};",
            f"CREATE TRIGGER {qtrg} AFTER {event.upper()} ON {qs}.{qt} FOR EACH ROW EXECUTE FUNCTION {qfn}();"
        ]
        await self._execute_ddl_async(sql_list, schema, table, trigger_name, function_name)

    def _create_column_trigger(self, schema: str, table: str, pk: Optional[str], columns: List[str], event: str, channel: str):
        if not columns:
            return
        trig_base = f"__sub_{table}_{event}_{self._channel[-6:]}_{len(self._created_triggers)}"
        trigger_name = trig_base
        function_name = f"{trig_base}_fn"
        ch = channel

        if len(columns) == 1:
            col_field = f"'column', '{columns[0]}'"
        else:
            arr = ','.join(f"'{c}'" for c in columns)
            col_field = f"'columns', ARRAY[{arr}]"

        old_fields = [f"'{pk}', OLD.{pk}"] if pk else []
        new_fields = [f"'{pk}', NEW.{pk}"] if pk else []
        for c in columns:
            if pk and c == pk:
                continue  # evitar duplicar el primary key en old/new
            old_fields.append(f"'{c}', OLD.{c}")
            new_fields.append(f"'{c}', NEW.{c}")
        old_json = f"json_build_object({', '.join(old_fields)})" if old_fields else "NULL"
        new_json = f"json_build_object({', '.join(new_fields)})" if new_fields else "NULL"

        payload = f"json_build_object('event', lower(TG_OP), 'schema', TG_TABLE_SCHEMA, 'table', TG_TABLE_NAME, {col_field}, 'pk_name', '{pk}', 'old', CASE WHEN TG_OP IN ('UPDATE','DELETE') THEN {old_json} END, 'new', CASE WHEN TG_OP IN ('UPDATE','INSERT') THEN {new_json} END)::text"

        qs, qt, qtrg, qfn = map(self._qident, [schema, table, trigger_name, function_name])

        if event == 'update':
            change_cond = ' OR '.join([f"NEW.{c} IS DISTINCT FROM OLD.{c}" for c in columns]) or 'FALSE'
            body = (
                f"IF {change_cond} THEN\n"
                f"    PERFORM pg_notify('{ch}', {payload});\n"
                f"END IF;\n"
                f"RETURN NEW;"
            )
        elif event == 'delete':
            body = (
                f"PERFORM pg_notify('{ch}', {payload});\n"
                f"RETURN OLD;"
            )
        else:  # insert
            body = (
                f"PERFORM pg_notify('{ch}', {payload});\n"
                f"RETURN NEW;"
            )

        sql = f"""
        CREATE OR REPLACE FUNCTION {qfn}() RETURNS trigger AS $$
        BEGIN
            {body}
        END; $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {qtrg} ON {qs}.{qt};
        CREATE TRIGGER {qtrg} AFTER {event.upper()} ON {qs}.{qt} FOR EACH ROW EXECUTE FUNCTION {qfn}();
        """
        self._execute_ddl(sql, schema, table, trigger_name, function_name)

    async def _create_column_trigger_async(self, schema: str, table: str, pk: Optional[str], 
                                           columns: List[str], event: str, channel: str):
        """Async version of _create_column_trigger."""
        if not columns:
            return
        trig_base = f"__sub_{table}_{event}_{self._channel[-6:]}_{len(self._created_triggers)}"
        trigger_name = trig_base
        function_name = f"{trig_base}_fn"
        ch = channel

        if len(columns) == 1:
            col_field = f"'column', '{columns[0]}'"
        else:
            arr = ','.join(f"'{c}'" for c in columns)
            col_field = f"'columns', ARRAY[{arr}]"

        old_fields = [f"'{pk}', OLD.{pk}"] if pk else []
        new_fields = [f"'{pk}', NEW.{pk}"] if pk else []
        for c in columns:
            if pk and c == pk:
                continue
            old_fields.append(f"'{c}', OLD.{c}")
            new_fields.append(f"'{c}', NEW.{c}")
        old_json = f"json_build_object({', '.join(old_fields)})" if old_fields else "NULL"
        new_json = f"json_build_object({', '.join(new_fields)})" if new_fields else "NULL"

        payload = f"json_build_object('event', lower(TG_OP), 'schema', TG_TABLE_SCHEMA, 'table', TG_TABLE_NAME, {col_field}, 'pk_name', '{pk}', 'old', CASE WHEN TG_OP IN ('UPDATE','DELETE') THEN {old_json} END, 'new', CASE WHEN TG_OP IN ('UPDATE','INSERT') THEN {new_json} END)::text"

        qs, qt, qtrg, qfn = map(self._qident, [schema, table, trigger_name, function_name])

        if event == 'update':
            change_cond = ' OR '.join([f"NEW.{c} IS DISTINCT FROM OLD.{c}" for c in columns]) or 'FALSE'
            body = (
                f"IF {change_cond} THEN\n"
                f"    PERFORM pg_notify('{ch}', {payload});\n"
                f"END IF;\n"
                f"RETURN NEW;"
            )
        elif event == 'delete':
            body = (
                f"PERFORM pg_notify('{ch}', {payload});\n"
                f"RETURN OLD;"
            )
        else:  # insert
            body = (
                f"PERFORM pg_notify('{ch}', {payload});\n"
                f"RETURN NEW;"
            )

        sql_list = [
            f"CREATE OR REPLACE FUNCTION {qfn}() RETURNS trigger AS $$\nBEGIN\n{body}\nEND; $$ LANGUAGE plpgsql;",
            f"DROP TRIGGER IF EXISTS {qtrg} ON {qs}.{qt};",
            f"CREATE TRIGGER {qtrg} AFTER {event.upper()} ON {qs}.{qt} FOR EACH ROW EXECUTE FUNCTION {qfn}();"
        ]
        await self._execute_ddl_async(sql_list, schema, table, trigger_name, function_name)

    def _execute_ddl(self, sql: str, schema: str, table: str, trigger_name: str, function_name: str):
        """
        Execute DDL using Transaction (leverages engine middleware/hooks).
        
        Routes through engine's execution pipeline for full observability.
        """
        if self._engine_async:
            # For async engines, we need to run in async context
            # This will be called from Start() which runs in thread with asyncio.run()
            # So we can use the async version
            raise RuntimeError("Use _execute_ddl_async for async engines")
        
        try:
            # Use Transaction for DDL to leverage middleware/hooks
            with Transaction(self.engine) as tx:
                # Execute DDL in transaction context
                # Note: PostgreSQL DDL is transactional, but we set AUTOCOMMIT for triggers
                tx._execute_sql_in_tx_sync(sql)
                self._created_triggers.append((schema, table, trigger_name, function_name))
                self.engine._debug("[SUBSCRIBER] Created trigger via Transaction: {}.{}", 
                                 schema, trigger_name)
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error creating trigger {}: {}", trigger_name, e)
            raise TriggerCreationError(f"Failed to create trigger {trigger_name}: {e}") from e

    async def _execute_ddl_async(self, sql: str, schema: str, table: str, 
                                 trigger_name: str, function_name: str):
        """
        Execute DDL using AsyncTransaction (leverages engine middleware/hooks).
        
        Async version for async engines.
        """
        try:
            # Use AsyncTransaction for DDL
            async with AsyncTransaction(self.engine) as tx:
                if isinstance(sql, list):
                    for s in sql:
                        if s.strip():
                            await tx._execute_sql_in_tx_async(s)
                else:
                    await tx._execute_sql_in_tx_async(sql)
                
                self._created_triggers.append((schema, table, trigger_name, function_name))
                self.engine._debug("[SUBSCRIBER] Created trigger via AsyncTransaction: {}.{}", 
                                 schema, trigger_name)
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error creating async trigger {}: {}", trigger_name, e)
            raise TriggerCreationError(f"Failed to create async trigger {trigger_name}: {e}") from e

    # --------------------------------------------------------
    # LISTEN LOOPS
    # --------------------------------------------------------
    def _listen_loop_sync(self, channels: List[str]):
        """Sync LISTEN loop using engine or own pool."""
        conn = None
        try:
            conn = self._acquire_connection_sync()
            conn.autocommit = True
            cur = conn.cursor()
            
            for ch in channels:
                cur.execute(f"LISTEN {ch};")
                self.engine._debug("[SUBSCRIBER] LISTEN {}", ch)
            
            try:
                while self._running:
                    # psycopg3 uses notifies() generator with timeout
                    for n in conn.notifies(timeout=self.config.listen_timeout):
                        try:
                            payload = json.loads(n.payload)
                        except Exception:
                            payload = {"raw": n.payload, "channel": n.channel}
                        try:
                            self._callback(payload)
                        except Exception as e:
                            self.engine._debug("[SUBSCRIBER] Callback error: {}", e)
                        # Break after processing to check _running flag
                        break
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
        finally:
            if conn is not None:
                self._release_connection_sync(conn)

    async def _listen_loop_async(self, channels: List[str]):
        """Async LISTEN loop using engine or own pool."""
        conn = None
        try:
            conn = await self._acquire_connection_async()
            
            for ch in channels:
                await conn.add_listener(ch, self._async_handler)
                self.engine._debug("[SUBSCRIBER] LISTEN {} (async)", ch)
            
            try:
                while self._running:
                    await asyncio.sleep(self.config.listen_timeout)
            finally:
                # Remove listeners
                for ch in channels:
                    try:
                        await conn.remove_listener(ch, self._async_handler)
                    except:
                        pass
        finally:
            if conn is not None:
                await self._release_connection_async(conn)

    def _async_handler(self, conn, pid, channel, payload):
        """Handle async NOTIFY messages."""
        try:
            data = json.loads(payload)
        except Exception:
            data = {"raw": payload, "channel": channel}
        try:
            self._callback(data)
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Async callback error: {}", e)

    def _wrap_payload_callback(self, cb: Callable):
        sig = inspect.signature(cb)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("El callback debe aceptar el payload como primer parámetro.")
        first = params[0]
        if first.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise ValueError("El primer parámetro del callback debe aceptar el payload.")

        def _wrapper(payload):
            return cb(payload)

        return _wrapper

    # --------------------------------------------------------
    # LIMPIEZA
    # --------------------------------------------------------
    def _drop_created_triggers(self):
        """Drop triggers created by this subscription using engine pool."""
        if not self._created_triggers:
            return
        
        conn = None
        try:
            conn = self._acquire_connection_sync()
            conn.autocommit = True
            cur = conn.cursor()
            try:
                for schema, table, trg, fn in self._created_triggers:
                    try:
                        cur.execute(f'DROP TRIGGER IF EXISTS "{trg}" ON "{schema}"."{table}";')
                        cur.execute(f'DROP FUNCTION IF EXISTS "{fn}"();')
                        self.engine._debug("[SUBSCRIBER] Dropped trigger: {}.{}", schema, trg)
                    except Exception as e:
                        self.engine._debug("[SUBSCRIBER] Error dropping trigger {}: {}", trg, e)
            finally:
                cur.close()
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error in _drop_created_triggers: {}", e)
        finally:
            if conn is not None:
                self._release_connection_sync(conn)
            self._created_triggers = []
            
    async def _drop_created_triggers_async(self):
        """Drop triggers created by this subscription using engine pool (async)."""
        if not self._created_triggers:
            return
        
        conn = None
        try:
            conn = await self._acquire_connection_async()
            try:
                for schema, table, trg, fn in self._created_triggers:
                    try:
                        await conn.execute(f'DROP TRIGGER IF EXISTS "{trg}" ON "{schema}"."{table}";')
                        await conn.execute(f'DROP FUNCTION IF EXISTS "{fn}"();')
                        self.engine._debug("[SUBSCRIBER] Dropped trigger (async): {}.{}", schema, trg)
                    except Exception as e:
                        self.engine._debug("[SUBSCRIBER] Error dropping async trigger {}: {}", trg, e)
            finally:
                pass
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error in _drop_created_triggers_async: {}", e)
        finally:
            if conn is not None:
                await self._release_connection_async(conn)
            self._created_triggers = []

    def _drop_existing_sub_triggers(self):
        """
        Cleanup old __sub_ triggers from target tables using engine pool.
        Best-effort cleanup of triggers from previous runs that didn't Stop().
        """
        targets = {(grp['schema'], grp['table']) for grp in self._group_items()}
        if not targets:
            return
        
        conn = None
        try:
            conn = self._acquire_connection_sync()
            conn.autocommit = True
            cur = conn.cursor()
            try:
                for schema, table in targets:
                    try:
                        cur.execute("""
                            SELECT trigger_name
                            FROM information_schema.triggers
                            WHERE event_object_schema=%s
                              AND event_object_table=%s
                              AND trigger_name LIKE '__sub_%%';
                        """, (schema, table))
                        rows = cur.fetchall()
                        for (trg,) in rows:
                            cur.execute(f'DROP TRIGGER IF EXISTS "{trg}" ON "{schema}"."{table}";')
                            # Drop associated function
                            fn = f"{trg}_fn"
                            cur.execute(f'DROP FUNCTION IF EXISTS "{fn}"();')
                        if rows:
                            self.engine._debug("[SUBSCRIBER] Cleaned up {} old triggers from {}.{}", 
                                             len(rows), schema, table)
                    except Exception as e:
                        self.engine._debug("[SUBSCRIBER] Error cleaning triggers from {}.{}: {}", 
                                         schema, table, e)
            finally:
                cur.close()
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error in _drop_existing_sub_triggers: {}", e)
        finally:
            if conn is not None:
                self._release_connection_sync(conn)

    async def _drop_existing_sub_triggers_async(self):
        """
        Async version: Cleanup old __sub_ triggers from target tables.
        Best-effort cleanup of triggers from previous runs that didn't Stop().
        """
        targets = {(grp['schema'], grp['table']) for grp in self._group_items()}
        if not targets:
            return
        
        conn = None
        try:
            conn = await self._acquire_connection_async()
            # For asyncpg, we can't set autocommit like psycopg
            # asyncpg connections are in autocommit by default for DDL
            
            for schema, table in targets:
                try:
                    # Query triggers
                    rows = await conn.fetch("""
                        SELECT trigger_name
                        FROM information_schema.triggers
                        WHERE event_object_schema=$1
                          AND event_object_table=$2
                          AND trigger_name LIKE '__sub_%';
                    """, schema, table)
                    
                    for row in rows:
                        trg = row['trigger_name']
                        await conn.execute(f'DROP TRIGGER IF EXISTS "{trg}" ON "{schema}"."{table}";')
                        # Drop associated function
                        fn = f"{trg}_fn"
                        await conn.execute(f'DROP FUNCTION IF EXISTS "{fn}"();')
                    
                    if rows:
                        self.engine._debug("[SUBSCRIBER] Cleaned up {} old triggers from {}.{} (async)", 
                                         len(rows), schema, table)
                except Exception as e:
                    self.engine._debug("[SUBSCRIBER] Error cleaning async triggers from {}.{}: {}", 
                                     schema, table, e)
        except Exception as e:
            self.engine._debug("[SUBSCRIBER] Error in _drop_existing_sub_triggers_async: {}", e)
        finally:
            if conn is not None:
                await self._release_connection_async(conn)

    # --------------------------------------------------------
    # DSN
    # --------------------------------------------------------
    def _default_dsn(self):
        cfg = getattr(self.engine, 'config', None)
        if not cfg:
            raise ValueError("Engine configuration no disponible para construir DSN.")
        if getattr(cfg, 'dsn', None):
            return cfg.dsn
        dbname = cfg.database or "postgres"
        user = cfg.username or ""
        password = cfg.password or ""
        host = cfg.host or "localhost"
        port = cfg.port or 5432
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
