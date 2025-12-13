# PSQLModel Engine Core Reference

This document provides a comprehensive technical reference for `psqlmodel/core/engine.py`, the central component of the ORM responsible for database connectivity, query execution, and schema management.

## Overview

The `Engine` acts as the bridge between your Python code and the PostgreSQL database. It handles:

1.  **Connection Management**: Pooling for both Synchronous (`psycopg2`) and Asynchronous (`asyncpg`) modes.
2.  **Execution Lifecycle**: A pipeline that supports middlewares, execution hooks, logging, and error handling.
3.  **Schema Management**: Automatic detection of models, DDL generation, and database initialization.
4.  **Reliability**: Health checks, automatic pool resizing, and connection repair.

---

## 1. Engine Configuration (`EngineConfig`)

The `EngineConfig` dataclass holds all configuration parameters. It automatically parses standard connection strings (DSN) in its `__post_init__` method.

### Key Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dsn` | `str` | `None` | Connection URL (e.g., `postgresql://user:pass@host:5432/db`). |
| `async_` | `bool` | `False` | Application mode. `False` for Sync (`psycopg`), `True` for Async (`asyncpg`). |
| `pool_size` | `int` | `20` | Initial number of connections in the pool. |
| `max_pool_size` | `int` | `None` | Hard limit for auto-expanding pools. |
| `auto_adjust_pool_size` | `bool` | `False` | Allows the pool to grow beyond `pool_size` under load. |
| `models_path` | `str` | `None` | Path to directory or file containing models to scan. |
| `ensure_database` | `bool` | `True` | Auto-create database if it doesn't exist. |
| `ensure_tables` | `bool` | `True` | Auto-create tables from models on startup. |
| `ensure_migrations` | `bool` | `False` | Auto-run migrations on startup (see section 7). |
| `migrations_path` | `str` | `./migrations` | Path for migrations directory. |
| `enable_metrics` | `bool` | `True` | Enables internal query counting and stats. |
| `enable_query_tracer` | `bool` | `True` | Keeps a circular buffer of recent queries for debugging. |

---

## 2. The `Engine` Class

The `Engine` class is the main entry point. It is designed to be **thread-safe** and manages double resources for mixed workloads (though usually you run in either sync or async mode).

### Connection Management

The engine does not expose raw connections directly to the user very often. Instead, it provides context managers that handle acquisition and release automatically.

#### Sync Mechanism (using `queue.Queue`)
- **Initialization**: Creates `pool_size` connections immediately.
- **Acquire (`acquire_sync`)**:
  - Pops a connection from the queue.
  - Checks **Pre-ping** (SELECT 1) if enabled.
  - Checks **Pool Recycle** (closes connection if it's too old).
  - Grows pool dynamically if empty and `auto_adjust_pool_size` is True.
- **Release (`release_sync`)**: Puts the connection back into the queue and updates its "last used" timestamp.

#### Async Mechanism (using `asyncpg` Pool)
- **Initialization**: Lazily handled or explicitly via `create_engine`.
- **Acquire (`acquire`)**:
  - Delegates to `asyncpg.pool.Pool.acquire()`.
  - Implements custom logic to handle timeouts and resize the pool (destroying and recreating it with a larger size limit) if `auto_adjust` is enabled.

### Execution Pipeline

Every query goes through a pipeline designed to ensure safety and observability.

**Steps:**
1.  **Normalization**: Converts input (QueryBuilder objects) to raw SQL string + parameters.
2.  **Middlewares**:
    - Wrappers that execute *around* the query.
    - Used for Logging, Metrics, Timeout enforcement, and automatic Retries.
    - Sorted by priority (Higher priority executes 'outermost').
3.  **Hooks (`before_execute`)**: Functions called before the SQL hits the DB.
4.  **Core Execution**:
    - **Sync**: `cur.execute()` -> `cur.fetchall()`.
    - **Async**: `conn.fetch()`.
    - **Parametrization**: Async path automatically converts Python-style `%s` placeholders to Postgres native `$1, $2...`.
5.  **Hooks (`after_execute`)**: Functions called with results.
6.  **Cleanup**: Metrics update and connection release.

### Error Handling

The engine wraps specialized driver errors into generic ORM exceptions:

- **`DatabaseNotFoundError`**: Caught during connection init. The engine can suggest creating the DB or do it automatically if `ensure_database=True`.
- **`ConnectionError`**: Generic wrapper for network/driver failures.
- **`TimeoutError`**: Raised when middleware execution time exceeds the limit or connection acquisition times out.

---

## 3. Schema & DDL (`EnsureDatabaseTables`)

One of the most powerful features is the auto-setup capability.

**How it works:**
1.  **Scanning**: `EnsureDatabaseTables(engine)` iterates over files in `models_path` (or the current project) to import Python modules.
2.  **Registration**: Importing modules triggers the `@table` decorators, registering models in the system.
3.  **Dependency Resolution**:
    - It analyzes Foreign Keys in `CREATE TABLE` statements.
    - It sorts tables topologically so dependencies are created first (e.g., `Users` before `Posts`).
4.  **Junction Tables**: `_generate_junction_tables` automatically creates intermediate tables for `ManyToMany` relationships detected in the models.
5.  **Triggers**: `_ensure_triggers_sync/async` install PL/Python or PL/pgSQL triggers for models decorated with `@Trigger`.
6.  **Plan Execution**: The sorted list of SQL statements (`_last_ddl_plan`) is executed in order.

---

## 4. Health & Observability

### Health Monitor
- A background Daemon Thread (Sync) or asyncio Task (Async).
- Periodically runs `_repair_pool`:
  - Checks if connections are closed/broken.
  - Attempts to seamlessly replace broken connections.

### Metrics & Tracer
- **Metrics**: Counts total queries, errors, and breaks them down by Table and Statement type (SELECT, INSERT, etc.).
- **Tracer**: Stores the last N (default 200) queries with metadata (duration, SQL, params, error state).
- **Access**: `engine.metrics()` or `engine.get_query_trace()`.

---

## 5. Main Functions Reference

### `create_engine(...)`
Factory function. Best practice is to use this instead of instantiating `Engine` directly.
- **Actions**:
  1. Creates config.
  2. Connects to `postgres` admin DB to creating target DB if missing (`ensure_database`).
  3. Initializes the pool.
  4. Starts health monitor.
  5. Runs schema setup (`ensure_tables`).

### `engine.execute(query, *params)` / `execute_async(...)`
- Executes arbitrary SQL or QueryBuilder objects.
- Returns list of rows (tuples or dict-like objects depending on backend).
- **Safe**: Always uses server-side parametrization.

### `engine.transaction()` / `transaction_async()`
- Returns a `Transaction` object.
- Used to wrap a block of code in a specific DB transaction scope.

### `engine.dispose()`
- Cleanly closes the pool and stops background threads.

---

## 6. Implementation Notes & "Magic"

- **Signal Handling (Sync)**: To enforce timeouts in synchronous code, the engine uses `signal.setitimer`. This is restricted to the main thread to avoid crashes.
- **Parameter Conversion**: The async path does string replacement on the SQL query to convert `%s` to `$N`. This allows the QueryBuilder to be backend-agnostic.
- **Lazy Imports**: The engine imports `Transaction` or `Table` logic only when needed to avoid circular dependencies during initialization.

---

## 7. Automatic Migrations (`ensure_migrations`)

When `ensure_migrations=True`, the engine automatically manages database schema evolution on startup.

### Usage

```python
from psqlmodel import create_engine

engine = create_engine(
    dsn="postgresql://user:pass@localhost:5432/mydb",
    models_path="./app/models",
    
    ensure_migrations=True,           # Enable auto-migrations
    migrations_path="./migrations",   # Where to store migration files
)
```

### Behavior

When the engine starts with `ensure_migrations=True`:

1. **Initialize**: Creates migrations directory and tracking tables if they don't exist.
2. **Detect Drift**: Compares Python models against actual database schema.
3. **Auto-Generate**: If drift is detected, generates a migration file with the changes.
4. **Apply**: Automatically applies all pending migrations.

### Debug Output

```
[ENGINE] Migrations initialized at ./migrations
[ENGINE] Auto-generated migration: 20251212_215117
[ENGINE] Applied 1 migration(s)
```

### `engine.ensure_migrations()`

Can also be called manually:

```python
engine.ensure_migrations()  # Run migrations at any time
```

### Important Notes

> [!IMPORTANT]
> Auto-generated migrations may require review before production use. The system generates commented warnings for potentially destructive changes.

> [!TIP]
> Set `debug=True` to see detailed migration logs during development.
