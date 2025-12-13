# PSQLModel Session Reference

This document provides a comprehensive technical reference for `psqlmodel/core/session.py`.
The **Session** is the primary interface for persistence and object retrieval. It implements the **Unit of Work** pattern and manages the **Identity Map**.

## Overview

- **Integration**: Wraps a database `Transaction` and coordinates with the `Engine`.
- **Identity Map**: Ensures that within a single session, a unique database row is represented by a unique Python object (deduplication by Primary Key).
- **Unit of Work**: Tracks changes (Insert, Update, Delete) and flushes them to the database in order.
- **Context Management**: Designed to be used as a context manager (`with Session(...):`).

---

## 1. Synchronous Session

Use `Session` with a synchronous Engine (`psycopg2`).

### Basic Usage

```python
from psqlmodel import Session

with Session(engine) as session:
    # 1. Fetch
    user = session.get(User, 1)

    # 2. Modify (tracked automatically)
    user.name = "New Name"
    session.add(user)  # Register for persistence

    # 3. Create
    order = Order(user_id=1, total=100)
    session.add(order)

    # 4. Commit (flushes changes + commits transaction)
    session.commit()
```

### Auto-Commit / Auto-Rollback

By default:
- **`auto_commit=False`**: You must call `session.commit()` explicitly.
- **`auto_rollback=True`**: If an exception occurs, the transaction is automatically rolled back.

```python
# Auto-commit mode
with Session(engine, auto_commit=True) as session:
    user = User(name="Auto")
    session.add(user)
# Implicit commit happens here if no error
```

### Atomic Mode

- **`atomic=False`** (Default): `commit()` ends the current transaction but keeps the `Session` open for reuse (starting a new transaction automatically on next operation).
- **`atomic=True`**: `commit()` closes the `Session` entirely.

---

## 2. Asynchronous Session (`AsyncSession`)

Use `AsyncSession` with an asynchronous Engine (`asyncpg`). The API parallels the synchronous one but is non-blocking.

### Basic Usage

```python
from psqlmodel import AsyncSession

async with AsyncSession(engine) as session:
    # 1. Fetch
    user = await session.get(User, 1)

    # 2. Modify
    user.status = "active"
    await session.add(user)

    # 3. Query
    results = await session.exec(Select(Order).Where(Order.user_id == 1))
    orders = await results.all()

    # 4. Commit
    await session.commit()
```

---

## 3. Core API (Common)

Both `Session` and `AsyncSession` share mostly the same API signature (one being awaiting, the other blocking).

### Persistence

- **`add(model)`**: Registers a model for insertion or update.
- **`delete(model)`**: Registers a model for deletion.
- **`flush()`**: Sends pending changes to the database *without* committing the transaction. Useful for getting generated IDs before commit.
- **`commit()`**: Flushes changes + commits DB transaction.
- **`rollback()`**: Rolls back DB transaction.
- **`refresh(model)`**: Reloads model attributes from the database (discarding local changes).

### Querying

- **`get(Model, pk)`**: Fetch by Primary Key using the Identity Map cache.
- **`exec(query)`**: Executes a `SelectQuery`, `InsertQuery`, or raw SQL string.
    - Returns `QueryResult` (sync) or `AsyncQueryResult` (async).
- **`exec_one(query)`**: Helper to get the first result or `None`.
- **`exec_scalar(query)`**: Helper to get the first column of the first row (e.g., `COUNT(*)`).

### Bulk Operations

- **`bulk_insert(models)`**: High-performance multi-row insert (bypasses Identity Map/Dirty Tracking).
- **`bulk_update(models)`**: High-performance multi-row update.

---

## 4. Session Manager (`SessionManager`)

The `SessionManager` uses `contextvars` to make the current session globally accessible within a context, decoupling low-level components (like Query Builder) from explicit session passing.

```python
from psqlmodel import SessionManager

# In a library/helper function:
def get_current_user_count():
    # Automatically finds the active session
    session = SessionManager.require_current() 
    return session.exec_scalar("SELECT COUNT(*) FROM users")

# Usage:
with Session(engine) as session:
    count = get_current_user_count()
```

- **`SessionManager.current()`**: Returns active session or `None`.
- **`SessionManager.require_current()`**: Raises error if no session is active.
- **Async variants**: `.current_async()` / `.require_current_async()`.

---

## 5. Parallel Execution

Sessions provide utilities for parallel queries (readers), utilizing the Engine's connection pool (bypassing the current transaction).

- **`session.parallel_exec(tasks)`**: 
  - Sync: Uses `ThreadPoolExecutor`.
  - Async: Uses `asyncio.gather`.

```python
# Async Example
users, orders = await session.parallel_exec([
    Select(User).all(),
    Select(Order).all()
])
```

## 6. Relationship Loading (`Include`)

The `Session` handles eager loading declared via `.Include()`.

```python
# Sync
users = session.exec(
    Select(User).Include(User.orders)
).all()

# Async
users = await session.exec(
    Select(User).Include(User.orders)
).all()
```

It automatically groups related IDs and performs optimized batch queries (1 query for main model + 1 query per included relation).
