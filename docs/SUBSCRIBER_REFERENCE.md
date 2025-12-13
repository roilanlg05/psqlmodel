# PSQLModel Subscriber Reference

This document provides a comprehensive technical reference for `psqlmodel/db/subscriber.py`, which implements a robust Real-Time subscription system using PostgreSQL's `LISTEN/NOTIFY`.

## Overview

The `Subscribe` features allow your application to react to database changes in real-time without polling.
It works by:
1.  Automatically creating `AFTER INSERT/UPDATE/DELETE` triggers.
2.  Using a dedicated thread (Sync) or asyncio Task (Async) to `LISTEN`.
3.  Invoking your Python callback with the JSON payload whenever a change occurs.

### Usage Example

```python
from psqlmodel import Subscribe

# Create the factory from the engine
sub = Subscribe.engine(engine)

# Define callback
def on_user_change(payload):
    print(f"User changed: {payload['event']} - ID: {payload['new']['id']}")

# Start listening
# Monitors USERS table for INSERT and UPDATE events
sub(User).OnEvent("insert", "update").Exec(on_user_change).Start()
```

---

## 1. Subscribe Factory

The entry point is `Subscribe.engine(engine)`.

### Configuration (`SubscriberConfig`)
You can pass custom configuration when creating the factory:

```python
sub = Subscribe.engine(
    engine,
    use_engine_pool=True,       # Reuse engine connections (recommended)
    auto_delete_trigger=True,   # Cleanup DB triggers on exit
    daemon_mode=False,          # Run listener in background thread
    listen_timeout=1.0          # Poll interval
)
```

---

## 2. Defining Subscriptions

### Targets
- **Model (Table)**: Monitor all columns.
  ```python
  sub(User)
  ```
- **Columns**: Monitor specific columns (only fires if those columns change).
  ```python
  sub(User.status, User.email)
  ```

### Events (`.OnEvent()`)
- `"insert"`
- `"update"`
- `"delete"`
- `"change"` (alias for insert + update + delete)

### Direct Channel (`.OnChannel()`)
Listen to a custom raw PostgreSQL channel (manually sent NOTIFYs) instead of table changes.

```python
sub().OnChannel("my_custom_channel").Exec(handler).Start()
```

### Callback (`.Exec()`)
The function must accept a single argument: `payload` (dict).

---

## 3. Payload Structure

The payload received in the callback is a dictionary containing:

| Key | Description |
|-----|-------------|
| `event` | "insert", "update", or "delete". |
| `schema` | Schema name (e.g., "public"). |
| `table` | Table name. |
| `pk_name`| Name of the Primary Key column. |
| `old` | Dict of old values (for UPDATE/DELETE). `None` for INSERT. |
| `new` | Dict of new values (for INSERT/UPDATE). `None` for DELETE. |

**Example Payload:**
```json
{
  "event": "update",
  "schema": "public",
  "table": "users",
  "pk_name": "id",
  "old": {"id": 1, "name": "Old Name"},
  "new": {"id": 1, "name": "New Name"}
}
```

---

## 4. Async Support (`StartAsync`)

The subscriber supports full AsyncIO integration.

```python
# Instead of .Start(), use await .StartAsync()
await sub(User).OnEvent("insert").Exec(async_handler).StartAsync()
```

- **`StartAsync()`**: Creates triggers via `AsyncTransaction` and spawns a background `asyncio.Task` for the listener loop.
- **Parallel DDL**: If `config.parallel_ddl=True`, it creates triggers on multiple tables concurrently.

---

## 5. Internals

- **Triggers**: Creating a subscription generates a uniquely named PostgreSQL function and trigger (e.g., `__sub_users_insert_a1b2c3`).
- **Cleanup**: Calling `.Stop()` (or on interpreter exit) removes these temporary triggers from the database to keep schema clean.
- **Connection Management**:
    - **Shared Pool (`use_engine_pool=True`)**: Borrows a connection from the engine.
    - **Dedicated Pool (`use_engine_pool=False`)**: Maintains its own pool (useful for high-throughput isolation).
