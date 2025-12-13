# PSQLModel Class Reference

This document provides a comprehensive technical reference for `psqlmodel/orm/model.py`, the foundational class for all entities in the ORM.

## Overview

`PSQLModel` is the base class that all your database models must inherit from. It provides:

1.  **Object-Relational Mapping**: Automatically maps class attributes to database columns.
2.  **Persistence Methods**: Built-in `save()` and `delete()` (both sync and async).
3.  **State Management**: Tracks "dirty" fields (changes made to the object) to optimize UPDATE queries.
4.  **Session Integration**: Automatically detects active sessions/transactions to include operations in the current unit of work.

---

## 1. Class Structure & Metadata

Each `PSQLModel` subclass carries metadata injected by the `@table` decorator.

### Internal Attributes
| Attribute | Description |
|-----------|-------------|
| `__tablename__` | Name of the database table. |
| `__schema__` | Database schema (default: `public`). |
| `__columns__` | Dictionary `{'attr_name': Column()}` mapping. |

### Initialization (`__init__`)
```python
user = User(name="Alice", age=30)
```
- **Validation**: Rejects arguments that don't match defined columns (unless `ignore_unknown=True` in update methods).
- **Defaults**: Applies default values defined in `Column(default=...)`.
- **Nullable Checks**: Raises `ValueError` if a non-nullable field is missing (except Primary Keys, which are allowed to be None for auto-generation).

---

## 2. CRUD Operations

The model supports both **Active Record** style (methods on the instance) and **Data Mapper** style (via Session).

### Synchronous Methods
- **`save(conn=None, dsn=None)`**:
    - **Upsert Logic**: Uses `INSERT ... ON CONFLICT (pk) DO UPDATE` to handle both creation and updates.
    - **Optimization**: Only updates fields that have changed (Dirty Tracking).
    - **Session Awareness**: If a `Session` is active, delegates to `session.add(self)`. Otherwise, opens a temporary `psycopg2` connection.
- **`delete(conn=None, dsn=None)`**:
    - Deletes the row based on the Primary Key.
    - requires PK to be set on the instance.

### Asynchronous Methods
- **`save_async(conn=None, dsn=None)`**:
    - Async version using `asyncpg`.
    - Delegates to `AsyncSession` if active.
- **`delete_async(conn=None, dsn=None)`**:
    - Async version of delete.

---

## 3. Dirty Tracking (`DirtyTrackingMixin`)

The model automatically tracks which fields have been modified since loading or saving.

- **Mechanism**: Intercepts `__setattr__`.
- **Benefit**: When calling `save()`, the generated SQL only includes modified columns in the `UPDATE` clause, preventing race conditions on unrelated fields and reducing write load.
- **API**:
    - `_get_dirty_field_names()`: Returns list of changed attributes.
    - `mark_clean()`: Resets tracking (called automatically after save).
    - `update_from_dict(data, partial=True)`: Updates multiple fields while maintaining dirty state.

---

## 4. Session Binding

Models are "session-aware". They can resolve their context in two ways:

1.  **Explicit Binding**: `session.add(model)` sets `model.__session__`.
2.  **Context Context**: `SessionManager.current()` (Thread-local) or `SessionManager.current_async()` (ContextVar).

This allows you to write code like this:
```python
with Session(engine) as session:
    user = User(name="Bob")
    user.save()  # Automatically uses 'session' and its transaction
```

---

## 5. Serialization & IDE Support

### `to_dict()`
Returns a dictionary of column values. Useful for API responses.
```python
data = user.to_dict()
# {'id': 1, 'name': 'Bob', ...}
```

### `from_dict(data)`
Factory method to create an instance from a dictionary, filtering out extra keys.

### IDE Autocompletion
The class uses a clever `__init_subclass__` hook to ensure that `Column` definitions are compatible with type checkers (mypy/pyright) and IDEs (VSCode/PyCharm), preserving standard Python type hints.

---

## 6. Dynamic Attribute Access (`__getattr__`)

To support fluent coding and partial loading, accessing a column that hasn't been loaded yet (e.g., from a partial SELECT) returns `None` instead of raising `AttributeError`, while still alerting on truly non-existent attributes.

---

## Best Practices

1.  **Prefer Sessions**: While `user.save(dsn=...)` works for scripts, always use `Session` in applications for transaction safety.
2.  **Type Hints**: Always use standard Python types (`str`, `int`) alongside `Column`.
    ```python
    name: str = Column()  # Good
    name = Column()       # Bad (IDE won't know it's a string)
    ```
3.  **Partial Updates**: Use `user.update_from_dict(patch_data)` for PATCH endpoints in APIs.
