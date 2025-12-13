# PSQLModel Table Reference

This document provides a comprehensive technical reference for `psqlmodel/orm/table.py`, which is responsible for defining database schemas, constraints, and indexes using Python classes.

## Overview

The `table` module maps Python classes to PostgreSQL tables. It provides a rich API for:

1.  **Table Definition**: Using the `@table` decorator on `PSQLModel` subclasses.
2.  **Constraints**: Defining Primary Keys, Foreign Keys, Unique constraints, and Check constraints.
3.  **Indexes**: Creating simple, composite, and partial indexes.
4.  **DDL Generation**: Converting Python definitions into raw SQL `CREATE TABLE` statements.

---

## 1. The `@table` Decorator

The core of the ORM's declarative system. It transforms a standard Python class into a database model.

### Signature

```python
@table(
    name: str,
    *,
    schema: str = "public",
    constraints: List[Constraint] = None,
    indexes: List[Index] = None,
    options: TableOptions = None,
    comment: str = None
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | **Required**. The name of the table in PostgreSQL. |
| `schema` | `str` | The database schema (default: `public`). |
| `constraints` | `List[Constraint]` | A list of table-level constraints (FK, Unique, Check). |
| `indexes` | `List[Index]` | A list of table-level indexes (e.g., composite indexes). |
| `options` | `TableOptions` | Advanced PostgreSQL-specific options (tablespace, partitioning). |
| `comment` | `str` | A comment to attach to the table (`COMMENT ON TABLE`). |

### How It Works

1.  **Metadata Attachment**: Stores the provided arguments as metadata on the class (e.g., `__tablename__`, `__schema__`).
2.  **Column Detection**: Iterates over class attributes. If an attribute has a type hint but is not a `Relationship`, it is converted into a `Column` object.
3.  **Type Resolution**: Maps Python types (`int`, `str`, `dict`) to `psqlmodel.types` (e.g., `Integer`, `Text`, `JSONB`).
4.  **IDE Support**: Crucially, it restores `__annotations__` at the end so IDES (VSCode, PyCharm) still see the original types for autocompletion.

---

## 2. Defining Constraints

Constraints enforce data integrity at the database level. They are passed to the `constraints` list in the `@table` decorator.

### `UniqueConstraint`
Ensures values in a column (or set of columns) are unique across the table.
- **Usage**: `UniqueConstraint(User.email)` or `UniqueConstraint("col1", "col2")`.

### `PrimaryKeyConstraint`
Defines a composite primary key. (Single-column PKs are usually defined via `Column(primary_key=True)`).
- **Usage**: `PrimaryKeyConstraint(OrderItem.order_id, OrderItem.product_id)`.

### `ForeignKeyConstraint`
Enforces referential integrity.
- **Args**: `columns`, `references` ("table.column"), `on_delete`, `on_update`.
- **Usage**: `ForeignKeyConstraint([User.role_id], "roles.id", on_delete="CASCADE")`.

### `CheckConstraint`
Validates data against a SQL expression.
- **Usage**: `CheckConstraint("age >= 18")`.

### `ExcludeConstraint`
Support for PostgreSQL `EXCLUDE` constraints (useful for ensuring time ranges don't overlap).
- **Usage**: `ExcludeConstraint("USING gist (room WITH =, during WITH &&)")`.

---

## 3. Defining Indexes (`Index`)

Indexes optimize query performance. Single-column indexes can be defined in `Column(index=True)`, but the `Index` class allows more control.

### Parameters
- `columns`: One or more columns to index.
- `unique`: `bool` (creates a UNIQUE INDEX).
- `method`: `str` (e.g., `"btree"`, `"gin"`, `"gist"`).
- `where`: `str` (Partial index condition).
- `include`: `List[Column]` (Covering index - `INCLUDE` clause).

### Examples
```python
Index(User.email)
Index(User.last_name, User.first_name)
Index(Log.data, method="gin")  # JSONB index
Index(User.active, where="active = true")  # Partial index
```

---

## 4. Advanced Table Options (`TableOptions`)

Controls physical storage and partitioning properties.

- `tablespace`: Specify where the table is stored on disk.
- `unlogged`: `True` to create an `UNLOGGED` table (improves write speed, but data is lost on crash).
- `partition_by`: Defines partitioning strategy (e.g., `RANGE (created_at)`).
- `inherits`: List of parent tables for PostgreSQL inheritance.

---

## 5. DDL Generation (`generate_table_ddl`)

The `generate_table_ddl(model_cls)` function is the engine that converts the Model class into SQL.

### Process:
1.  **Schema Creation**: Generates `CREATE SCHEMA IF NOT EXISTS`.
2.  **Column Definitions**: Iterates over `__columns__`, calling `col.ddl()` for each.
3.  **Inline Indexes**: Converts `Column(index=True)` into `Index` objects.
4.  **Table Constraints**: Appends constraints defined in the decorator.
5.  **Table Creation**: Assembles the `CREATE TABLE` statement using column defs and constraints.
6.  **Index Creation**: Generates `CREATE INDEX` statements for all collected indexes.
7.  **Comments**: Generates `COMMENT ON TABLE` if provided.

This function returns a `List[str]` of SQL statements, which the `Engine` executes in order during initialization.

---

## Example Usage

```python
from psqlmodel import PSQLModel, table, Column
from psqlmodel.table import UniqueConstraint, Index, TableOptions

@table(
    name="users",
    schema="identity",
    constraints=[
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint("age >= 0", name="ck_users_valid_age")
    ],
    indexes=[
        Index("created_at", method="btree"),
        Index("metadata", method="gin")
    ],
    options=TableOptions(unlogged=False),
    comment="Main user registry"
)
class User(PSQLModel):
    id: int = Column(primary_key=True)
    email: str = Column(nullable=False)
    age: int = Column(default=18)
    metadata: dict = Column(default={})
    created_at: datetime = Column(default=lambda: datetime.now())
```
