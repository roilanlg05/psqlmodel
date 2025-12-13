# PSQLModel Column & Expressions Reference

This document provides a comprehensive technical reference for `psqlmodel/orm/column.py`, which handles both the definition of database columns and the construction of detailed SQL expressions used in queries.

## Overview

In `psqlmodel`, a `Column` object serves two purposes:
1.  **Schema Definition**: Defines type, constraints, and default values for DDL.
2.  **Query Construction**: Acts as a builder for SQL expressions (e.g., `User.age > 18`).

The module also provides a rich set of SQL functions (`Count`, `Sum`, `Coalesce`, `Case`, etc.) that interoperate seamlessly with columns.

---

## 1. Defining Columns

### `Column` Class

```python
id: int = Column(primary_key=True)
email: str = Column(nullable=False, unique=True, max_len=255)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_key` | `bool` | `False` | Marks column as Primary Key. |
| `nullable` | `bool` | `True` | Allows NULL values. |
| `default` | `Any` | `None` | Default value (literal) or python callable. |
| `foreign_key` | `str` | `None` | Reference to another table (e.g., `"users.id"`). |
| `unique` | `bool` | `False` | Creates a UNIQUE index/constraint. |
| `index` | `bool` | `False` | Creates a standard index. |
| `max_len` | `int` | `None` | Max length for VARCHAR/CHAR. |
| `on_delete` | `str` | `None` | FK behavior (CASCADE, SET NULL, etc.). |

### Validation
The `validate_value(val)` method enforces `max_len`, `min_value`, etc., at runtime (before DB insertion), providing immediate feedback.

---

## 2. Query Expressions (Operator Overloading)

Columns implement python magic methods to generate SQL expressions safely.

### Comparison
| Python | Generates SQL |
|--------|---------------|
| `User.age == 18` | `users.age = 18` |
| `User.age != 18` | `users.age != 18` |
| `User.age > 18` | `users.age > 18` |
| `User.name.In("a", "b")` | `users.name IN ('a', 'b')` |
| `User.name.NotIn("x")` | `users.name NOT IN ('x')` |

### Arithmetic
| Python | Generates SQL |
|--------|---------------|
| `Item.price * 2` | `items.price * 2` |
| `Item.price + Item.tax` | `items.price + items.tax` |

### Logical
| Python | Generates SQL |
|--------|---------------|
| `(A > 1) & (B < 2)` | `(A > 1 AND B < 2)` |
| `(A > 1) | (B < 2)` | `(A > 1 OR B < 2)` |

---

## 3. SQL Functions & Aggregates

The module exports standard SQL functions wrapper classes.

### Case / When
```python
from psqlmodel import Case

# CASE WHEN age < 18 THEN 'minor' ELSE 'adult' END
Case().When(User.age < 18, "minor").Else("adult").As("status")
```

### Aggregates
- **`Count(col)`**
- **`Sum(col)`**
- **`Avg(col)`**
- **`RowNumber()`** (Window function only)

#### Windows Functions
Aggregates can be transformed into window functions using `.Over()`:

```python
# SUM(amount) OVER (PARTITION BY user_id ORDER BY date)
Sum(Transaction.amount).Over().PartitionBy(Transaction.user_id).OrderBy(Transaction.date)
```

### General Functions
- **`Coalesce(a, b)`**: Return first non-null.
- **`Now()`**: Current timestamp.
- **`Concat(a, b)`**: String concatenation.
- **`Lower(col)` / `Upper(col)`**: String casing.

---

## 4. JSONB Support

PostgreSQL JSONB operations are first-class citizens.

- **`JsonbExtract(col, key)`**: `col->'key'`
- **`JsonbExtractText(col, key)`**: `col->>'key'`
- **`JsonbBuildObject(k, v, ...)`**: Constructs a JSON object.
- **`JsonbAgg(expr)`**: Aggregates rows into a JSON array.

---

## 5. Subqueries & Exists

### Subqueries
You can use a `Select` query as a scalar value:

```python
sub = Select(Post.id).Where(Post.user_id == User.id)
Scalar(sub) # (SELECT posts.id FROM ...)
```

### Exists
```python
Exists(sub)     # EXISTS (SELECT ...)
NotExists(sub)  # NOT EXISTS (SELECT ...)
```

---

## 6. Internal Mechanics (`SQLExpression`)

All query components inherit from `SQLExpression`.
- **`to_sql_params()`**: The core method. Returns a tuple `(sql_string, list_of_params)`.
- **Safety**: Params are **never** interpolated into the string using f-strings inside the library. They are passed separately to the driver (`psycopg2` or `asyncpg`) to prevent SQL Injection.
