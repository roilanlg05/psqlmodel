# PSQLModel Types Reference

This document provides a comprehensive technical reference for `psqlmodel/orm/types.py`, which defines the SQL type system for mapping Python types to PostgreSQL DDL.

## Overview

You use these types in your model definitions as Python type hints. The `Column` machinery interprets them to generate the correct `CREATE TABLE` syntax.

### Usage Example

```python
from psqlmodel import PSQLModel, table, Column
from psqlmodel.types import varchar, integer, jsonb, timestamp, uuid

@table(name="users")
class User(PSQLModel):
    id: uuid = Column(primary_key=True)  # UUID DEFAULT gen_random_uuid()
    name: varchar = Column(max_len=100)  # VARCHAR(100)
    score: integer = Column()            # INTEGER
    data: jsonb = Column()               # JSONB
    created_at: timestamp = Column()     # TIMESTAMP
```

---

## 1. Numeric Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `smallint` | `SMALLINT` | `int` | 2 bytes |
| `integer` | `INTEGER` | `int` | 4 bytes, Standard |
| `bigint` | `BIGINT` | `int` | 8 bytes |
| `serial` | `SERIAL` | `int` | Auto-increment 4-byte |
| `bigserial`| `BIGSERIAL`| `int` | Auto-increment 8-byte |
| `numeric(p, s)` | `NUMERIC(p, s)` | `Decimal` | Exact precision e.g. `numeric(10,2)` |
| `real` | `REAL` | `float` | 4-byte float |
| `double` | `DOUBLE PRECISION` | `float` | 8-byte float |
| `money` | `MONEY` | `Decimal` | Currency |

---

## 2. Text Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `varchar` | `VARCHAR` | `str` | Use `max_len` in `Column` |
| `char` | `CHAR` | `str` | Fixed length, use `max_len` |
| `text` | `TEXT` | `str` | Unlimited length |

---

## 3. Date & Time Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `date` | `DATE` | `datetime.date` | |
| `time(tz=False)` | `TIME` / `TIME WITH TZ` | `datetime.time` | |
| `timestamp` | `TIMESTAMP` | `datetime` | |
| `timestamptz` | `TIMESTAMP WITH TZ` | `datetime` | Timezone aware |
| `interval` | `INTERVAL` | `timedelta` | Duration |

---

## 4. JSON Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `json` | `JSON` | `dict`/`list` | Text-based storage |
| `jsonb` | `JSONB` | `dict`/`list` | **Recommended**. Binary, indexable |

---

## 5. Network Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `inet` | `INET` | `ipaddress` | IPv4/IPv6 host |
| `cidr` | `CIDR` | `ipaddress` | Network block |
| `macaddr` | `MACADDR` | `str` | MAC Address |

---

## 6. Special Types

| Type Class | SQL DDL | Python | Note |
|------------|---------|--------|------|
| `uuid` | `UUID` | `uuid.UUID` | Auto-generates default |
| `boolean` | `BOOLEAN` | `bool` | True/False |
| `bytea` | `BYTEA` | `bytes` | Binary data |
| `array(T)` | `T[]` | `list` | e.g. `array(integer)` |

---

## 7. Geometric & Range Types

We support advanced Postgres types:
- **Geometry**: `point`, `line`, `lseg`, `box`, `path`, `polygon`, `circle`.
- **Ranges**: `int4range`, `int8range`, `numrange`, `tsrange`, `tstzrange`, `daterange`.
- **Search**: `tsvector`, `tsquery` (Full Text Search).

---
