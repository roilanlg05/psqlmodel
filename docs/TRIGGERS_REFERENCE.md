# PSQLModel Triggers Reference

This document provides a comprehensive technical reference for `psqlmodel/db/triggers.py`, the module responsible for defining and deploying database triggers specifically designed for PostgreSQL using a fluent Python DSL.

## Overview

The Triggers module allows you to write Python functions and "compile" them into PostgreSQL triggers (using `PL/Python3u` or `PL/pgSQL`).

Key features:
1.  **Fluent DSL**: `Trigger(Model).BeforeInsert()...`
2.  **PL/Python Integration**: Write trigger logic in standard Python, which runs *inside* the database.
3.  **Complex Logic Support**: Access `Old` and `New` row states, perform HTTP requests, publish to Redis/Kafka directly from the DB.
4.  **Auto-Deployment**: The Engine automatically creates these triggers when `ensure_tables()` runs.

---

## 1. Defining Triggers

Triggers are defined using the `TriggerBuilder` (accessed via `Trigger()`) and attached to models via the `@trigger` decorator.

### Basic Syntax

```python
from psqlmodel import trigger, Trigger, Old, New, PSQLModel, table, Column

# 1. Define the trigger logic using the DSL
validation_trigger = (
    Trigger(User)
        .BeforeInsert()
        .BeforeUpdate()
        .ForEachRow()
        .Exec(validate_user_data)
)

# 2. Attach it to the model
@trigger(validation_trigger)
@table(name="users")
class User(PSQLModel):
    ...
```

---

## 2. Trigger Builder API (`TriggerBuilder`)

### Timing & Events
- **`.BeforeInsert()`**
- **`.AfterInsert()`**
- **`.BeforeUpdate(columns=None)`**: If `columns` provided (e.g., `User.email`), becomes `UPDATE OF email`.
- **`.AfterUpdate(columns=None)`**
- **`.BeforeDelete()`**
- **`.AfterDelete()`**

Note: You can chain multiple events (e.g., `.BeforeInsert().BeforeUpdate()`) to create a multi-event trigger (`OR` condition).

### Granularity
- **`.ForEachRow()`**: Fires once for every row modified (standard for data validation/audit).
- **`.ForEachStatement()`**: Fires once per SQL command.

### Conditions (`WHEN`)
- **`.When(condition)`**: Adds a `WHEN` clause to the trigger. The trigger function only runs if this SQL condition is true.
- **`Old` / `New`**: Magic objects representing the row state before and after the change.

```python
# Only run if email actually changed
.When(Old.email).IsDistinctFrom(New.email)

# Only if status is 'active'
.When(New.status == 'active')
```

### Execution
- **`.Exec(func)`**: Specifies the Python function to execute.
- **`.Language("plpgsql" | "plpython3u")`**: Explicitly sets the language. Defaults to `plpython3u`.

---

## 3. Writing Trigger Functions (PL/Python)

The function you pass to `.Exec()` is transpiled into a PostgreSQL Function.

### Context Availability
Inside your function, you have access to:
- **`Old`**: Dictionary-like object with previous values (None for INSERT).
- **`New`**: Dictionary-like object with new values (None for DELETE).
- **`TD`**: Technical dictionary with low-level trigger info (`event`, `table_name`, `when`).

### Helper Functions
The module injects several helpers into the PL/Python environment:
- **`TG_OP()`**: Returns "INSERT", "UPDATE", "DELETE".
- **`plpy`**: The standard PostgreSQL Python driver (for `plpy.execute`, `plpy.notice`).

### Integrations (Side Effects)
Special classes are injected to perform external actions safely:

1.  **`Notify(channel).Message(payload)`**: Sends a PostgreSQL `NOTIFY`.
2.  **`HttpPost(url, body=...)`**: Performs a synchronous HTTP POST (via `requests`).
3.  **`RedisPublish(channel, payload)`**: Publishes to Redis (via `redis-py`).
4.  **`KafkaProduce(topic, value=...)`**: Sends message to Kafka.

### Example Function

```python
def audit_log_logic(Old, New):
    import json
    
    # Logic runs INSIDE PostgreSQL
    if New.balance < 0:
        plpy.error("Balance cannot be negative!")
        
    # Send notification if high value transaction
    if New.amount > 10000:
        return Notify("high_value_tx").Message({"user": New.id, "amt": New.amount})
```

---

## 4. SQL Generation

When `engine.ensure_tables()` runs (or you call `get_trigger_sql()` manually), the system:

1.  Inspects the Python function source code (`inspect.getsource`).
2.  Injects the Preamble (imports, helper classes `Notify`, `HttpPost`, etc.).
3.  Wraps logic in a `CREATE FUNCTION ... LANGUAGE plpython3u` block.
4.  Generates the `CREATE TRIGGER` statement linking the event to the function.

### Limitations vs. Standard Python
- The code must be valid **independent** code (cannot access global variables outside the function scope, as it runs in a separate process/interpreter inside Postgres).
- Must import its own dependencies (`import json`, etc.) inside the function body.

---

## 5. Deployment

Triggers are deployed automatically if `ensure_tables=True` is set in the Engine. The DDL generation ensures:
- `CREATE EXTENSION IF NOT EXISTS plpython3u`
- `DROP TRIGGER IF EXISTS ...` replacement strategy (idempotent operation).

### Naming Convention
Trigger names are auto-generated to avoid collisions: `{table}_{func_name}_{hash}`. You can override this with `.Name("my_custom_name")`.
