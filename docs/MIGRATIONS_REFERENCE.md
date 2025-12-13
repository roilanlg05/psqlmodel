# PSQLModel Migration System Reference

This document provides a comprehensive technical reference for `psqlmodel/migrations/`, the schema migration system for PSQLModel ORM.

## Overview

The migration system provides:
- **Hash-based schema tracking**: SHA256 hashes of tables, triggers, and indexes
- **Detailed drift detection**: Compare Python models vs actual database state
- **Versioned migrations**: Timestamped migration files with `up`/`down` methods
- **CLI integration**: `python -m psqlmodel migrate <command>`
- **Engine-style logging**: Optional external logger support

---

## 1. Quick Start

### Initialize

```bash
# Set environment variables
export PSQL_USER=myuser
export PSQL_PASSWORD=mypass
export PSQL_HOST=localhost
export PSQL_DB=mydb

# Initialize migrations
python -m psqlmodel migrate init
```

### Check Status

```bash
python -m psqlmodel migrate status
```

Output:
```
==================================================
📊 MIGRATION STATUS
==================================================
   Initialized:      ✅ Yes
   Migrations path:  /path/to/project/migrations
   Current version:  20251212_191130
   Applied:          3
   Pending:          1
   Schema drift:     ✅ No
==================================================
```

### Generate Migration

```bash
python -m psqlmodel migrate autogenerate "Add email column to users"
```

### Apply Migrations

```bash
python -m psqlmodel migrate upgrade
```

### Rollback

```bash
python -m psqlmodel migrate downgrade -1
```

### Automatic Schema Drift Detection

The engine automatically detects differences between your models and the database on startup:

```python
# Enabled by default
engine = create_engine("postgresql://user:pass@localhost/db")
# Warning: [PSQLMODEL] Schema drift detected: ~public.users. Run '...'

# Disable if needed
engine = create_engine("...", check_schema_drift=False)
```

**Warning symbols:**
| Symbol | Meaning |
|--------|---------|
| `+table` | New table (in model, not in DB) |
| `-table` | Removed table (in DB, not in model) |
| `~table` | Modified table (columns changed) |

**What triggers `~table` (modified):**
- Column type changed (e.g. `uuid` → `str`)
- Column added/removed
- Constraint changed (nullable, unique, primary key)
- Default value changed

---

## 2. Python API

### MigrationManager

```python
from psqlmodel import create_engine, MigrationManager, MigrationConfig

engine = create_engine("postgresql://user:pass@localhost/db")

# Default configuration
manager = MigrationManager(engine)

# Custom configuration
config = MigrationConfig(
    migrations_path="./db/migrations",  # Custom path
    debug=True,                          # Enable logging
    logger=my_logger_func,               # External logger
    fail_on_drift=False,                 # Warning vs Error
)
manager = MigrationManager(engine, config)

# Initialize (creates directory + DB tables)
manager.init()

# Check status
status = manager.status()
print(f"Pending: {status.pending_count}")

# Validate schema
warnings = manager.validate()
for w in warnings:
    print(w)

# Auto-generate migration
migration = manager.autogenerate("Add users table")

# Apply migrations
manager.upgrade()

# Rollback
manager.downgrade("-1")

# History
for record in manager.history():
    print(f"{record['version']}: {record['message']}")
```

---

## 3. Configuration (`MigrationConfig`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `migrations_path` | `str` | `./migrations/` | Directory for migration files |
| `state_table_name` | `str` | `_psqlmodel_schema_state` | DB table for current hashes |
| `history_table_name` | `str` | `_psqlmodel_migrations` | DB table for history |
| `state_file_name` | `str` | `.schema_state.json` | Local state file |
| `auto_detect_changes` | `bool` | `True` | Check drift on init |
| `fail_on_drift` | `bool` | `False` | Raise error vs warning |
| `debug` | `bool` | `False` | Enable verbose output |
| `logger` | `Callable` | `None` | External logger function |
| `version_format` | `str` | `%Y%m%d_%H%M%S` | Timestamp format for versions |

---

## 4. Migration File Structure

Generated migrations have this structure:

```python
# migrations/20251212_191130_add_users_table.py

from psqlmodel.migrations import Migration


class AddUsersTable(Migration):
    """Add users table"""
    
    version = "20251212_191130"
    message = "Add users table"
    depends_on = None  # Or previous version
    
    def up(self, engine):
        """Apply the migration."""
        engine.execute("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL
            )
        """)
    
    def down(self, engine):
        """Revert the migration."""
        engine.execute("DROP TABLE IF EXISTS users CASCADE")
```

---

## 5. Schema Hashing

The `SchemaHasher` generates deterministic SHA256 hashes for schema objects:

```python
from psqlmodel.migrations.hasher import SchemaHasher

hasher = SchemaHasher()

# Hash a single model
hash = hasher.hash_table(User)

# Hash all models
expected = hasher.hash_models([User, Order, Product])

# Hash actual database state
actual = hasher.hash_database_schema(engine)
```

### Hash Structure

```json
{
    "tables": {
        "public.users": "sha256:abc123...",
        "public.orders": "sha256:def456..."
    },
    "triggers": {
        "public.users.update_timestamp": "sha256:..."
    },
    "indexes": {
        "public.users.idx_email": "sha256:..."
    },
    "meta": {
        "version": "1.0",
        "computed_at": "2025-12-12T19:00:00"
    }
}
```

---

## 6. Schema Diffing

The `SchemaDiffer` compares expected vs actual schema:

```python
from psqlmodel.migrations.differ import SchemaDiffer

differ = SchemaDiffer()
diff = differ.compare(expected, actual)

if diff.has_changes:
    print(f"Total changes: {diff.total_changes}")
    
    # Detailed warnings
    for warning in differ.format_warnings(diff):
        print(warning)
```

### Diff Result

```python
diff.new_tables       # Tables in models, not in DB
diff.removed_tables   # Tables in DB, not in models
diff.modified_tables  # Tables with hash mismatch

diff.new_triggers
diff.removed_triggers
diff.modified_triggers

diff.new_indexes
diff.removed_indexes
diff.modified_indexes
```

---

## 7. State Management

State is persisted in two places:

1. **Local file** (`migrations/.schema_state.json`): For offline/fast comparison
2. **Database table** (`_psqlmodel_schema_state`): For consistency across developers

The `StateManager` handles synchronization:

```python
from psqlmodel.migrations.state import StateManager

state_manager = StateManager(engine, config)

# Check consistency
warnings = state_manager.validate_state_consistency()

# Sync from DB to local
state_manager.sync_from_db()

# Sync from local to DB
state_manager.sync_to_db()
```

---

## 8. Exceptions

| Exception | Description |
|-----------|-------------|
| `MigrationError` | Base exception |
| `SchemaDriftError` | Schema mismatch detected |
| `MigrationNotFoundError` | Migration file missing |
| `MigrationAlreadyAppliedError` | Duplicate application |
| `MigrationDependencyError` | Dependency not applied |
| `RollbackError` | Rollback failed |
| `StateCorruptionError` | Local/DB state mismatch |

---

## 9. CLI Commands

```bash
python -m psqlmodel migrate <command> [options]
```

### Core Commands

| Command | Description |
|---------|-------------|
| `init` | Create migrations directory and state tables |
| `status` | Show current migration status |
| `autogenerate "msg"` | Generate migration from diff |
| `upgrade [version]` | Apply pending migrations |
| `downgrade [target]` | Revert migrations |
| `history` | Show applied migrations |

### Alembic-Compatible Commands

| Command | Description |
|---------|-------------|
| `check` | Check for schema changes (exit code 1 if found) |
| `current` | Show current migration version |
| `stamp <ver>` | Mark version as applied without running |
| `sql [upgrade]` | Generate SQL without executing |

### Advanced Commands

| Command | Description |
|---------|-------------|
| `heads` | Show all migration heads (branches) |
| `merge "msg"` | Merge multiple heads into one |
| `squash "msg"` | Squash migrations into one |
| `dry-run` | Preview migrations without applying |
| `verify` | Check migration file integrity |
| `seed` | Load seed data from JSON/YAML |

### Profile Management

| Action | Description |
|--------|-------------|
| `users --list` | List all saved profiles |
| `users --save <name>` | Save a connection profile |
| `users --remove <name>` | Remove a saved profile |
| `users --use <name>` | Set a profile as default |

---

## 10. Enterprise Features

### Transactional DDL

```bash
# Default: wrapped in transaction with auto-rollback
python -m psqlmodel migrate upgrade

# Disable for DDL that can't run in transactions
python -m psqlmodel migrate upgrade --no-transaction
```

### Migration Locking

Uses PostgreSQL advisory locks to prevent concurrent execution:

```bash
# Default: acquires lock before upgrade
python -m psqlmodel migrate upgrade

# Disable locking
python -m psqlmodel migrate upgrade --no-lock
```

### Checksum Validation

```bash
# Verify no migration files were modified after apply
python -m psqlmodel migrate verify
```

### Async Migrations

```python
from psqlmodel.migrations import MigrationManager

manager = MigrationManager(engine)
job_id = manager.upgrade_async()

# Check status later
status = manager.get_async_status(job_id)
# {"status": "running|completed|failed", "count": 3}
```

---

## 11. Squash & Seed

### Squash Migrations

Combine multiple migrations into one:

```bash
# Squash all migrations
python -m psqlmodel migrate squash "Initial schema"

# Squash specific range
python -m psqlmodel migrate squash "Initial schema" --from 20251201_000000 --to 20251210_000000
```

### Data Seeding

Create `migrations/seeds/dev.json`:

```json
{
  "users": [
    {"id": 1, "name": "Admin", "email": "admin@example.com"}
  ],
  "roles": [
    {"id": 1, "name": "admin"},
    {"id": 2, "name": "user"}
  ]
}
```

Load seed data:

```bash
# Load dev seeds
python -m psqlmodel migrate seed -e dev

# Load specific file
python -m psqlmodel migrate seed -f path/to/seeds.json
```

---

## 12. CI/CD Integration

### GitHub Actions

File: `.github/workflows/migrations.yml`

```yaml
name: Database Migrations
on: [push, pull_request]

jobs:
  check-migrations:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
    steps:
      - uses: actions/checkout@v4
      - run: pip install psycopg && pip install -e .
      - run: python -m psqlmodel migrate check
      - run: python -m psqlmodel migrate verify
```

### GitLab CI

A complete GitLab CI template is available at `psqlmodel/scripts/.gitlab-ci.yml`.

**Installation:**
```bash
# Copy template to your project root
cp psqlmodel/scripts/.gitlab-ci.yml .gitlab-ci.yml
```

**Configuration:**

File: `.gitlab-ci.yml`

```yaml
stages:
  - check
  - migrate

variables:
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
  POSTGRES_DB: test_db

# Check migrations on PRs/MRs
check-migrations:
  stage: check
  image: python:3.11
  services:
    - postgres:15
  variables:
    DB_HOST: postgres
    DB_PORT: 5432
    DB_USER: postgres
    DB_PASSWORD: postgres
    DB_NAME: test_db
  script:
    - pip install psycopg
    - pip install -e .
    - python -m psqlmodel migrate check
    - python -m psqlmodel migrate verify
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_COMMIT_BRANCH == "develop"

# Apply migrations to production (manual)
apply-migrations:
  stage: migrate
  image: python:3.11
  variables:
    DB_HOST: $PROD_DB_HOST
    DB_PORT: $PROD_DB_PORT
    DB_USER: $PROD_DB_USER
    DB_PASSWORD: $PROD_DB_PASSWORD
    DB_NAME: $PROD_DB_NAME
  script:
    - pip install psycopg
    - pip install -e .
    - python -m psqlmodel migrate upgrade
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  when: manual
  environment:
    name: production
```

**GitLab CI Variables (Settings > CI/CD > Variables):**
| Variable | Description |
|----------|-------------|
| `PROD_DB_HOST` | Production database host |
| `PROD_DB_PORT` | Production database port |
| `PROD_DB_USER` | Production database user |
| `PROD_DB_PASSWORD` | Production database password (masked) |
| `PROD_DB_NAME` | Production database name |

### Pre-commit Hook

Automatically blocks commits if migrations are out of sync.

**Installation:**
```bash
# Copy hook to Git
cp psqlmodel/scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**What it does:**
1. Runs `migrate check` - blocks if schema changes without migration
2. Runs `migrate verify` - blocks if migration files were modified after apply

**Skip in emergencies:**
```bash
git commit --no-verify -m "Emergency fix"
```

---

## 13. Database Tables

```sql
CREATE TABLE _psqlmodel_schema_state (
    object_type VARCHAR(50) NOT NULL,
    object_name VARCHAR(255) NOT NULL,
    hash VARCHAR(64) NOT NULL,
    PRIMARY KEY (object_type, object_name)
);

CREATE TABLE _psqlmodel_migrations (
    version VARCHAR(50) PRIMARY KEY,
    message TEXT,
    applied_at TIMESTAMP DEFAULT NOW(),
    checksum VARCHAR(64) NOT NULL,
    execution_time_ms INTEGER,
    rolled_back_at TIMESTAMP
);
```

---

## 14. Profile Storage

Profiles stored in `~/.psqlmodel/config.json`:

```json
{
  "profiles": {
    "mydb": {
      "username": "user",
      "password": "pass",
      "host": "localhost",
      "database": "myapp"
    }
  },
  "default": "mydb"
}
```
