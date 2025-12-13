# PSQLModel Replication Reference

This document provides a comprehensive technical reference for `psqlmodel/db/replication.py`, which provides native support for PostgreSQL Logical Replication within the ORM.

## Overview

The Replication module allows you to configure and manage:
1.  **Publications** (Publisher side): Define what data to stream.
2.  **Subscriptions** (Subscriber side): Consume data streams from a publisher.
3.  **ReplicationManager**: High-level utility for setup and health checks.

This is NOT for triggers/notifications (use `Subscribe`/`Subscriber` module for that). This module manages **server-to-server** data synchronization.

---

## 1. Publisher Side (`Publication`)

A `Publication` defines a set of tables to be replicated.

### Creating a Publication

```python
from psqlmodel.db.replication import Publication

# Initialize
pub = Publication(engine, "analytics_pub")

# Option A: Publish specific tables
pub.create(
    tables=[User, Order],
    publish_insert=True,
    publish_update=True,
    publish_delete=True
)

# Option B: Publish ALL tables
pub.create(all_tables=True)
```

### Managing Tables

```python
# Add more tables later
pub.add_table(Product, Payment)

# Remove tables
pub.remove_table(User)
```

### Clean Up

```python
pub.drop(cascade=True)
```

---

## 2. Subscriber Side (`Subscription`)

A `Subscription` connects to a publisher connection string and consumes the stream.

### Creating a Subscription

```python
from psqlmodel.db.replication import Subscription

sub = Subscription(engine, "analytics_sub")

sub.create(
    connection_string="host=publisher_host port=5432 user=rep_user password=pass dbname=production",
    publication_name="analytics_pub",
    copy_data=True,      # Initial data sync
    create_slot=True,    # Create replication slot on publisher
    enabled=True
)
```

### Management

```python
# Pause/Resume
sub.disable()
sub.enable()

# Refresh schema (if tables added to publication)
sub.refresh(copy_data=True)

# Drop
sub.drop()
```

---

## 3. High-Level Manager (`ReplicationManager`)

The `ReplicationManager` simplifies the setup by combining validation and creation steps.

### Setup Publisher

```python
from psqlmodel.db.replication import ReplicationManager

# Validates wal_level='logical' before creating
pub = ReplicationManager.setup_publisher(
    engine,
    pub_name="my_app_pub",
    tables=[User, Order]
)
```

### Setup Subscriber

```python
# Auto-creates tables if missing, then subscribes
sub = ReplicationManager.setup_subscriber(
    dest_engine,
    sub_name="my_app_sub",
    publisher_dsn="...",
    pub_name="my_app_pub",
    tables=[User, Order],
    create_tables=True
)
```

---

## 4. Monitoring & Health Checks

The module provides tools to monitor lag and potential issues.

### Publisher Health

```python
health = ReplicationManager.health_check_publisher(pub_engine)

if not health['healthy']:
    print("Issues:", health['issues'])
    print("Warnings:", health['warnings'])

# Detailed Lag Stats
lag_info = ReplicationManager.get_replication_lag(pub_engine)
# Returns list of dicts with: write_lag, flush_lag, replay_lag
```

### Subscriber Health

```python
health = ReplicationManager.health_check_subscriber(sub_engine, "my_app_sub")

# Check table sync states (i=init, d=dumping, s=synced, r=ready)
status = ReplicationManager.get_subscription_status(sub_engine)
```

---

## 5. Requirements

For Logical Replication to work:
1.  **wal_level**: Must be set to `logical` in `postgresql.conf` on the publisher.
2.  **User Permissions**: The replication user must have `REPLICATION` attribute or be superuser.
3.  **Primary Keys**: Replicated tables must have a Primary Key (or REPLICA IDENTITY FULL).
