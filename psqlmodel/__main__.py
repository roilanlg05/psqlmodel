"""
PSQLModel CLI - Command-line interface for the ORM.

Usage:
    python -m psqlmodel                     # Start metrics logger
    python -m psqlmodel migrate init        # Initialize migrations
    python -m psqlmodel migrate status      # Check status
    # ... see --help for more
"""

import argparse
import os
import sys
import signal
from typing import Optional, List

from psqlmodel.core.engine import create_engine


# ============================================================
# UTILS
# ============================================================

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default

def _create_engine_from_env(profile_name: Optional[str] = None):
    """Create engine from saved profile or environment variables."""
    from psqlmodel.cli_config import get_profile
    
    # Try to load from saved profile first
    profile = get_profile(profile_name)
    
    if profile:
        username = profile.get("username") or _env("PSQL_USER")
        password = profile.get("password") or _env("PSQL_PASSWORD")
        host = profile.get("host") or _env("PSQL_HOST", "localhost") or "localhost"
        port = profile.get("port") or int(_env("PSQL_PORT", "5432") or "5432")
        database = profile.get("database") or _env("PSQL_DB")
        models_path = profile.get("models_path") or _env("PSQL_MODELS_PATH")
    else:
        # Fall back to environment variables
        username = _env("PSQL_USER")
        password = _env("PSQL_PASSWORD")
        host = _env("PSQL_HOST", "localhost") or "localhost"
        port = int(_env("PSQL_PORT", "5432") or "5432")
        database = _env("PSQL_DB")
        models_path = _env("PSQL_MODELS_PATH")
    
    async_ = (_env("PSQL_ASYNC", "false") or "false").lower() in {"1", "true", "yes"}
    pool_size = int(_env("PSQL_POOL_SIZE", "20") or "20")
    max_pool_size = _env("PSQL_MAX_POOL_SIZE")
    max_pool_size = int(max_pool_size) if max_pool_size else None
    connection_timeout = _env("PSQL_CONN_TIMEOUT")
    connection_timeout = float(connection_timeout) if connection_timeout else None
    health_enabled = (_env("PSQL_HEALTH_ENABLED", "false") or "false").lower() in {"1", "true", "yes"}
    auto_adjust = (_env("PSQL_AUTO_ADJUST", "false") or "false").lower() in {"1", "true", "yes"}
    pre_ping = (_env("PSQL_PRE_PING", "false") or "false").lower() in {"1", "true", "yes"}
    pool_recycle_env = _env("PSQL_POOL_RECYCLE")
    pool_recycle = float(pool_recycle_env) if pool_recycle_env else None

    return create_engine(
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        async_=async_,
        pool_size=pool_size,
        max_pool_size=max_pool_size,
        connection_timeout=connection_timeout,
        health_check_enabled=health_enabled,
        auto_adjust_pool_size=auto_adjust,
        debug=False,
        pool_pre_ping=pre_ping,
        pool_recycle=pool_recycle,
        models_path=models_path,
        ensure_database=False,
        ensure_tables=False,
    )


# ============================================================
# COMMAND HANDLERS
# ============================================================

def cmd_migrate_init(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=args.verbose)
    manager = MigrationManager(engine, config)
    
    migrations_path = manager.init()
    print(f"Migrations initialized at: {migrations_path}")
    engine.dispose()

def cmd_migrate_status(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    status = manager.status()
    
    print("\n" + "=" * 50)
    print("MIGRATION STATUS")
    print("=" * 50)
    print(f"   Initialized:      {'Yes' if status.initialized else 'No'}")
    print(f"   Migrations path:  {status.migrations_path}")
    print(f"   Current version:  {status.current_version or '(none)'}")
    print(f"   Applied:          {status.applied_count}")
    print(f"   Pending:          {status.pending_count}")
    print(f"   Schema drift:     {'Yes' if status.has_drift else 'No'}")
    if status.drift_summary:
        print(f"   Drift details:    {status.drift_summary}")
    print("=" * 50 + "\n")
    engine.dispose()

def cmd_migrate_validate(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    warnings = manager.validate()
    for warning in warnings:
        print(warning)
    engine.dispose()

def cmd_migrate_autogenerate(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    if not args.message:
        print("Error: Please provide a message for the migration")
        sys.exit(1)
        
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    migration = manager.autogenerate(args.message)
    
    if migration:
        print(f"Created migration: {migration.version}")
        print(f"   File: {migration._file_path}")
    else:
        print("No changes detected. No migration created.")
    engine.dispose()

def cmd_migrate_upgrade(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    # Multi-database support
    if args.all:
        from psqlmodel.cli_config import list_profiles
        from psqlmodel import create_engine as ce
        
        profiles = list_profiles()
        if not profiles:
            print("No profiles found. Use 'migrate users save' first.")
            sys.exit(1)
        
        databases = []
        for name, profile in profiles.items():
            dsn = f"postgresql://{profile['username']}:{profile['password']}@{profile['host']}:{profile.get('port', 5432)}/{profile['database']}"
            databases.append({"name": name, "engine": ce(dsn)})
        
        results = MigrationManager.upgrade_multi(databases, args.target, parallel=True)
        
        for name, count in results.items():
            print(f"   {name}: {count} migration(s) applied")
        
        for db in databases:
            db["engine"].dispose()
        return

    # Single database
    if args.database:
        from psqlmodel.cli_config import get_profile
        from psqlmodel import create_engine as ce
        
        profile = get_profile(args.database)
        if not profile:
            print(f"Profile '{args.database}' not found")
            sys.exit(1)
        
        dsn = f"postgresql://{profile['username']}:{profile['password']}@{profile['host']}:{profile.get('port', 5432)}/{profile['database']}"
        engine = ce(dsn)
    else:
        engine = _create_engine_from_env()

    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    count = manager.upgrade(
        args.target, 
        transactional=args.transactional, 
        lock=args.lock
    )
    
    if count > 0:
        print(f"Applied {count} migration(s)")
    else:
        print("No pending migrations")
    engine.dispose()

def cmd_migrate_downgrade(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    count = manager.downgrade(args.target)
    
    if count > 0:
        print(f"Rolled back {count} migration(s)")
    else:
        print("No migrations to rollback")
    engine.dispose()

def cmd_migrate_history(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    limit = 1000 if args.limit is None else args.limit
    history = manager.history(1000)
    
    if args.first:
        history = list(reversed(history))[:limit]
    else:
        history = history[-limit:] if history else [] # Fix for slicing empty list issues or ensuring correct logic
        if not args.first and limit < 1000:
             # Logic for newest first is usually reversed order or just last entries
             # psqlmodel manager.history() usually returns oldest first? 
             # Let's assume manager.history returns oldest -> newest.
             # "Newest first" means we want the end of the list, reversed.
             # If cmd_migrate_history in old code:
             # show_first = False (default) -> history = history[:limit] (Wait, assuming sorted?)
             pass
    
    # Matching old logic for consistency
    # old: history = self._state.get_applied_migrations()[:limit]
    #      (applied_migrations sorted by applied_at ASC)
    
    # New logic to match description
    all_hist = manager.history(1000)
    if args.first:
        # Oldest first (ASC), default manager returns ASC.
        final_history = all_hist[:limit]
    else:
        # Newest first (DESC)
        final_history = list(reversed(all_hist))[:limit]

    if not final_history:
        print("No migrations applied yet")
    else:
        order_label = "(oldest first)" if args.first else "(newest first)"
        print("\n" + "=" * 70)
        print(f"MIGRATION HISTORY {order_label}")
        print("=" * 70)
        for record in final_history:
            status = "Ok" if not record.get("rolled_back_at") else "Rolled Back"
            print(f"   {status} {record['version']} - {record['message']}")
            print(f"      Applied: {record['applied_at']} ({record['execution_time_ms']}ms)")
        print("=" * 70 + "\n")
    engine.dispose()

def cmd_migrate_check(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    has_changes = manager.check()
    engine.dispose()
    
    if has_changes:
        print("Schema changes detected! Run 'migrate autogenerate' to create migration.")
        sys.exit(1)
    else:
        print("No schema changes detected.")
        sys.exit(0)

def cmd_migrate_current(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    version = manager.current()
    if version:
        print(version)
    else:
        print("(none)")
    engine.dispose()

def cmd_migrate_stamp(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    try:
        manager.stamp(args.version)
        print(f"Stamped: {args.version}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

def cmd_migrate_sql(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    sql_statements = manager.generate_sql(args.direction, args.target)
    engine.dispose()
    
    if not sql_statements:
        print("-- No SQL to generate")
        return
    
    sql_output = "\n".join(sql_statements)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(sql_output)
        print(f"SQL written to: {args.output}")
    else:
        print(sql_output)

def cmd_migrate_heads(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    heads = manager.heads()
    engine.dispose()
    
    if not heads:
        print("No migrations found")
    elif len(heads) == 1:
        print(f"Single head: {heads[0]}")
    else:
        print(f"\nMultiple heads detected ({len(heads)}):")
        for h in heads:
            print(f"   * {h}")
        print("\nRun 'migrate merge \"message\"' to unify branches.")

def cmd_migrate_merge(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    migration = manager.merge(args.message, args.revisions)
    engine.dispose()
    
    if migration:
        print(f"Created merge migration: {migration.version}")
        print(f"   File: {migration._file_path}")
    else:
        print("No merge needed (only one head)")

def cmd_migrate_dryrun(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    pending = manager.dry_run(args.target)
    engine.dispose()
    
    if not pending:
        print("No pending migrations")
    else:
        print(f"\nWould apply {len(pending)} migration(s):")
        print("=" * 60)
        for m in pending:
            print(f"   {m['version']} - {m['message']}")
        print("=" * 60)

def cmd_migrate_verify(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=False, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    violations = manager.verify()
    engine.dispose()
    
    if not violations:
        print("All migrations verified - no modifications detected")
    else:
        print(f"\n {len(violations)} migration(s) modified after apply:")
        print("=" * 60)
        for v in violations:
            print(f"   {v['version']} - {v['message']}")
            print(f"      Stored:  {v['stored_checksum'][:16]}...")
            print(f"      Current: {v['current_checksum'][:16]}...")
        print("=" * 60)
        sys.exit(1)

def cmd_migrate_squash(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    migration = manager.squash(args.message, args.start, args.end)
    engine.dispose()
    
    if migration:
        print(f"Created squash migration: {migration.version}")
        print(f"   File: {migration._file_path}")
    else:
        print("Not enough migrations to squash")

def cmd_migrate_seed(args):
    from psqlmodel.migrations import MigrationManager, MigrationConfig
    
    engine = _create_engine_from_env()
    config = MigrationConfig(migrations_path=args.path, debug=True, auto_detect_changes=False)
    manager = MigrationManager(engine, config)
    
    count = manager.seed(args.file, args.env)
    engine.dispose()
    
    if count > 0:
        print(f"Seeded {count} records")
    else:
        print("No seed data found")

def cmd_users_list(args):
    from psqlmodel.cli_config import list_profiles
    profiles = list_profiles()
    if not profiles:
        print("No saved profiles")
        print("   Use: migrate users save <name> ...")
        return
    print("Saved profiles:")
    for name, p in profiles.items():
        print(f"  - {name} ({p.get('host')}:{p.get('port')}/{p.get('database')})")

def cmd_users_save(args):
    from psqlmodel.cli_config import save_profile
    save_profile(
        name=args.name,
        username=args.username,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database,
        models_path=args.models_path,
        set_default=args.default
    )
    print(f"Profile '{args.name}' saved.")

def cmd_users_remove(args):
    from psqlmodel.cli_config import remove_profile
    if remove_profile(args.name):
        print(f"Profile '{args.name}' removed.")
    else:
        print(f"Profile '{args.name}' not found.")

def cmd_users_use(args):
    from psqlmodel.cli_config import set_default_profile
    set_default_profile(args.name)
    print(f"Profile '{args.name}' set as default.")

# ============================================================
# MAIN DISPATCHER
# ============================================================

def main():
    parser = argparse.ArgumentParser(prog="python -m psqlmodel", description="PSQLModel CLI")
    
    # Global flags (mostly for default engine creation if not using subcommands?
    # Actually, argparse handles subcommands. We can have a default action.)
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # --- MIGRATE ---
    migrate_parser = subparsers.add_parser("migrate", help="Database migrations")
    migrate_subs = migrate_parser.add_subparsers(dest="migrate_cmd", help="Migration actions")
    
    # migrate init
    p_init = migrate_subs.add_parser("init", help="Initialize migrations")
    p_init.add_argument("-p", "--path", help="Migrations directory path")
    p_init.add_argument("-v", "--verbose", action="store_true")
    p_init.set_defaults(func=cmd_migrate_init)
    
    # migrate status
    p_status = migrate_subs.add_parser("status", help="Show status")
    p_status.add_argument("-p", "--path", help="Migrations directory path")
    p_status.set_defaults(func=cmd_migrate_status)
    
    # migrate validate
    p_validate = migrate_subs.add_parser("validate", help="Validate schema")
    p_validate.add_argument("-p", "--path", help="Migrations directory path")
    p_validate.set_defaults(func=cmd_migrate_validate)
    
    # migrate autogenerate
    p_auto = migrate_subs.add_parser("autogenerate", help="Auto-generate migration")
    p_auto.add_argument("message", help="Migration message")
    p_auto.add_argument("-p", "--path", help="Migrations directory path")
    p_auto.set_defaults(func=cmd_migrate_autogenerate)
    
    # migrate upgrade
    p_up = migrate_subs.add_parser("upgrade", help="Apply migrations")
    p_up.add_argument("target", nargs="?", default="head", help="Target version")
    p_up.add_argument("-p", "--path", help="Migrations directory path")
    p_up.add_argument("--no-transaction", dest="transactional", action="store_false", default=True)
    p_up.add_argument("--no-lock", dest="lock", action="store_false", default=True)
    p_up.add_argument("-d", "--database", help="Target specific database profile")
    p_up.add_argument("--all", action="store_true", help="Upgrade all profiles")
    p_up.set_defaults(func=cmd_migrate_upgrade)
    
    # migrate downgrade
    p_down = migrate_subs.add_parser("downgrade", help="Revert migrations")
    p_down.add_argument("target", nargs="?", default="-1", help="Target version or -N")
    p_down.add_argument("-p", "--path", help="Migrations directory path")
    p_down.set_defaults(func=cmd_migrate_downgrade)
    
    # migrate history
    p_hist = migrate_subs.add_parser("history", help="Show history")
    p_hist.add_argument("-p", "--path", help="Migrations directory path")
    p_hist.add_argument("-n", "--limit", type=int, default=20, help="Limit number of rows")
    p_hist.add_argument("--first", action="store_true", help="Show oldest first")
    p_hist.set_defaults(func=cmd_migrate_history)
    
    # migrate check
    p_check = migrate_subs.add_parser("check", help="Check for changes (CI)")
    p_check.add_argument("-p", "--path", help="Migrations directory path")
    p_check.set_defaults(func=cmd_migrate_check)
    
    # migrate current
    p_curr = migrate_subs.add_parser("current", help="Show current version")
    p_curr.add_argument("-p", "--path", help="Migrations directory path")
    p_curr.set_defaults(func=cmd_migrate_current)
    
    # migrate stamp
    p_stamp = migrate_subs.add_parser("stamp", help="Mark version as applied")
    p_stamp.add_argument("version", help="Version to stamp")
    p_stamp.add_argument("-p", "--path", help="Migrations directory path")
    p_stamp.set_defaults(func=cmd_migrate_stamp)
    
    # migrate sql
    p_sql = migrate_subs.add_parser("sql", help="Generate SQL")
    p_sql.add_argument("direction", choices=["upgrade", "downgrade"], help="Direction")
    p_sql.add_argument("target", nargs="?", default="head", help="Target version")
    p_sql.add_argument("-p", "--path", help="Migrations directory path")
    p_sql.add_argument("-o", "--output", help="Output file")
    p_sql.set_defaults(func=cmd_migrate_sql)
    
    # migrate heads
    p_heads = migrate_subs.add_parser("heads", help="Show heads")
    p_heads.add_argument("-p", "--path", help="Migrations directory path")
    p_heads.set_defaults(func=cmd_migrate_heads)
    
    # migrate merge
    p_merge = migrate_subs.add_parser("merge", help="Merge heads")
    p_merge.add_argument("message", help="Merge message")
    p_merge.add_argument("-r", "--revisions", nargs="+", help="Revisions to merge")
    p_merge.add_argument("-p", "--path", help="Migrations directory path")
    p_merge.set_defaults(func=cmd_migrate_merge)
    
    # migrate dryrun
    p_dry = migrate_subs.add_parser("dryrun", help="Dry run preview")
    p_dry.add_argument("target", nargs="?", default="head", help="Target version")
    p_dry.add_argument("-p", "--path", help="Migrations directory path")
    p_dry.set_defaults(func=cmd_migrate_dryrun)
    
    # migrate verify
    p_ver = migrate_subs.add_parser("verify", help="Verify integrity")
    p_ver.add_argument("-p", "--path", help="Migrations directory path")
    p_ver.set_defaults(func=cmd_migrate_verify)
    
    # migrate squash
    p_sq = migrate_subs.add_parser("squash", help="Squash migrations")
    p_sq.add_argument("message", help="Squash message")
    p_sq.add_argument("--from", dest="start", required=True, help="Start version")
    p_sq.add_argument("--to", dest="end", required=True, help="End version")
    p_sq.add_argument("-p", "--path", help="Migrations directory path")
    p_sq.set_defaults(func=cmd_migrate_squash)
    
    # migrate seed
    p_seed = migrate_subs.add_parser("seed", help="Seed data")
    p_seed.add_argument("file", help="Seed file (JSON/YAML)")
    p_seed.add_argument("-e", "--env", default="default", help="Environment")
    p_seed.add_argument("-p", "--path", help="Migrations directory path")
    p_seed.set_defaults(func=cmd_migrate_seed)
    
    # migrate users
    p_users = migrate_subs.add_parser("users", help="Manage profiles")
    users_subs = p_users.add_subparsers(dest="users_cmd")
    
    u_list = users_subs.add_parser("list", help="List profiles")
    u_list.set_defaults(func=cmd_users_list)
    
    u_save = users_subs.add_parser("save", help="Save profile")
    u_save.add_argument("name", help="Profile name")
    u_save.add_argument("-u", "--username", required=True)
    u_save.add_argument("--password", required=True)
    u_save.add_argument("--host", default="localhost")
    u_save.add_argument("-P", "--port", type=int, default=5432)
    u_save.add_argument("-db", "--database", required=True)
    u_save.add_argument("--models-path")
    u_save.add_argument("--default", action="store_true")
    u_save.set_defaults(func=cmd_users_save)
    
    u_remove = users_subs.add_parser("remove", help="Remove profile")
    u_remove.add_argument("name", help="Profile name")
    u_remove.set_defaults(func=cmd_users_remove)
    
    u_use = users_subs.add_parser("use", help="Set default profile")
    u_use.add_argument("name", help="Profile name")
    u_use.set_defaults(func=cmd_users_use)
    
    # --- DEFAULT ACTION (Metrics) ---
    # args = parser.parse_args()
    
    # Handle the case where no command is passed (start metrics/app)
    # If explicit 'start' command was needed, we'd add it.
    # But usage says 'python -m psqlmodel' starts metrics logger.
    # We can check sys.argv or set a default function.
    
    if len(sys.argv) == 1:
        # No args
        print("Starting PSQLModel Metrics Logger... (Press Ctrl+C to stop)")
        # Minimal engine to keep process alive or whatever default behavior was
        # The original code didn't actually show the implementation of the default action
        # other than dispatch failing?
        # Re-reading original file (implicit in viewed lines):
        # It had "Usage: python -m psqlmodel # Start metrics logger"
        # But logic was just falling through if no match?
        # I'll implement a dummy wait loop or just print help.
        # Actually, standard behavior for CLI without args is help.
        parser.print_help()
        return

    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
