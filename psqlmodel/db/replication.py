"""
PostgreSQL Logical Replication Support
Native PUBLICATION/SUBSCRIPTION management integrated with PSQLModel ORM

Features:
- Publication management (publisher side)
- Subscription management (subscriber side)
- Full Engine integration
- Transaction support
- Sync/async compatible
"""

from typing import List, Optional, Dict, Any, Type
from ..core.transactions import Transaction, AsyncTransaction


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplicationError(Exception):
    """Base exception for replication errors"""
    pass

class PublicationError(ReplicationError):
    """Publication-specific errors"""
    pass

class SubscriptionError(ReplicationError):
    """Subscription-specific errors"""
    pass


# ============================================================
# PUBLICATION (Publisher Side)
# ============================================================

class Publication:
    """
    Manage PostgreSQL PUBLICATION objects (publisher side).
    
    Publications define what data to replicate from the source database.
    
    Example:
        from psqlmodel import create_engine
        from psqlmodel.replication import Publication
        from models import User, Order
        
        engine = create_engine(dsn="postgresql://...")
        
        # Create publication for specific tables
        pub = Publication(engine, "my_publication")
        pub.create([User, Order])
        
        # Or all tables
        pub.create(all_tables=True)
    """
    
    def __init__(self, engine, name: str):
        """
        Initialize Publication manager.
        
        Args:
            engine: PSQLModel Engine instance
            name: Publication name
        """
        if not engine:
            raise ValueError("Must provide a valid engine")
        if not name:
            raise ValueError("Publication name is required")
        
        self.engine = engine
        self.name = name
        self._engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
    
    def create(self, tables: Optional[List[Type]] = None, all_tables: bool = False,
               publish_insert: bool = True, publish_update: bool = True,
               publish_delete: bool = True, publish_truncate: bool = True,
               if_not_exists: bool = True):
        """
        Create a publication.
        
        Args:
            tables: List of Model classes to publish (mutually exclusive with all_tables)
            all_tables: Publish all tables in database
            publish_insert: Publish INSERT operations
            publish_update: Publish UPDATE operations
            publish_delete: Publish DELETE operations
            publish_truncate: Publish TRUNCATE operations
            if_not_exists: Don't error if publication already exists
            
        Example:
            pub.create([User, Order])
            pub.create(all_tables=True)
        """
        if tables and all_tables:
            raise ValueError("Cannot specify both 'tables' and 'all_tables'")
        
        if not tables and not all_tables:
            raise ValueError("Must specify either 'tables' or 'all_tables=True'")
        
        # Build SQL
        sql_parts = []
        
        # CREATE PUBLICATION
        if if_not_exists:
            # PostgreSQL doesn't have IF NOT EXISTS for publications, check manually
            check_sql = f"SELECT 1 FROM pg_publication WHERE pubname = '{self.name}'"
            if self._engine_async:
                import asyncio
                exists = asyncio.run(self._check_exists_async(check_sql))
            else:
                exists = self._check_exists_sync(check_sql)
            
            if exists:
                self.engine._debug("[REPLICATION] Publication '{}' already exists, skipping", self.name)
                return
        
        sql = f'CREATE PUBLICATION "{self.name}"'
        
        # FOR clause
        if all_tables:
            sql += ' FOR ALL TABLES'
        else:
            table_names = self._to_table_names(tables)
            tables_str = ', '.join(f'"{t}"' for t in table_names)
            sql += f' FOR TABLE {tables_str}'
        
        # WITH clause (operations to publish)
        with_parts = []
        if not publish_insert:
            with_parts.append("publish='update, delete" + (", truncate" if publish_truncate else "") + "'")
        elif not publish_update:
            with_parts.append("publish='insert, delete" + (", truncate" if publish_truncate else "") + "'")
        elif not publish_delete:
            with_parts.append("publish='insert, update" + (", truncate" if publish_truncate else "") + "'")
        elif not publish_truncate:
            with_parts.append("publish='insert, update, delete'")
        
        if with_parts:
            sql += ' WITH (' + ', '.join(with_parts) + ')'
        
        # Execute via Transaction
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Created publication: {}", self.name)
        except Exception as e:
            self.engine._debug("[REPLICATION] Error creating publication: {}", e)
            raise PublicationError(f"Failed to create publication '{self.name}': {e}") from e
    
    def drop(self, if_exists: bool = True, cascade: bool = False):
        """
        Drop a publication.
        
        Args:
            if_exists: Don't error if publication doesn't exist
            cascade: Drop dependent subscriptions
            
        Example:
            pub.drop()
            pub.drop(cascade=True)
        """
        sql = f'DROP PUBLICATION '
        if if_exists:
            sql += 'IF EXISTS '
        sql += f'"{self.name}"'
        if cascade:
            sql += ' CASCADE'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Dropped publication: {}", self.name)
        except Exception as e:
            self.engine._debug("[REPLICATION] Error dropping publication: {}", e)
            raise PublicationError(f"Failed to drop publication '{self.name}': {e}") from e
    
    def add_table(self, *tables: Type):
        """
        Add tables to an existing publication.
        
        Args:
            *tables: Model classes to add
            
        Example:
            pub.add_table(Product, Category)
        """
        if not tables:
            raise ValueError("Must provide at least one table")
        
        table_names = self._to_table_names(list(tables))
        tables_str = ', '.join(f'"{t}"' for t in table_names)
        sql = f'ALTER PUBLICATION "{self.name}" ADD TABLE {tables_str}'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Added tables to publication '{}': {}", 
                             self.name, table_names)
        except Exception as e:
            raise PublicationError(f"Failed to add tables to publication: {e}") from e
    
    def remove_table(self, *tables: Type):
        """
        Remove tables from a publication.
        
        Args:
            *tables: Model classes to remove
            
        Example:
            pub.remove_table(Product)
        """
        if not tables:
            raise ValueError("Must provide at least one table")
        
        table_names = self._to_table_names(list(tables))
        tables_str = ', '.join(f'"{t}"' for t in table_names)
        sql = f'ALTER PUBLICATION "{self.name}" DROP TABLE {tables_str}'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Removed tables from publication '{}': {}", 
                             self.name, table_names)
        except Exception as e:
            raise PublicationError(f"Failed to remove tables from publication: {e}") from e
    
    @staticmethod
    def list(engine) -> List[Dict[str, Any]]:
        """
        List all publications in the database.
        
        Args:
            engine: Engine instance
            
        Returns:
            List of publication info dicts
            
        Example:
            pubs = Publication.list(engine)
            for pub in pubs:
                # Access pub['pubname'], pub['puballtables'], etc.
                pass
        """
        sql = """
            SELECT pubname, pubowner::regrole AS owner, puballtables, pubinsert, 
                   pubupdate, pubdelete, pubtruncate
            FROM pg_publication
            ORDER BY pubname
        """
        
        engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
        
        if engine_async:
            import asyncio
            return asyncio.run(Publication._list_async(engine, sql))
        else:
            return Publication._list_sync(engine, sql)
    
    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _to_table_names(self, tables: List[Type]) -> List[str]:
        """Convert Model classes to table names."""
        table_names = []
        for table in tables:
            if hasattr(table, '__tablename__'):
                schema = getattr(table, '__schema__', 'public') or 'public'
                tablename = table.__tablename__
                # Include schema if not public
                if schema != 'public':
                    table_names.append(f'{schema}.{tablename}')
                else:
                    table_names.append(tablename)
            else:
                raise ValueError(f"Table {table} must have __tablename__ attribute")
        return table_names
    
    def _execute_sync(self, sql: str):
        """Execute SQL using Transaction."""
        with Transaction(self.engine) as tx:
            tx._execute_sql_in_tx_sync(sql)
    
    async def _execute_async(self, sql: str):
        """Execute SQL using AsyncTransaction."""
        async with AsyncTransaction(self.engine) as tx:
            await tx._execute_sql_in_tx_async(sql)
    
    def _check_exists_sync(self, sql: str) -> bool:
        """Check if publication exists (sync)."""
        with Transaction(self.engine) as tx:
            result = tx._execute_sql_in_tx_sync(sql)
            return result is not None and len(result) > 0
    
    async def _check_exists_async(self, sql: str) -> bool:
        """Check if publication exists (async)."""
        async with AsyncTransaction(self.engine) as tx:
            result = await tx._execute_sql_in_tx_async(sql)
            return result is not None and len(result) > 0
    
    @staticmethod
    def _list_sync(engine, sql: str) -> List[Dict[str, Any]]:
        """List publications (sync)."""
        with Transaction(engine) as tx:
            rows = tx._execute_sql_in_tx_sync(sql)
            return [dict(row) for row in (rows or [])]
    
    @staticmethod
    async def _list_async(engine, sql: str) -> List[Dict[str, Any]]:
        """List publications (async)."""
        async with AsyncTransaction(engine) as tx:
            rows = await tx._execute_sql_in_tx_async(sql)
            return [dict(row) for row in (rows or [])]


# ============================================================
# SUBSCRIPTION (Subscriber Side)
# ============================================================

class Subscription:
    """
    Manage PostgreSQL SUBSCRIPTION objects (subscriber side).
    
    Subscriptions connect to a publisher and replicate data.
    
    Example:
        from psqlmodel import create_engine
        from psqlmodel.replication import Subscription
        
        engine = create_engine(dsn="postgresql://subscriber_host/db")
        
        sub = Subscription(engine, "my_subscription")
        sub.create(
            connection_string="host=publisher port=5432 dbname=db user=rep password=pwd",
            publication_name="my_publication",
            copy_data=True  # Initial data copy
        )
    """
    
    def __init__(self, engine, name: str):
        """
        Initialize Subscription manager.
        
        Args:
            engine: PSQLModel Engine instance (subscriber database)
            name: Subscription name
        """
        if not engine:
            raise ValueError("Must provide a valid engine")
        if not name:
            raise ValueError("Subscription name is required")
        
        self.engine = engine
        self.name = name
        self._engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
    
    def create(self, connection_string: str, publication_name: str,
               copy_data: bool = True, create_slot: bool = True,
               enabled: bool = True, slot_name: Optional[str] = None,
               synchronous_commit: str = 'off'):
        """
        Create a subscription.
        
        Args:
            connection_string: Connection string to publisher
            publication_name: Name of publication on publisher
            copy_data: Perform initial data copy
            create_slot: Create replication slot
            enabled: Enable subscription immediately
            slot_name: Custom slot name (default: subscription name)
            synchronous_commit: Synchronous commit mode
            
        Example:
            sub.create(
                connection_string="host=pub port=5432 dbname=db user=rep password=pwd",
                publication_name="my_pub"
            )
        """
        if not connection_string:
            raise ValueError("connection_string is required")
        if not publication_name:
            raise ValueError("publication_name is required")
        
        # Build SQL
        sql = f"CREATE SUBSCRIPTION \"{self.name}\" "
        sql += f"CONNECTION '{connection_string}' "
        sql += f"PUBLICATION \"{publication_name}\""
        
        # WITH clause
        with_parts = []
        if not copy_data:
            with_parts.append("copy_data = false")
        if not create_slot:
            with_parts.append("create_slot = false")
        if not enabled:
            with_parts.append("enabled = false")
        if slot_name:
            with_parts.append(f"slot_name = '{slot_name}'")
        if synchronous_commit != 'off':
            with_parts.append(f"synchronous_commit = '{synchronous_commit}'")
        
        if with_parts:
            sql += ' WITH (' + ', '.join(with_parts) + ')'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Created subscription: {}", self.name)
        except Exception as e:
            self.engine._debug("[REPLICATION] Error creating subscription: {}", e)
            raise SubscriptionError(f"Failed to create subscription '{self.name}': {e}") from e
    
    def drop(self, if_exists: bool = True, cascade: bool = False):
        """
        Drop a subscription.
        
        Args:
            if_exists: Don't error if subscription doesn't exist
            cascade: Drop dependent objects
            
        Example:
            sub.drop()
        """
        sql = 'DROP SUBSCRIPTION '
        if if_exists:
            sql += 'IF EXISTS '
        sql += f'"{self.name}"'
        if cascade:
            sql += ' CASCADE'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Dropped subscription: {}", self.name)
        except Exception as e:
            self.engine._debug("[REPLICATION] Error dropping subscription: {}", e)
            raise SubscriptionError(f"Failed to drop subscription '{self.name}': {e}") from e
    
    def enable(self):
        """Enable a subscription."""
        sql = f'ALTER SUBSCRIPTION "{self.name}" ENABLE'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Enabled subscription: {}", self.name)
        except Exception as e:
            raise SubscriptionError(f"Failed to enable subscription: {e}") from e
    
    def disable(self):
        """Disable a subscription."""
        sql = f'ALTER SUBSCRIPTION "{self.name}" DISABLE'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Disabled subscription: {}", self.name)
        except Exception as e:
            raise SubscriptionError(f"Failed to disable subscription: {e}") from e
    
    def refresh(self, copy_data: bool = True):
        """
        Refresh publication (useful if tables were added to publication).
        
        Args:
            copy_data: Copy data for new tables
            
        Example:
            sub.refresh()  # Update to latest publication schema
        """
        sql = f'ALTER SUBSCRIPTION "{self.name}" REFRESH PUBLICATION'
        if not copy_data:
            sql += ' WITH (copy_data = false)'
        
        try:
            if self._engine_async:
                import asyncio
                asyncio.run(self._execute_async(sql))
            else:
                self._execute_sync(sql)
            
            self.engine._debug("[REPLICATION] Refreshed subscription: {}", self.name)
        except Exception as e:
            raise SubscriptionError(f"Failed to refresh subscription: {e}") from e
    
    @staticmethod
    def list(engine) -> List[Dict[str, Any]]:
        """
        List all subscriptions in the database.
        
        Args:
            engine: Engine instance
            
        Returns:
            List of subscription info dicts
            
        Example:
            subs = Subscription.list(engine)
            for sub in subs:
                # Access sub['subname'], sub['subenabled'], etc.
                pass
        """
        sql = """
            SELECT subname, subowner::regrole AS owner, subenabled, subconninfo, 
                   subslotname, subsynccommit, subpublications
            FROM pg_subscription
            ORDER BY subname
        """
        
        engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
        
        if engine_async:
            import asyncio
            return asyncio.run(Subscription._list_async(engine, sql))
        else:
            return Subscription._list_sync(engine, sql)
    
    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _execute_sync(self, sql: str):
        """Execute SQL using Transaction."""
        with Transaction(self.engine) as tx:
            tx._execute_sql_in_tx_sync(sql)
    
    async def _execute_async(self, sql: str):
        """Execute SQL using AsyncTransaction."""
        async with AsyncTransaction(self.engine) as tx:
            await tx._execute_sql_in_tx_async(sql)
    
    @staticmethod
    def _list_sync(engine, sql: str) -> List[Dict[str, Any]]:
        """List subscriptions (sync)."""
        with Transaction(engine) as tx:
            rows = tx._execute_sql_in_tx_sync(sql)
            return [dict(row) for row in (rows or [])]
    
    @staticmethod
    async def _list_async(engine, sql: str) -> List[Dict[str, Any]]:
        """List subscriptions (async)."""
        async with AsyncTransaction(engine) as tx:
            rows = await tx._execute_sql_in_tx_async(sql)
            return [dict(row) for row in (rows or [])]


# ============================================================
# REPLICATION MANAGER (High-level Helper)
# ============================================================

class ReplicationManager:
    """
    High-level helper for setting up replication.
    
    Simplifies the process of configuring publisher and subscriber servers.
    
    Example:
        from psqlmodel.replication import ReplicationManager
        
        # Setup publisher
        ReplicationManager.setup_publisher(
            pub_engine,
            pub_name="my_pub",
            tables=[User, Order]
        )
        
        # Setup subscriber
        ReplicationManager.setup_subscriber(
            sub_engine,
            sub_name="my_sub",
            publisher_dsn="host=pub port=5432 dbname=db user=rep password=pwd",
            pub_name="my_pub",
            tables=[User, Order]
        )
    """
    
    @staticmethod
    def setup_publisher(engine, pub_name: str, tables: Optional[List[Type]] = None,
                       all_tables: bool = False, validate_config: bool = True):
        """
        Complete publisher setup.
        
        Args:
            engine: Publisher engine
            pub_name: Publication name
            tables: List of Model classes to publish
            all_tables: Publish all tables
            validate_config: Validate PostgreSQL configuration first
            
        Returns:
            Publication instance
            
        Raises:
            ReplicationError: If validation fails or publication cannot be created
            
        Example:
            pub = ReplicationManager.setup_publisher(
                engine,
                pub_name="analytics_pub",
                tables=[User, Order, Payment]
            )
        """
        engine._debug("[REPLICATION] Setting up publisher: {}", pub_name)
        
        # Validate configuration
        if validate_config:
            ReplicationManager.validate_publisher_config(engine)
        
        # Create publication
        pub = Publication(engine, pub_name)
        try:
            pub.create(tables=tables, all_tables=all_tables)
            engine._debug("[REPLICATION] Publisher setup complete")
            return pub
        except Exception as e:
            engine._debug("[REPLICATION] Publisher setup failed: {}", e)
            raise ReplicationError(f"Failed to setup publisher: {e}") from e
    
    @staticmethod
    def setup_subscriber(engine, sub_name: str, publisher_dsn: str, pub_name: str,
                        tables: Optional[List[Type]] = None, create_tables: bool = True,
                        copy_data: bool = True, enabled: bool = True):
        """
        Complete subscriber setup.
        
        Args:
            engine: Subscriber engine
            sub_name: Subscription name
            publisher_dsn: Connection string to publisher
            pub_name: Publication name on publisher
            tables: List of Model classes (for auto-creating tables)
            create_tables: Auto-create tables if they don't exist
            copy_data: Perform initial data copy
            enabled: Enable subscription immediately
            
        Returns:
            Subscription instance
            
        Example:
            sub = ReplicationManager.setup_subscriber(
                engine,
                sub_name="analytics_sub",
                publisher_dsn="host=pub port=5432 dbname=db user=rep password=pwd",
                pub_name="analytics_pub",
                tables=[User, Order, Payment],
                create_tables=True
            )
        """
        engine._debug("[REPLICATION] Setting up subscriber: {}", sub_name)
        
        # Auto-create tables if requested
        if create_tables and tables:
            ReplicationManager._create_tables_if_not_exist(engine, tables)
        
        # Create subscription
        sub = Subscription(engine, sub_name)
        try:
            sub.create(
                connection_string=publisher_dsn,
                publication_name=pub_name,
                copy_data=copy_data,
                enabled=enabled
            )
            engine._debug("[REPLICATION] Subscriber setup complete")
            return sub
        except Exception as e:
            engine._debug("[REPLICATION] Subscriber setup failed: {}", e)
            raise ReplicationError(f"Failed to setup subscriber: {e}") from e
    
    @staticmethod
    def validate_publisher_config(engine):
        """
        Validate that publisher PostgreSQL is properly configured.
        
        Checks:
        - wal_level is 'logical'
        - max_replication_slots > 0
        - max_wal_senders > 0
        
        Args:
            engine: Publisher engine
            
        Raises:
            ReplicationError: If configuration is invalid
            
        Example:
            ReplicationManager.validate_publisher_config(engine)
        """
        engine._debug("[REPLICATION] Validating publisher configuration")
        
        checks = [
            ("wal_level", "logical", "wal_level must be 'logical' for replication"),
            ("max_replication_slots", 1, "max_replication_slots must be > 0"),
            ("max_wal_senders", 1, "max_wal_senders must be > 0"),
        ]
        
        errors = []
        
        for setting_name, expected_value, error_msg in checks:
            try:
                sql = f"SELECT setting FROM pg_settings WHERE name = '{setting_name}'"
                
                engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
                
                if engine_async:
                    import asyncio
                    result = asyncio.run(ReplicationManager._execute_check_async(engine, sql))
                else:
                    result = ReplicationManager._execute_check_sync(engine, sql)
                
                if not result:
                    errors.append(f"Cannot read setting '{setting_name}'")
                    continue
                
                value = result[0]['setting']
                
                # Check value
                if setting_name == "wal_level":
                    if value != expected_value:
                        errors.append(f"{error_msg} (current: {value})")
                else:
                    if int(value) < expected_value:
                        errors.append(f"{error_msg} (current: {value})")
                
            except Exception as e:
                errors.append(f"Error checking {setting_name}: {e}")
        
        if errors:
            error_text = "\n".join(f"  - {e}" for e in errors)
            engine._debug("[REPLICATION] Configuration validation failed:\n{}", error_text)
            raise ReplicationError(
                f"Publisher configuration is invalid:\n{error_text}\n\n"
                "Please update postgresql.conf and restart PostgreSQL:\n"
                "  wal_level = logical\n"
                "  max_replication_slots = 10\n"
                "  max_wal_senders = 10"
            )
        
        engine._debug("[REPLICATION] Publisher configuration validated successfully")
    
    @staticmethod
    def _create_tables_if_not_exist(engine, tables: List[Type]):
        """Auto-create tables on subscriber if they don't exist."""
        engine._debug("[REPLICATION] Auto-creating tables on subscriber")
        
        for model in tables:
            try:
                # Use the model's create_table method if available
                if hasattr(model, 'create_table'):
                    model.create_table(engine)
                    engine._debug("[REPLICATION] Created table for {}", model.__name__)
                else:
                    engine._debug("[REPLICATION] Warning: {} has no create_table method", 
                                model.__name__)
            except Exception as e:
                # Table might already exist, log but don't fail
                engine._debug("[REPLICATION] Table creation for {} failed (might exist): {}", 
                            model.__name__, e)
    
    @staticmethod
    def _execute_check_sync(engine, sql: str) -> List[Dict[str, Any]]:
        """Execute config check query (sync)."""
        with Transaction(engine) as tx:
            rows = tx._execute_sql_in_tx_sync(sql)
            return [dict(row) for row in (rows or [])]
    
    @staticmethod
    async def _execute_check_async(engine, sql: str) -> List[Dict[str, Any]]:
        """Execute config check query (async)."""
        async with AsyncTransaction(engine) as tx:
            rows = await tx._execute_sql_in_tx_async(sql)
            return [dict(row) for row in (rows or [])]
    
    # --------------------------------------------------------
    # MONITORING & STATUS (Phase 3)
    # --------------------------------------------------------
    
    @staticmethod
    def get_replication_lag(engine) -> List[Dict[str, Any]]:
        """
        Get replication lag information from publisher side.
        
        Shows lag between publisher and its subscriptions (WAL sender info).
        
        Args:
            engine: Publisher engine
            
        Returns:
            List of replication lag stats per subscription
            
        Example:
            lag_info = ReplicationManager.get_replication_lag(pub_engine)
            for info in lag_info:
                # Access info['application_name'], info['write_lag'], etc.
                pass
        """
        sql = """
            SELECT 
                application_name,
                client_addr,
                state,
                sync_state,
                write_lag,
                flush_lag,
                replay_lag,
                pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS pending_bytes
            FROM pg_stat_replication
            ORDER BY application_name
        """
        
        engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
        
        if engine_async:
            import asyncio
            return asyncio.run(ReplicationManager._execute_check_async(engine, sql))
        else:
            return ReplicationManager._execute_check_sync(engine, sql)
    
    @staticmethod
    def get_replication_slots(engine) -> List[Dict[str, Any]]:
        """
        Get replication slot information.
        
        Shows active and inactive replication slots.
        
        Args:
            engine: Publisher engine
            
        Returns:
            List of replication slot info
            
        Example:
            slots = ReplicationManager.get_replication_slots(pub_engine)
            for slot in slots:
                # Access slot['slot_name'], slot['active'], etc.
                pass
        """
        sql = """
            SELECT 
                slot_name,
                slot_type,
                database,
                active,
                restart_lsn,
                confirmed_flush_lsn,
                pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn) AS lag_bytes
            FROM pg_replication_slots
            ORDER BY slot_name
        """
        
        engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
        
        if engine_async:
            import asyncio
            return asyncio.run(ReplicationManager._execute_check_async(engine, sql))
        else:
            return ReplicationManager._execute_check_sync(engine, sql)
    
    @staticmethod
    def get_subscription_status(engine, sub_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get subscription worker status on subscriber side.
        
        Shows sync status, last error, and worker state.
        
        Args:
            engine: Subscriber engine
            sub_name: Optional subscription name filter
            
        Returns:
            List of subscription worker statuses
            
        Example:
            status = ReplicationManager.get_subscription_status(sub_engine, "my_sub")
            for s in status:
                # Access s['relname'], s['srsubstate'], etc.
                pass
        """
        sql = """
            SELECT 
                sr.srsubid,
                s.subname,
                sr.srrelid::regclass AS relname,
                sr.srsubstate,
                sr.srsublsn
            FROM pg_subscription_rel sr
            JOIN pg_subscription s ON s.oid = sr.srsubid
        """
        
        if sub_name:
            sql += f" WHERE s.subname = '{sub_name}'"
        
        sql += " ORDER BY s.subname, sr.srrelid"
        
        engine_async = bool(getattr(getattr(engine, 'config', None), 'async_', False))
        
        if engine_async:
            import asyncio
            return asyncio.run(ReplicationManager._execute_check_async(engine, sql))
        else:
            return ReplicationManager._execute_check_sync(engine, sql)
    
    @staticmethod
    def health_check_publisher(engine) -> Dict[str, Any]:
        """
        Comprehensive health check for publisher.
        
        Checks:
        - Configuration (wal_level, slots, senders)
        - Active replication slots
        - WAL sender status
        - Lag information
        
        Args:
            engine: Publisher engine
            
        Returns:
            Dict with health status and details
            
        Example:
            health = ReplicationManager.health_check_publisher(pub_engine)
            if health['healthy']:
                pass  # Publisher is healthy
            else:
                # Check health['issues'] for problems
                pass
        """
        issues = []
        warnings = []
        
        # Check configuration
        try:
            ReplicationManager.validate_publisher_config(engine)
        except ReplicationError as e:
            issues.append(f"Configuration invalid: {e}")
        
        # Check replication slots
        try:
            slots = ReplicationManager.get_replication_slots(engine)
            inactive_slots = [s for s in slots if not s.get('active')]
            if inactive_slots:
                warnings.append(f"{len(inactive_slots)} inactive replication slot(s)")
            
            # Check lag
            for slot in slots:
                lag_bytes = slot.get('lag_bytes', 0)
                if lag_bytes and lag_bytes > 10 * 1024 * 1024:  # > 10MB
                    warnings.append(f"Slot '{slot['slot_name']}' has {lag_bytes} bytes lag")
        except Exception as e:
            issues.append(f"Cannot read replication slots: {e}")
        
        # Check WAL senders
        try:
            lag_info = ReplicationManager.get_replication_lag(engine)
            for info in lag_info:
                state = info.get('state')
                if state != 'streaming':
                    warnings.append(f"Replication to '{info['application_name']}' state: {state}")
        except Exception as e:
            warnings.append(f"Cannot read replication lag: {e}")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def health_check_subscriber(engine, sub_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive health check for subscriber.
        
        Checks:
        - Subscription enabled status
        - Worker states
        - Sync completion
        
        Args:
            engine: Subscriber engine
            sub_name: Optional subscription name filter
            
        Returns:
            Dict with health status and details
            
        Example:
            health = ReplicationManager.health_check_subscriber(sub_engine, "my_sub")
            if health['healthy']:
                pass  # Subscriber is healthy
        """
        issues = []
        warnings = []
        
        # Check subscriptions
        try:
            subs = Subscription.list(engine)
            
            if sub_name:
                subs = [s for s in subs if s['subname'] == sub_name]
            
            if not subs:
                issues.append(f"No subscriptions found{' named ' + sub_name if sub_name else ''}")
            
            for sub in subs:
                if not sub.get('subenabled'):
                    warnings.append(f"Subscription '{sub['subname']}' is disabled")
            
            # Check worker status
            status = ReplicationManager.get_subscription_status(engine, sub_name)
            for s in status:
                state = s.get('srsubstate')
                if state != 'r':  # 'r' = ready/streaming
                    state_map = {'i': 'initializing', 'd': 'data copying', 's': 'synchronized', 'r': 'ready'}
                    warnings.append(
                        f"Table '{s['relname']}' in state: {state_map.get(state, state)}"
                    )
        except Exception as e:
            issues.append(f"Cannot read subscription status: {e}")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        }
