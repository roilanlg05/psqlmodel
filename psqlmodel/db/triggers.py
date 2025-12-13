"""
Trigger DSL for PSQLModel ORM

Usage:
    from psqlmodel.triggers import Trigger, Old, New, trigger

    user_trigger = (
        Trigger(User)
            .AfterUpdate(User)
            .ForEachRow()
            .When(Old.email).IsDistinctFrom(New.email)
            .Exec(verify_email)
    )

    @trigger(user_trigger)
    @table(name="users")
    class User(PSQLModel):
        id: int = Column(primary_key=True)
        email: str = Column(nullable=False)

Notas clave (Trigger 2.0):
- NO emite NOTIFY por defecto.
- Se elimina la lógica de "primary notifier" y canales automáticos.
- .Exec(...) guarda la función en:
    * function (principal)
    * exec_func / _exec_func (aliases)
    * get_exec_func() (getter)
- Mantiene compatibilidad con MiMotor vía get_trigger_sql().
"""

from typing import Callable
import hashlib
import inspect
import textwrap


# ============================================================
# OLD / NEW MAGIC OBJECTS
# ============================================================

class TriggerColumnReference:
    """Represents OLD.column or NEW.column in trigger context."""

    def __init__(self, context: str, column_name: str = None):
        """
        Args:
            context: "OLD" or "NEW"
            column_name: Column name (None if accessing via __getattr__)
        """
        self.context = context
        self.column_name = column_name

    def to_sql(self) -> str:
        """Convert to SQL: OLD.email or NEW.email"""
        if not self.column_name:
            raise ValueError(f"{self.context} requires a column name")
        return f"{self.context}.{self.column_name}"

    # Comparison operators for WHEN conditions
    def IsDistinctFrom(self, other):
        """IS DISTINCT FROM comparison."""
        return TriggerCondition(self, "IS DISTINCT FROM", other)

    def IsNotDistinctFrom(self, other):
        """IS NOT DISTINCT FROM comparison."""
        return TriggerCondition(self, "IS NOT DISTINCT FROM", other)

    def __eq__(self, other):
        """= comparison."""
        return TriggerCondition(self, "=", other)

    def __ne__(self, other):
        """!= comparison."""
        return TriggerCondition(self, "!=", other)

    def __lt__(self, other):
        """< comparison."""
        return TriggerCondition(self, "<", other)

    def __le__(self, other):
        """<= comparison."""
        return TriggerCondition(self, "<=", other)

    def __gt__(self, other):
        """> comparison."""
        return TriggerCondition(self, ">", other)

    def __ge__(self, other):
        """>= comparison."""
        return TriggerCondition(self, ">=", other)

    def IsNull(self):
        """IS NULL check."""
        return TriggerCondition(self, "IS NULL", None)

    def IsNotNull(self):
        """IS NOT NULL check."""
        return TriggerCondition(self, "IS NOT NULL", None)


class TriggerContext:
    """Magic object that provides column access: Old.email, New.email"""

    def __init__(self, context: str):
        self.context = context

    def __getattr__(self, name: str):
        """Dynamic column access: Old.email → TriggerColumnReference("OLD", "email")"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return TriggerColumnReference(self.context, name)


# Global instances
Old = TriggerContext("OLD")
New = TriggerContext("NEW")


# ============================================================
# TRIGGER CONDITIONS
# ============================================================

class TriggerCondition:
    """Represents a WHEN condition in a trigger."""

    def __init__(self, left, operator: str, right):
        self.left = left
        self.operator = operator
        self.right = right

    def to_sql(self) -> str:
        """Convert to SQL: OLD.email IS DISTINCT FROM NEW.email"""
        left_sql = self.left.to_sql() if hasattr(self.left, 'to_sql') else str(self.left)

        if self.operator in ("IS NULL", "IS NOT NULL"):
            return f"{left_sql} {self.operator}"

        if self.right is None:
            right_sql = "NULL"
        elif hasattr(self.right, 'to_sql'):
            right_sql = self.right.to_sql()
        elif isinstance(self.right, str):
            right_sql = f"'{self.right}'"
        else:
            right_sql = str(self.right)

        return f"{left_sql} {self.operator} {right_sql}"

    # Logical operators
    def And(self, other):
        """AND between conditions."""
        return CompoundTriggerCondition(self, "AND", other)

    def Or(self, other):
        """OR between conditions."""
        return CompoundTriggerCondition(self, "OR", other)


class CompoundTriggerCondition:
    """Represents compound conditions: cond1 AND cond2"""

    def __init__(self, left, operator: str, right):
        self.left = left
        self.operator = operator
        self.right = right

    def to_sql(self) -> str:
        left_sql = self.left.to_sql()
        right_sql = self.right.to_sql()
        return f"({left_sql} {self.operator} {right_sql})"

    def And(self, other):
        return CompoundTriggerCondition(self, "AND", other)

    def Or(self, other):
        return CompoundTriggerCondition(self, "OR", other)


# ============================================================
# TRIGGER BUILDER
# ============================================================

class TriggerBuilder:
    """DSL builder for database triggers."""

    def __init__(self, model):
        """
        Args:
            model: PSQLModel class this trigger is attached to
        """
        self.model = model
        self.timing = None        # BEFORE / AFTER (set by first event method)
        self.events = []          # List of events for multi-event triggers
        self.update_columns = []  # For UPDATE OF column1, column2, ...
        self.for_each = None      # ROW / STATEMENT
        self.when_condition = None
        self.language = None      # plpgsql / plpython3u (explicit override)

        # Exec function metadata (consistent)
        self.function = None
        self.exec_func = None
        self._exec_func = None

        self.trigger_name = None
        
        # Backward compatibility: maintain self.event for single-event access
        self.event = None

    # Timing + Event methods (now support chaining)
    def BeforeInsert(self, model=None):
        if self.timing is None:
            self.timing = "BEFORE"
        if "INSERT" not in self.events:
            self.events.append("INSERT")
        self.event = "INSERT"  # backward compat
        return self

    def BeforeUpdate(self, model_or_column=None):
        if self.timing is None:
            self.timing = "BEFORE"
        if "UPDATE" not in self.events:
            self.events.append("UPDATE")
        self.event = "UPDATE"  # backward compat
        # Support column-specific UPDATE
        if model_or_column is not None and hasattr(model_or_column, '__name__'):
            # It's a column, track it
            if hasattr(model_or_column, 'name'):
                self.update_columns.append(model_or_column.name)
        return self

    def BeforeDelete(self, model=None):
        if self.timing is None:
            self.timing = "BEFORE"
        if "DELETE" not in self.events:
            self.events.append("DELETE")
        self.event = "DELETE"  # backward compat
        return self

    def AfterInsert(self, model=None):
        if self.timing is None:
            self.timing = "AFTER"
        if "INSERT" not in self.events:
            self.events.append("INSERT")
        self.event = "INSERT"  # backward compat
        return self

    def AfterUpdate(self, model_or_column=None):
        if self.timing is None:
            self.timing = "AFTER"
        if "UPDATE" not in self.events:
            self.events.append("UPDATE")
        self.event = "UPDATE"  # backward compat
        # Support column-specific UPDATE
        if model_or_column is not None and hasattr(model_or_column, 'name'):
            self.update_columns.append(model_or_column.name)
        return self

    def AfterDelete(self, model=None):
        if self.timing is None:
            self.timing = "AFTER"
        if "DELETE" not in self.events:
            self.events.append("DELETE")
        self.event = "DELETE"  # backward compat
        return self
    
    def Of(self, *columns):
        """
        Specify columns for UPDATE OF trigger.
        Alias for column-specific update tracking.
        
        Usage:
            .AfterUpdate().Of(Order.status, Order.total)
        """
        for col in columns:
            if hasattr(col, 'name'):
                if col.name not in self.update_columns:
                    self.update_columns.append(col.name)
        return self

    # For each clause
    def ForEachRow(self):
        self.for_each = "ROW"
        return self

    def ForEachStatement(self):
        self.for_each = "STATEMENT"
        return self
    
    # Language specification
    def Language(self, lang: str):
        """
        Explicitly set trigger function language.
        
        Args:
            lang: "plpgsql" or "plpython3u"
        
        Usage:
            .Language("plpython3u")
        """
        self.language = lang.lower()
        return self

    # WHEN condition
    def When(self, condition):
        """
        Add WHEN condition.

        Args:
            condition: TriggerColumnReference OR TriggerCondition/CompoundTriggerCondition
        """
        if isinstance(condition, (TriggerCondition, CompoundTriggerCondition)):
            self.when_condition = condition
            return self

        condition._trigger_builder = self
        return condition

    def _set_when_condition(self, condition):
        self.when_condition = condition
        return self

    # Execute function
    def Exec(self, func: Callable):
        """
        Guarda la función Python asociada al trigger.
        Nota: Esta función sirve como metadata del DSL (naming/identidad).
        """
        self.function = func
        self.exec_func = func
        self._exec_func = func

        if not self.trigger_name:
            self.trigger_name = self._generate_trigger_name()
        return self

    def get_exec_func(self):
        return self.function

    def Name(self, name: str):
        self.trigger_name = name
        return self

    def _generate_trigger_name(self) -> str:
        table_name = getattr(self.model, '__tablename__', 'table')
        func_name = self.function.__name__ if self.function else 'trigger'
        content = f"{table_name}_{self.timing}_{self.event}_{func_name}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:6]
        return f"{table_name}_{func_name}_{hash_suffix}"

    def to_sql(self) -> tuple[str, str]:
        """
        Generate SQL for trigger function and trigger.

        Returns:
            (function_sql, trigger_sql)

        Supports:
        - Multi-event triggers: AFTER INSERT OR UPDATE OR DELETE
        - Column-specific UPDATE: UPDATE OF column1, column2
        - Language specification: .Language("plpgsql") or .Language("plpython3u")
        """
        # Validate
        if not self.timing:
            raise ValueError("Trigger must specify timing (BEFORE/AFTER)")
        if not self.events and not self.event:
            raise ValueError("Trigger must specify at least one event (INSERT/UPDATE/DELETE)")
        if not self.for_each:
            raise ValueError("Trigger must specify FOR EACH ROW or FOR EACH STATEMENT")
        if not self.function:
            raise ValueError("Trigger must specify Exec() function")

        # Get schema and table info
        schema = getattr(self.model, '__schema__', 'public') or 'public'
        table_name = getattr(self.model, '__tablename__', None)

        if not table_name:
            if self.trigger_name and '_' in self.trigger_name:
                parts = self.trigger_name.split('_')
                if len(parts) >= 2:
                    table_name = parts[0]
            if not table_name:
                raise ValueError(f"Cannot generate trigger SQL: model {self.model} has no __tablename__")

        function_name = f"{self.trigger_name}_fn"
        
        # Determine language (explicit override or auto-detect)
        if self.language:
            use_plpython = (self.language == "plpython3u")
        else:
            use_plpython = getattr(self, '_use_plpython', True)

        # Generate function SQL
        function_sql = self._generate_function_sql(
            function_name,
            schema=schema,
            table_name=table_name,
            use_plpython=use_plpython
        )

        # Add extension if using plpython
        if use_plpython:
            extension_sql = "CREATE EXTENSION IF NOT EXISTS plpython3u;"
            function_sql = f"{extension_sql}\n{function_sql}"

        # Build trigger SQL
        trigger_sql = [
            f"DROP TRIGGER IF EXISTS {self.trigger_name} ON {schema}.{table_name};",
            f"CREATE TRIGGER {self.trigger_name}",
        ]
        
        # Build event clause (multi-event support)
        events_list = self.events if self.events else [self.event]
        event_clause = f"{self.timing} {' OR '.join(events_list)}"
        
        # Add UPDATE OF columns if specified
        if self.update_columns and "UPDATE" in events_list:
            col_list = ", ".join(self.update_columns)
            # Replace UPDATE with UPDATE OF columns
            event_clause = event_clause.replace("UPDATE", f"UPDATE OF {col_list}")
        
        trigger_sql.append(f"{event_clause} ON {schema}.{table_name}")
        trigger_sql.append(f"FOR EACH {self.for_each}")
        
        # Add WHEN clause
        if self.when_condition:
            trigger_sql.append(f"WHEN ({self.when_condition.to_sql()})")
        
        trigger_sql.append(f"EXECUTE FUNCTION {function_name}();")

        return function_sql, "\n".join(trigger_sql)

    # --------------------------------------------------------
    # Function SQL generation (NO NOTIFY default)
    # --------------------------------------------------------

    def _generate_function_sql(self, function_name: str, schema: str, table_name: str, use_plpython: bool = True) -> str:
        if use_plpython:
            try:
                return self._generate_plpython_function(function_name)
            except Exception:
                pass
        return self._generate_plpgsql_function(function_name)

    def _generate_plpython_function(self, function_name: str) -> str:
        """
        Enhanced PL/Python trigger function with full DSL support.

        Supports:
        - External integrations: Notify, HttpPost, RedisPublish, KafkaProduce
        - Trigger functions: Json, Now, TG_OP, CurrentAppUserId
        - Query builders: Insert, Select
        - Old/New.to_json() serialization
        """
        func_body = []
        
        # Imports
        func_body.append("import json")
        func_body.append("import plpy")
        func_body.append("from datetime import datetime")
        func_body.append("")
        
        # Json wrapper class
        func_body.append("class Json(dict):")
        func_body.append("    def __init__(self, **kwargs):")
        func_body.append("        super().__init__(**kwargs)")
        func_body.append("        self._is_json = True")
        func_body.append("")
        
        # Now() marker
        func_body.append("def Now():")
        func_body.append("    return 'NOW()'  # Will be replaced in SQL context")
        func_body.append("")
        
        # TG_OP() helper
        func_body.append("def TG_OP():")
        func_body.append("    return TD.get('event', 'UNKNOWN').upper()")
        func_body.append("")
        
        # Trigger context functions
        func_body.append("def TG_TABLE_NAME():")
        func_body.append("    return TD.get('table_name', '')")
        func_body.append("")
        
        func_body.append("def TG_SCHEMA_NAME():")
        func_body.append("    return TD.get('table_schema', '')")
        func_body.append("")
        
        func_body.append("def TG_WHEN():")
        func_body.append("    return TD.get('when', '')")
        func_body.append("")
        
        func_body.append("def TG_LEVEL():")
        func_body.append("    return TD.get('level', '')")
        func_body.append("")
        
        func_body.append("def TG_TABLE_OID():")
        func_body.append("    return TD.get('relid', '')")
        func_body.append("")
        
        func_body.append("def TG_ARGV():")
        func_body.append("    return TD.get('args', [])")
        func_body.append("")
        
        # CurrentAppUserId() helper
        func_body.append("def CurrentAppUserId():")
        func_body.append("    try:")
        func_body.append("        result = plpy.execute(\"SELECT current_setting('app.current_user_id', true) as user_id\")")
        func_body.append("        return result[0]['user_id'] if result else None")
        func_body.append("    except:")
        func_body.append("        return None")
        func_body.append("")
        
        # Notify class
        func_body.append("class Notify:")
        func_body.append("    def __init__(self, trigger_or_channel):")
        func_body.append("        if hasattr(trigger_or_channel, 'trigger_name'):")
        func_body.append("            self.channel = trigger_or_channel.trigger_name")
        func_body.append("        else:")
        func_body.append("            self.channel = str(trigger_or_channel)")
        func_body.append("        self._payload = None")
        func_body.append("    def Message(self, payload):")
        func_body.append("        self._payload = payload")
        func_body.append("        return self")
        func_body.append("    def execute(self):")
        func_body.append("        payload = self._payload")
        func_body.append("        if payload is not None and not isinstance(payload, str):")
        func_body.append("            payload = json.dumps(payload, default=str)")
        func_body.append("        if payload is None:")
        func_body.append("            sql = f\"NOTIFY {self.channel}\"")
        func_body.append("        else:")
        func_body.append("            payload = payload.replace(\"'\", \"''\")  # Escape quotes")
        func_body.append("            sql = f\"NOTIFY {self.channel}, '{payload}'\"")
        func_body.append("        plpy.execute(sql)")
        func_body.append("")
        
        # HttpPost class
        func_body.append("class HttpPost:")
        func_body.append("    def __init__(self, url, body=None, content_type='application/json', timeout=5, headers=None):")
        func_body.append("        self.url = url")
        func_body.append("        self.body = body")
        func_body.append("        self.content_type = content_type")
        func_body.append("        self.timeout = timeout")
        func_body.append("        self.headers = headers or {}")
        func_body.append("    def execute(self):")
        func_body.append("        try:")
        func_body.append("            import requests")
        func_body.append("            body_data = self.body")
        func_body.append("            if hasattr(body_data, '_is_json') or isinstance(body_data, dict):")
        func_body.append("                body_data = json.dumps(body_data, default=str)")
        func_body.append("            headers = {'Content-Type': self.content_type, **self.headers}")
        func_body.append("            requests.post(self.url, data=body_data, headers=headers, timeout=self.timeout)")
        func_body.append("        except Exception as e:")
        func_body.append("            plpy.notice(f'HTTP POST failed: {e}')")
        func_body.append("")
        
        # RedisPublish class
        func_body.append("class RedisPublish:")
        func_body.append("    def __init__(self, channel, payload, redis_url='redis://localhost:6379/0'):")
        func_body.append("        self.channel = channel")
        func_body.append("        self.payload = payload")
        func_body.append("        self.redis_url = redis_url")
        func_body.append("    def execute(self):")
        func_body.append("        try:")
        func_body.append("            import redis")
        func_body.append("            r = redis.from_url(self.redis_url)")
        func_body.append("            payload_str = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, default=str)")
        func_body.append("            r.publish(self.channel, payload_str)")
        func_body.append("            r.close()")
        func_body.append("        except Exception as e:")
        func_body.append("            plpy.notice(f'Redis publish failed: {e}')")
        func_body.append("")
        
        # KafkaProduce class
        func_body.append("class KafkaProduce:")
        func_body.append("    def __init__(self, topic, key=None, value=None, bootstrap_servers='localhost:9092'):")
        func_body.append("        self.topic = topic")
        func_body.append("        self.key = key")
        func_body.append("        self.value = value")
        func_body.append("        self.bootstrap_servers = bootstrap_servers")
        func_body.append("    def execute(self):")
        func_body.append("        try:")
        func_body.append("            from kafka import KafkaProducer")
        func_body.append("            producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)")
        func_body.append("            key_bytes = str(self.key).encode('utf-8') if self.key else None")
        func_body.append("            value_data = self.value if isinstance(self.value, str) else json.dumps(self.value, default=str)")
        func_body.append("            value_bytes = value_data.encode('utf-8')")
        func_body.append("            producer.send(self.topic, key=key_bytes, value=value_bytes)")
        func_body.append("            producer.flush()")
        func_body.append("            producer.close()")
        func_body.append("        except Exception as e:")
        func_body.append("            plpy.notice(f'Kafka produce failed: {e}')")
        func_body.append("")
        
        # Insert class
        func_body.append("class Insert:")
        func_body.append("    def __init__(self, table):")
        func_body.append("        self.table = table")
        func_body.append("        self.values_dict = {}")
        func_body.append("    def Values(self, **kwargs):")
        func_body.append("        self.values_dict = kwargs")
        func_body.append("        return self")
        func_body.append("    def execute(self):")
        func_body.append("        columns = list(self.values_dict.keys())")
        func_body.append("        values = []")
        func_body.append("        for v in self.values_dict.values():")
        func_body.append("            if v == 'NOW()':")
        func_body.append("                values.append('NOW()')")
        func_body.append("            elif hasattr(v, '_is_json') or isinstance(v, dict):")
        func_body.append("                json_str = json.dumps(v, default=str).replace(\"'\", \"''\")")
        func_body.append("                values.append(f\"'{json_str}'::jsonb\")")
        func_body.append("            elif isinstance(v, str):")
        func_body.append("                values.append(f\"'{v.replace(\"'\", \"''\")}'\")  # Escape quotes")
        func_body.append("            else:")
        func_body.append("                values.append(str(v))")
        func_body.append("        col_list = ', '.join(columns)")
        func_body.append("        val_list = ', '.join(values)")
        func_body.append("        sql = f\"INSERT INTO {self.table} ({col_list}) VALUES ({val_list})\"")
        func_body.append("        plpy.execute(sql)")
        func_body.append("")
        
        # ORM Query Builder Execution Support
        func_body.append("# ORM Query Builder Support")
        func_body.append("def execute_query(query_obj):")
        func_body.append("    '''Execute real ORM query objects (SelectQuery, InsertQuery, etc.)'''")
        func_body.append("    if not hasattr(query_obj, 'to_sql_params'):")
        func_body.append("        return None")
        func_body.append("    try:")
        func_body.append("        sql, params = query_obj.to_sql_params()")
        func_body.append("        # Convert %s placeholders and inject params")
        func_body.append("        if params:")
        func_body.append("            for param in params:")
        func_body.append("                if hasattr(param, 'isoformat'):  # datetime")
        func_body.append("                    param_val = f\"'{param.isoformat()}'\"")
        func_body.append("                elif hasattr(param, '_is_json') or isinstance(param, dict):")
        func_body.append("                    param_str = json.dumps(param, default=str).replace(\"'\", \"''\")")
        func_body.append("                    param_val = f\"'{param_str}'::jsonb\"")
        func_body.append("                elif isinstance(param, str):")
        func_body.append("                    param_val = f\"'{param.replace(\"'\", \"''\")}'\"")
        func_body.append("                elif param is None:")
        func_body.append("                    param_val = 'NULL'")
        func_body.append("                else:")
        func_body.append("                    param_val = str(param)")
        func_body.append("                sql = sql.replace('%s', param_val, 1)")
        func_body.append("        result = plpy.execute(sql)")
        func_body.append("        return result")
        func_body.append("    except Exception as e:")
        func_body.append("        plpy.notice(f'ORM query execution failed: {e}')")
        func_body.append("        return None")
        func_body.append("")
        
        # TriggerRow class with to_json()
        func_body.append("class TriggerRow:")
        func_body.append("    def __init__(self, data):")
        func_body.append("        self._data = dict(data) if data else {}")
        func_body.append("        for k, v in self._data.items():")
        func_body.append("            setattr(self, k, v)")
        func_body.append("    def to_dict(self):")
        func_body.append("        return dict(self.__dict__, **{})")
        func_body.append("    def to_json(self):")
        func_body.append("        '''Serialize trigger row to JSON dict'''")
        func_body.append("        data = {}")
        func_body.append("        for k, v in self._data.items():")
        func_body.append("            if hasattr(v, 'isoformat'):  # datetime")
        func_body.append("                data[k] = v.isoformat()")
        func_body.append("            elif isinstance(v, (str, int, float, bool, type(None))):")
        func_body.append("                data[k] = v")
        func_body.append("            else:")
        func_body.append("                data[k] = str(v)")
        func_body.append("        return data")
        func_body.append("")
        
        # Initialize Old/New objects
        func_body.append("event = TD.get('event')")
        func_body.append("when = TD.get('when')")
        func_body.append("level = TD.get('level')")
        func_body.append("old_data = TD.get('old')")
        func_body.append("new_data = TD.get('new')")
        func_body.append("Old = TriggerRow(old_data) if old_data else None")
        func_body.append("New = TriggerRow(new_data) if new_data else None")
        func_body.append("")

        # Inject user function
        try:
            src = inspect.getsource(self.function)
            src = textwrap.dedent(src)
            func_body.append(src)
            
            # Inspect function signature to determine args
            sig = inspect.signature(self.function)
            if len(sig.parameters) == 2:
                func_body.append(f"result = {self.function.__name__}(Old, New)")
            elif len(sig.parameters) == 1:
                # If 1 arg, assume it wants New (or Old if delete? let's stick to New for now or pass context)
                # But typically triggers want both or none (using globals/context).
                # Let's assume it wants New if insert/update, Old if delete? 
                # Safer: pass New if available, else Old. 
                # Or just pass New as primary? 
                # Given the user pattern `validate_total(old, new)`, 2 args is the standard.
                # If 1 arg, maybe just pass New?
                func_body.append(f"result = {self.function.__name__}(New if New else Old)")
            else:
                func_body.append(f"result = {self.function.__name__}()")
        except Exception:
            func_body.append("result = None")
        func_body.append("")
        
        # Handle return value (execute external integrations)
        func_body.append("# Execute external integrations and ORM queries")
        func_body.append("if result is not None:")
        func_body.append("    # Check if it's an ORM query object (has to_sql_params)")
        func_body.append("    if hasattr(result, 'to_sql_params'):")
        func_body.append("        execute_query(result)")
        func_body.append("    # Check if it's an integration object")
        func_body.append("    elif isinstance(result, (Notify, HttpPost, RedisPublish, KafkaProduce, Insert)):")
        func_body.append("        result.execute()")
        func_body.append("    # Check if it's a list of objects")
        func_body.append("    elif isinstance(result, list):")
        func_body.append("        for item in result:")
        func_body.append("            if hasattr(item, 'to_sql_params'):  # ORM query")
        func_body.append("                execute_query(item)")
        func_body.append("            elif hasattr(item, 'execute'):  # Integration")
        func_body.append("                item.execute()")
        func_body.append("")
        
        # Reflect mutations on New/Old
        func_body.append("# Reflect mutations made to Old/New objects")
        func_body.append("if New is not None:")
        func_body.append("    tmp = dict(New.__dict__)")
        func_body.append("    tmp.pop('_data', None)")
        func_body.append("    new_data = tmp")
        func_body.append("if Old is not None:")
        func_body.append("    tmp_old = dict(Old.__dict__)")
        func_body.append("    tmp_old.pop('_data', None)")
        func_body.append("    old_data = tmp_old")
        func_body.append("")
        
        # Return appropriate value for plpython
        func_body.append("# Return appropriate value for plpython trigger")
        func_body.append("if level != 'ROW':")
        func_body.append("    return None")
        func_body.append("if event == 'DELETE':")
        func_body.append("    if old_data is not None:")
        func_body.append("        TD['old'] = old_data")
        func_body.append("    return 'OK'")
        func_body.append("else:")
        func_body.append("    if new_data is not None:")
        func_body.append("        TD['new'] = new_data")
        func_body.append("    return 'MODIFY'")

        python_code = '\n'.join(func_body)

        return f"""CREATE OR REPLACE FUNCTION {function_name}()
RETURNS trigger AS $$
{python_code}
$$ LANGUAGE plpython3u;"""

    def _generate_plpgsql_function(self, function_name: str) -> str:
        """
        PL/pgSQL trigger function SIN NOTIFY por defecto.
        """
        return f"""CREATE OR REPLACE FUNCTION {function_name}()
RETURNS trigger AS $$
BEGIN
    IF TG_LEVEL = 'STATEMENT' THEN
        RETURN NULL;while
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;"""


def Trigger(model):
    """
    Create a new trigger builder for a model.
    """
    return TriggerBuilder(model)


# ============================================================
# TRIGGER CONDITION CHAINING FIX
# ============================================================

def _patch_condition_return(original_method):
    """
    Wrap condition methods to return builder after setting condition.
    """
    def wrapper(self, *args, **kwargs):
        condition = original_method(self, *args, **kwargs)
        if hasattr(self, '_trigger_builder'):
            builder = self._trigger_builder
            builder._set_when_condition(condition)
            return builder
        return condition
    return wrapper


for method_name in ['IsDistinctFrom', 'IsNotDistinctFrom', 'IsNull', 'IsNotNull']:
    original = getattr(TriggerColumnReference, method_name)
    setattr(TriggerColumnReference, method_name, _patch_condition_return(original))


# ============================================================
# @trigger DECORATOR
# ============================================================

def trigger(*triggers: TriggerBuilder):
    """
    Decorator to attach triggers to a model.

    Mantiene:
    - __triggers__
    - get_trigger_sql()

    Eliminado:
    - cualquier lógica de "primary notifier" o canales automáticos.
    """
    def decorator(cls):
        if not hasattr(cls, '__triggers__'):
            cls.__triggers__ = []

        # Update trigger model references to actual class and schema
        for trg in triggers:
            trg.model = cls
            trg.schema = getattr(cls, '__schema__', 'public') or 'public'
            if trg.function:
                trg.trigger_name = trg._generate_trigger_name()

        cls.__triggers__.extend(triggers)

        if not hasattr(cls, 'get_trigger_sql'):
            @classmethod
            def get_trigger_sql(model_cls):
                sql_parts = []
                for trg in getattr(model_cls, '__triggers__', []):
                    func_sql, trg_sql = trg.to_sql()
                    sql_parts.append(func_sql)
                    sql_parts.append(trg_sql)
                return '\n\n'.join(sql_parts)

            cls.get_trigger_sql = get_trigger_sql

        return cls

    return decorator
