"""
Trigger Functions & Helpers for PSQLModel Triggers

Provides helper functions usable inside trigger exec functions:
- Json() - JSON construction
- Now() - current timestamp
- TG_OP() - trigger operation
- CurrentAppUserId() - current_setting wrapper
- Query builders for use in triggers

Usage:
    from psqlmodel.trigger_functions import Json, Now, TG_OP, Insert, Select
    
    def on_user_created():
        payload = Json(
            event="user_created",
            user_id=New.id,
            created_at=Now()
        )
        Notify("events").Message(payload)
"""

from typing import Any, Dict, Optional
from datetime import datetime


# ============================================================
# JSON WRAPPER
# ============================================================

class Json(dict):
    """
    Wrapper for JSON construction in triggers.
    Signals that this dict should be serialized to JSON.
    
    Usage:
        payload = Json(
            event="user_created",
            user_id=New.id,
            email=New.email
        )
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_json = True


# ============================================================
# TIMESTAMP MARKER
# ============================================================

class NowMarker:
    """
    Marker for current timestamp in triggers.
    Renders as NOW() or CURRENT_TIMESTAMP in SQL.
    
    Usage:
        created_at = Now()
    """
    
    def to_sql(self) -> str:
        return "NOW()"
    
    def __repr__(self) -> str:
        return "Now()"


def Now() -> NowMarker:
    """Get current timestamp marker for use in triggers."""
    return NowMarker()


# ============================================================
# TRIGGER OPERATION
# ============================================================

class TG_OP_Marker:
    """
    Marker for TG_OP variable in triggers.
    Returns 'INSERT', 'UPDATE', or 'DELETE'.
    
    Usage:
        if TG_OP() == 'INSERT':
            # INSERT-specific logic
        
        # Or with In():
        if TG_OP().In('INSERT', 'UPDATE'):
            # ...
    """
    
    def to_sql(self) -> str:
        return "TG_OP"
    
    def In(self, *values: str) -> 'TG_OP_Condition':
        """Check if TG_OP is in given values."""
        return TG_OP_Condition(values)
    
    def __eq__(self, value: str) -> str:
        """Compare TG_OP with value."""
        return f"TG_OP = '{value}'"
    
    def __repr__(self) -> str:
        return "TG_OP()"


class TG_OP_Condition:
    """Condition for TG_OP.In(...) checks."""
    
    def __init__(self, values: tuple):
        self.values = values
    
    def to_sql(self) -> str:
        quoted = [f"'{v}'" for v in self.values]
        return f"TG_OP IN ({', '.join(quoted)})"


def TG_OP() -> TG_OP_Marker:
    """Get trigger operation marker."""
    return TG_OP_Marker()


# ============================================================
# CURRENT APP USER ID
# ============================================================

class CurrentAppUserIdMarker:
    """
    Marker for current_setting('app.current_user_id').
    Useful for audit trails.
    
    Usage:
        done_by = CurrentAppUserId()
    """
    
    def to_sql(self) -> str:
        return "current_setting('app.current_user_id', true)"
    
    def __repr__(self) -> str:
        return "CurrentAppUserId()"


def CurrentAppUserId() -> CurrentAppUserIdMarker:
    """Get current app user ID from session settings."""
    return CurrentAppUserIdMarker()


# ============================================================
# TRIGGER CONTEXT FUNCTIONS
# ============================================================

class TG_TABLE_NAME_Marker:
    """
    Marker for TG_TABLE_NAME variable in triggers.
    Returns the name of the table that fired the trigger.
    
    Usage:
        table_name = TG_TABLE_NAME()
    """
    
    def to_sql(self) -> str:
        return "TG_TABLE_NAME"
    
    def __repr__(self) -> str:
        return "TG_TABLE_NAME()"


def TG_TABLE_NAME() -> TG_TABLE_NAME_Marker:
    """Get the name of the table that fired the trigger."""
    return TG_TABLE_NAME_Marker()


class TG_SCHEMA_NAME_Marker:
    """
    Marker for TG_TABLE_SCHEMA variable in triggers.
    Returns the schema of the table that fired the trigger.
    
    Usage:
        schema_name = TG_SCHEMA_NAME()
    """
    
    def to_sql(self) -> str:
        return "TG_TABLE_SCHEMA"
    
    def __repr__(self) -> str:
        return "TG_SCHEMA_NAME()"


def TG_SCHEMA_NAME() -> TG_SCHEMA_NAME_Marker:
    """Get the schema of the table that fired the trigger."""
    return TG_SCHEMA_NAME_Marker()


class TG_WHEN_Marker:
    """
    Marker for TG_WHEN variable in triggers.
    Returns 'BEFORE' or 'AFTER'.
    
    Usage:
        when = TG_WHEN()
    """
    
    def to_sql(self) -> str:
        return "TG_WHEN"
    
    def __repr__(self) -> str:
        return "TG_WHEN()"


def TG_WHEN() -> TG_WHEN_Marker:
    """Get trigger timing: 'BEFORE' or 'AFTER'."""
    return TG_WHEN_Marker()


class TG_LEVEL_Marker:
    """
    Marker for TG_LEVEL variable in triggers.
    Returns 'ROW' or 'STATEMENT'.
    
    Usage:
        level = TG_LEVEL()
    """
    
    def to_sql(self) -> str:
        return "TG_LEVEL"
    
    def __repr__(self) -> str:
        return "TG_LEVEL()"


def TG_LEVEL() -> TG_LEVEL_Marker:
    """Get trigger level: 'ROW' or 'STATEMENT'."""
    return TG_LEVEL_Marker()


class TG_TABLE_OID_Marker:
    """
    Marker for TG_RELID variable in triggers.
    Returns the OID of the table that fired the trigger.
    
    Usage:
        table_oid = TG_TABLE_OID()
    """
    
    def to_sql(self) -> str:
        return "TG_RELID"
    
    def __repr__(self) -> str:
        return "TG_TABLE_OID()"


def TG_TABLE_OID() -> TG_TABLE_OID_Marker:
    """Get the OID of the table that fired the trigger."""
    return TG_TABLE_OID_Marker()


class TG_ARGV_Marker:
    """
    Marker for TG_ARGV variable in triggers.
    Returns list of trigger arguments.
    
    Usage:
        args = TG_ARGV()
        first_arg = TG_ARGV()[0]
    """
    
    def __getitem__(self, index: int):
        return f"TG_ARGV[{index}]"
    
    def to_sql(self) -> str:
        return "TG_ARGV"
    
    def __repr__(self) -> str:
        return "TG_ARGV()"


def TG_ARGV() -> TG_ARGV_Marker:
    """Get trigger arguments list."""
    return TG_ARGV_Marker()


# ============================================================
# QUERY BUILDER SUPPORT IN TRIGGERS
# ============================================================

class TriggerQuery:
    """Base class for query builders usable in triggers."""
    
    def to_plpython(self) -> str:
        """Convert to plpy.execute() call."""
        raise NotImplementedError


class TriggerSelect(TriggerQuery):
    """
    SELECT query builder for use in triggers.
    
    Usage:
        zone_id = Select(GeoZone.id).Where(
            GeoZone.area.Contains(New.location)
        ).Scalar()
    """
    
    def __init__(self, *columns):
        self.columns = columns
        self.where_clauses = []
        self.limit_value = None
    
    def Where(self, condition):
        """Add WHERE clause."""
        self.where_clauses.append(condition)
        return self
    
    def Limit(self, n: int):
        """Add LIMIT clause."""
        self.limit_value = n
        return self
    
    def Scalar(self) -> 'ScalarQueryMarker':
        """Mark this query to return scalar value."""
        return ScalarQueryMarker(self)
    
    def to_sql(self) -> str:
        """Generate SQL string."""
        # This is a simplified version - actual implementation
        # would need full column and condition resolution
        parts = ["SELECT"]
        if self.columns:
            col_parts = []
            for col in self.columns:
                if hasattr(col, 'to_sql'):
                    col_parts.append(col.to_sql())
                else:
                    col_parts.append(str(col))
            parts.append(", ".join(col_parts))
        
        # WHERE, LIMIT, etc. would be added here
        if self.where_clauses:
            parts.append("WHERE ...")  # Simplified
        
        if self.limit_value:
            parts.append(f"LIMIT {self.limit_value}")
        
        return " ".join(parts)
    
    def to_plpython(self) -> str:
        """Convert to plpy.execute() call."""
        sql = self.to_sql()
        return f"plpy.execute(\"{sql}\")"


class ScalarQueryMarker:
    """Marker for scalar query result."""
    
    def __init__(self, query: TriggerSelect):
        self.query = query
    
    def to_sql(self) -> str:
        return f"({self.query.to_sql()})"


def Select(*columns) -> TriggerSelect:
    """Create a SELECT query for use in triggers."""
    return TriggerSelect(*columns)


class TriggerInsert(TriggerQuery):
    """
    INSERT query builder for use in triggers.
    
    Usage:
        Insert("integration.outbox").Values(
            aggregate="user",
            aggregate_id=New.id,
            event_type="user_updated",
            payload=Json(id=New.id),
            created_at=Now()
        )
    """
    
    def __init__(self, table: str):
        self.table = table
        self.values_dict = {}
    
    def Values(self, **kwargs) -> 'TriggerInsert':
        """Set values to insert."""
        self.values_dict = kwargs
        return self
    
    def to_sql(self) -> str:
        """Generate SQL string."""
        columns = list(self.values_dict.keys())
        values = []
        
        for v in self.values_dict.values():
            if isinstance(v, NowMarker):
                values.append("NOW()")
            elif isinstance(v, Json):
                # Simplified - actual would need JSON serialization
                values.append(f"'{v}'::jsonb")
            elif hasattr(v, 'to_sql'):
                values.append(v.to_sql())
            elif isinstance(v, str):
                values.append(f"'{v}'")
            else:
                values.append(str(v))
        
        col_list = ", ".join(columns)
        val_list = ", ".join(values)
        
        return f"INSERT INTO {self.table} ({col_list}) VALUES ({val_list})"
    
    def to_plpython(self) -> str:
        """Convert to plpy.execute() call."""
        sql = self.to_sql()
        return f"plpy.execute(\"{sql}\")"


def Insert(table: str) -> TriggerInsert:
    """Create an INSERT query for use in triggers."""
    return TriggerInsert(table)


# ============================================================
# ROW SERIALIZATION HELPERS
# ============================================================

class RowSerializationMixin:
    """
    Mixin to add to_json() method to Old/New trigger objects.
    This will be injected into TriggerRow class during template generation.
    """
    
    @staticmethod
    def to_json_code() -> str:
        """
        Return Python code to add to TriggerRow class.
        This code will be injected into the plpython function template.
        """
        return '''
    def to_json(self):
        """Serialize trigger row to JSON."""
        import json
        data = {}
        for k, v in self._data.items():
            if hasattr(v, 'isoformat'):  # datetime
                data[k] = v.isoformat()
            elif isinstance(v, (str, int, float, bool, type(None))):
                data[k] = v
            else:
                data[k] = str(v)
        return data
'''


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'Json',
    'Now',
    'TG_OP',
    'CurrentAppUserId',
    'TG_TABLE_NAME',
    'TG_SCHEMA_NAME',
    'TG_WHEN',
    'TG_LEVEL',
    'TG_TABLE_OID',
    'TG_ARGV',
    'Select',
    'Insert',
    'RowSerializationMixin',
]
