"""
Global Trigger Helpers

Provides helper functions for applying triggers to multiple tables
and creating universal audit functions.
"""

from typing import List, Type, Callable
from .triggers import Trigger, TriggerBuilder


def apply_trigger_to_all(trigger_func: Callable, *models, **trigger_config) -> List[TriggerBuilder]:
    """
    Apply the same trigger function to multiple models.
    
    Args:
        trigger_func: The trigger function to apply
        *models: Model classes to apply the trigger to
        **trigger_config: Trigger configuration (timing, events, etc.)
        
    Returns:
        List of created TriggerBuilder instances
        
    Usage:
        def universal_audit():
            # Audit logic
            pass
        
        triggers = apply_trigger_to_all(
            universal_audit,
            User, Order, Product,
            timing="AFTER",
            events=["INSERT", "UPDATE", "DELETE"],
            for_each="ROW"
        )
    """
    triggers = []
    
    for model in models:
        trg = Trigger(model)
        
        # Apply timing and events
        timing = trigger_config.get("timing", "AFTER")
        events = trigger_config.get("events", ["INSERT", "UPDATE", "DELETE"])
        
        for event in events:
            method_name = f"{timing.capitalize()}{event.capitalize()}"
            if hasattr(trg, method_name):
                getattr(trg, method_name)()
        
        # Apply ForEachRow or ForEachStatement
        for_each = trigger_config.get("for_each", "ROW")
        if for_each == "ROW":
            trg.ForEachRow()
        else:
            trg.ForEachStatement()
        
        # Set language if specified
        if "language" in trigger_config:
            trg.Language(trigger_config["language"])
        
        # Execute function
        trg.Exec(trigger_func)
        
        # Add to model
        if not hasattr(model, '__triggers__'):
            model.__triggers__ = []
        model.__triggers__.append(trg)
        
        triggers.append(trg)
    
    return triggers


def create_universal_audit_trigger(audit_table: str = "audit.global_log"):
    """
    Create a universal audit trigger function.
    
    This function can be applied to any table and will log all changes.
    
    Args:
        audit_table: Full name of the audit table (schema.table)
        
    Returns:
        A trigger function that can be used with apply_trigger_to_all
        
    Usage:
        from psqlmodel.trigger_helpers import create_universal_audit_trigger, apply_trigger_to_all
        
        audit_func = create_universal_audit_trigger("audit.changes")
        apply_trigger_to_all(audit_func, User, Order, Product)
    """
    def universal_audit():
        from .trigger_functions import (
            TG_TABLE_NAME, TG_SCHEMA_NAME, TG_OP,
            CurrentAppUserId, Now, Json, Old, New
        )
        from ..query.builder import Insert
        
        # Use TG_TABLE_NAME() to know which table fired the trigger
        table_name = TG_TABLE_NAME()
        schema_name = TG_SCHEMA_NAME()
        operation = TG_OP()
        
        # Build audit record
        audit_data = Json(
            table_name=f"{schema_name}.{table_name}",
            operation=operation,
            record_id=New.id if New and hasattr(New, 'id') else (Old.id if Old and hasattr(Old, 'id') else None),
            old_data=Old.to_json() if Old and operation in ('UPDATE', 'DELETE') else None,
            new_data=New.to_json() if New and operation in ('INSERT', 'UPDATE') else None,
            changed_by=CurrentAppUserId(),
            changed_at=Now()
        )
        
        # Insert into audit table
        return Insert(audit_table).Values(**audit_data)
    
    return universal_audit


__all__ = [
    'apply_trigger_to_all',
    'create_universal_audit_trigger',
]
