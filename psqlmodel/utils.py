# psqlmodel/utils.py
import uuid as py_uuid
from datetime import datetime, timezone

__all__ = [
    "gen_default_uuid",
    "now",
    "Interval",
]

def gen_default_uuid():
    """Generate a new UUID4 as string.
    
    Can be used as Column default:
        id: uuid = Column(primary_key=True, default=gen_default_uuid)
    """
    return str(py_uuid.uuid4())

def now():
    """
    Returns the current timestamp (server-side).
    Maps to PostgreSQL's NOW() function.
    """
    return datetime.now(timezone.utc)


def Interval(value):
    """
    Returns a PostgreSQL INTERVAL expression.
    
    Args:
        value: Interval specification as string (e.g., '7 days', '1 hour', '30 minutes')
    
    Returns:
        RawExpression representing INTERVAL 'value'
    
    Examples:
        Interval('7 days')      → INTERVAL '7 days'
        Interval('1 hour')      → INTERVAL '1 hour'
        Interval('30 minutes')  → INTERVAL '30 minutes'
        Interval('1 year')      → INTERVAL '1 year'
        
    Usage:
        expires_at=Now() + Interval('7 days')
    """
    from psqlmodel.orm.column import RawExpression
    return RawExpression(f"INTERVAL '{value}'")
