# psqlmodel/types.py
"""
SQL Types for PostgreSQL ORM.

Each type maps to:
- PostgreSQL DDL type
- Python type hint (for type checking)
- Compatible with Column for model definitions

Usage:
    from psqlmodel.types import varchar, integer, jsonb, timestamp

    @table(name="users")
    class User(PSQLModel):
        id: serial = Column(primary_key=True)
        name: varchar = Column(nullable=False, max_len=50)  # VARCHAR(50) NOT NULL
        code: char = Column(max_len=10)  # CHAR(10)
        bits: bit = Column(max_len=8)  # BIT(8)
        metadata: jsonb = Column()
"""

from typing import Any, List as PyList


class SQLType:
    """Base class for SQL types. Concrete SQL types should inherit this.

    Note: user-facing type names are provided as lowercase classes/constructors
    (e.g. `uuid`, `timestamp`, `varchar`).
    """
    def ddl(self) -> str:
        raise NotImplementedError()


# ============================================================
# NUMERIC TYPES
# ============================================================

class smallint(SQLType):
    """Small integer: -32,768 to 32,767. Python: int"""
    def ddl(self):
        return "SMALLINT"


class integer(SQLType):
    """Standard integer: -2,147,483,648 to 2,147,483,647. Python: int"""
    def ddl(self):
        return "INTEGER"


class bigint(SQLType):
    """Large integer: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807. Python: int"""
    def ddl(self):
        return "BIGINT"


class serial(SQLType):
    """Auto-incrementing integer (1 to 2,147,483,647). Python: int"""
    def ddl(self):
        return "SERIAL"


class bigserial(SQLType):
    """Auto-incrementing big integer. Python: int"""
    def ddl(self):
        return "BIGSERIAL"


class smallserial(SQLType):
    """Auto-incrementing small integer (1 to 32,767). Python: int"""
    def ddl(self):
        return "SMALLSERIAL"


class numeric(SQLType):
    """Exact numeric with selectable precision. Python: Decimal
    
    Args:
        precision: Total number of digits (default 10)
        scale: Digits after decimal point (default 2)
    
    Example:
        price: numeric = Column()  # NUMERIC(10,2)
        price: numeric(15, 4) = Column()  # NUMERIC(15,4)
    """
    def __init__(self, precision: int = 10, scale: int = 2):
        self.precision = precision
        self.scale = scale

    def ddl(self):
        return f"NUMERIC({self.precision},{self.scale})"


# Alias for numeric
decimal = numeric


class real(SQLType):
    """Single precision floating-point (4 bytes). Python: float"""
    def ddl(self):
        return "REAL"


class double(SQLType):
    """Double precision floating-point (8 bytes). Python: float"""
    def ddl(self):
        return "DOUBLE PRECISION"


class money(SQLType):
    """Currency amount with locale formatting. Python: Decimal"""
    def ddl(self):
        return "MONEY"


# ============================================================
# TEXT/STRING TYPES
# ============================================================

class varchar(SQLType):
    """Variable-length string. Python: str
    Length is set via Column(max_len=...).
    """
    def ddl(self):
        return "VARCHAR"


class char(SQLType):
    """Fixed-length string, blank-padded. Python: str
    Length is set via Column(max_len=...).
    """
    def ddl(self):
        return "CHAR"


class text(SQLType):
    """Unlimited variable-length string. Python: str"""
    def ddl(self):
        return "TEXT"


# ============================================================
# BINARY TYPES
# ============================================================

class bytea(SQLType):
    """Binary data (bytes). Python: bytes"""
    def ddl(self):
        return "BYTEA"


# ============================================================
# DATE/TIME TYPES
# ============================================================

class date(SQLType):
    """Calendar date (year, month, day). Python: datetime.date"""
    def ddl(self):
        return "DATE"


class time(SQLType):
    """Time of day without time zone. Python: datetime.time
    
    Args:
        with_timezone: If True, includes time zone (default False)
    """
    def __init__(self, with_timezone: bool = False):
        self.with_timezone = with_timezone

    def ddl(self):
        if self.with_timezone:
            return "TIME WITH TIME ZONE"
        return "TIME"


class timestamp(SQLType):
    """Date and time. Python: datetime.datetime
    
    Note: Use Column(timez=True) for WITH TIME ZONE
    """
    def __init__(self):
        pass

    def ddl(self):
        return "TIMESTAMP"


class timestamptz(SQLType):
    """Date and time with time zone. Python: datetime.datetime"""
    def ddl(self):
        return "TIMESTAMP WITH TIME ZONE"


class interval(SQLType):
    """Time span/duration. Python: datetime.timedelta"""
    def ddl(self):
        return "INTERVAL"


# ============================================================
# BOOLEAN TYPE
# ============================================================

class boolean(SQLType):
    """True/False value. Python: bool"""
    def ddl(self):
        return "BOOLEAN"


# ============================================================
# UUID TYPE
# ============================================================

class uuid(SQLType):
    """Universally Unique Identifier. Python: uuid.UUID or str
    
    Note: Column automatically uses DEFAULT gen_random_uuid() in DDL
    """
    def ddl(self):
        return "UUID"


# ============================================================
# JSON TYPES
# ============================================================

class json(SQLType):
    """JSON data stored as text. Python: dict/list
    
    Note: Preserves whitespace, key order. No indexing.
    """
    def ddl(self):
        return "JSON"


class jsonb(SQLType):
    """Binary JSON data. Python: dict/list
    
    Note: Faster queries, supports indexing. Recommended over json.
    """
    def ddl(self):
        return "JSONB"


# ============================================================
# ARRAY TYPE
# ============================================================

class array(SQLType):
    """Array of another type. Python: list
    
    Args:
        element_type: The SQLType for array elements
    
    Example:
        tags: array(varchar(50)) = Column()  # VARCHAR(50)[]
        scores: array(integer) = Column()  # INTEGER[]
    """
    def __init__(self, element_type: SQLType):
        if isinstance(element_type, type):
            # If passed as class, instantiate it
            self.element_type = element_type()
        else:
            self.element_type = element_type

    def ddl(self):
        return f"{self.element_type.ddl()}[]"


# ============================================================
# NETWORK TYPES
# ============================================================

class inet(SQLType):
    """IPv4 or IPv6 host address. Python: ipaddress.ip_address"""
    def ddl(self):
        return "INET"


class cidr(SQLType):
    """IPv4 or IPv6 network. Python: ipaddress.ip_network"""
    def ddl(self):
        return "CIDR"


class macaddr(SQLType):
    """MAC address. Python: str"""
    def ddl(self):
        return "MACADDR"


class macaddr8(SQLType):
    """MAC address (EUI-64 format). Python: str"""
    def ddl(self):
        return "MACADDR8"


# ============================================================
# GEOMETRIC TYPES
# ============================================================

class point(SQLType):
    """Geometric point (x, y). Python: tuple[float, float]"""
    def ddl(self):
        return "POINT"


class line(SQLType):
    """Infinite line. Python: tuple"""
    def ddl(self):
        return "LINE"


class lseg(SQLType):
    """Line segment. Python: tuple"""
    def ddl(self):
        return "LSEG"


class box(SQLType):
    """Rectangular box. Python: tuple"""
    def ddl(self):
        return "BOX"


class path(SQLType):
    """Geometric path. Python: list[tuple]"""
    def ddl(self):
        return "PATH"


class polygon(SQLType):
    """Closed geometric path. Python: list[tuple]"""
    def ddl(self):
        return "POLYGON"


class circle(SQLType):
    """Circle (center + radius). Python: tuple"""
    def ddl(self):
        return "CIRCLE"


# ============================================================
# RANGE TYPES
# ============================================================

class int4range(SQLType):
    """Range of integer. Python: range or tuple"""
    def ddl(self):
        return "INT4RANGE"


class int8range(SQLType):
    """Range of bigint. Python: range or tuple"""
    def ddl(self):
        return "INT8RANGE"


class numrange(SQLType):
    """Range of numeric. Python: tuple"""
    def ddl(self):
        return "NUMRANGE"


class tsrange(SQLType):
    """Range of timestamp without time zone. Python: tuple"""
    def ddl(self):
        return "TSRANGE"


class tstzrange(SQLType):
    """Range of timestamp with time zone. Python: tuple"""
    def ddl(self):
        return "TSTZRANGE"


class daterange(SQLType):
    """Range of date. Python: tuple"""
    def ddl(self):
        return "DATERANGE"


# ============================================================
# SPECIAL TYPES
# ============================================================

class xml(SQLType):
    """XML data. Python: str"""
    def ddl(self):
        return "XML"


class tsvector(SQLType):
    """Text search document. Python: str"""
    def ddl(self):
        return "TSVECTOR"


class tsquery(SQLType):
    """Text search query. Python: str"""
    def ddl(self):
        return "TSQUERY"


class bit(SQLType):
    """Fixed-length bit string. Python: str
    Length is set via Column(max_len=...).
    """
    def ddl(self):
        return "BIT"


class varbit(SQLType):
    """Variable-length bit string. Python: str
    Length is set via Column(max_len=...).
    """
    def ddl(self):
        return "BIT VARYING"


# ============================================================
# RELATIONSHIP TYPE (for type hints)
# ============================================================

# Import Relation for type hints
from .relationships import Relation

__all__ = ['Relation']  # Export for: from psqlmodel.types import Relation
