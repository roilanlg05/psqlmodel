# psqlmodel/table.py
"""
Table decorator with full OOP support for constraints, indexes, and table options.

Usage:
    from psqlmodel.table import table, UniqueConstraint, CheckConstraint, Index, ForeignKeyConstraint
    
    @table(
        name="users",
        schema="auth",
        constraints=[
            UniqueConstraint(User.email),
            UniqueConstraint(User.username, name="uq_username"),
            CheckConstraint("age >= 18", name="ck_adult"),
            ForeignKeyConstraint([User.role_id], "roles.id", on_delete="CASCADE"),
        ],
        indexes=[
            Index(User.email),
            Index(User.name, User.country, unique=True),
        ],
    )
    class User(PSQLModel):
        ...
"""
from __future__ import annotations

from typing import get_type_hints, List, Optional, Any, Union, Sequence
import warnings

from .column import Column
from .model import PSQLModel


# ============================================================
# HELPER: column name resolver
# ============================================================

def _resolve_column_names(
    columns: Sequence[Union[Column, str, Any]],
    model_cls: Optional[type] = None,  # reservado para futuras extensiones
) -> List[str]:
    """
    Resolve column-like objects to their string names.

    Actual behavior:
      - str       -> tal cual
      - Column    -> column.name
      - anything  -> str(obj)

    model_cls se acepta pero no se usa aún para mantener compatibilidad
    y permitir futura resolución basada en el modelo.
    """
    col_names: List[str] = []
    for col in columns:
        if isinstance(col, str):
            col_names.append(col)
        elif isinstance(col, Column):
            col_names.append(col.name)
        else:
            col_names.append(str(col))
    return col_names


# ============================================================
# CONSTRAINT CLASSES (OOP)
# ============================================================

class Constraint:
    """Base class for all table constraints."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        """Generate DDL for this constraint."""
        raise NotImplementedError()


class UniqueConstraint(Constraint):
    """UNIQUE constraint on one or more columns.
    
    Example:
        UniqueConstraint(User.email)
        UniqueConstraint("email", "username", name="uq_email_username")
        UniqueConstraint(User.first_name, User.last_name, name="uq_full_name")
    """
    
    def __init__(self, *columns: Union[Column, str], name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        col_names = _resolve_column_names(self.columns, model_cls)
        constraint_name = self.name or f"uq_{table_name}_{'_'.join(col_names)}"
        cols_sql = ", ".join(col_names)
        return f"CONSTRAINT {constraint_name} UNIQUE ({cols_sql})"


class PrimaryKeyConstraint(Constraint):
    """Composite PRIMARY KEY constraint.
    
    Example:
        PrimaryKeyConstraint(OrderItem.order_id, OrderItem.product_id)
        PrimaryKeyConstraint("order_id", "product_id")
    """
    
    def __init__(self, *columns: Union[Column, str], name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        col_names = _resolve_column_names(self.columns, model_cls)
        constraint_name = self.name or f"pk_{table_name}"
        cols_sql = ", ".join(col_names)
        return f"CONSTRAINT {constraint_name} PRIMARY KEY ({cols_sql})"


class ForeignKeyConstraint(Constraint):
    """FOREIGN KEY constraint.
    
    Args:
        columns: List of local columns
        references: Target table.column(s) as string or tuple
        on_delete: CASCADE, SET NULL, SET DEFAULT, RESTRICT, NO ACTION
        on_update: CASCADE, SET NULL, SET DEFAULT, RESTRICT, NO ACTION
    
    Example:
        ForeignKeyConstraint([User.role_id], "roles.id", on_delete="CASCADE")
        ForeignKeyConstraint([OrderItem.order_id], "orders.id", on_delete="CASCADE", on_update="CASCADE")
    """
    
    def __init__(
        self,
        columns: Sequence[Union[Column, str]],
        references: str,
        *,
        on_delete: Optional[str] = None,
        on_update: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.columns = list(columns)
        self.references = references
        self.on_delete = on_delete
        self.on_update = on_update
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        col_names = _resolve_column_names(self.columns, model_cls)
        constraint_name = self.name or f"fk_{table_name}_{'_'.join(col_names)}"
        cols_sql = ", ".join(col_names)
        
        # Parse references: "table.column" or "schema.table.column"
        ref_parts = self.references.split(".")
        if len(ref_parts) == 2:
            ref_table, ref_col = ref_parts
        else:
            ref_table = ".".join(ref_parts[:-1])
            ref_col = ref_parts[-1]
        
        sql = f"CONSTRAINT {constraint_name} FOREIGN KEY ({cols_sql}) REFERENCES {ref_table}({ref_col})"
        
        if self.on_delete:
            sql += f" ON DELETE {self.on_delete.upper()}"
        if self.on_update:
            sql += f" ON UPDATE {self.on_update.upper()}"
        
        return sql


class CheckConstraint(Constraint):
    """CHECK constraint with SQL expression.
    
    Example:
        CheckConstraint("age >= 18", name="ck_adult")
        CheckConstraint("price > 0 AND price < 1000000")
    """
    
    def __init__(self, expression: str, *, name: Optional[str] = None):
        super().__init__(name)
        self.expression = expression
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        constraint_name = self.name or f"ck_{table_name}"
        return f"CONSTRAINT {constraint_name} CHECK ({self.expression})"


class ExcludeConstraint(Constraint):
    """EXCLUDE constraint (for range types, etc.).
    
    Example:
        ExcludeConstraint("USING gist (room WITH =, during WITH &&)", name="no_overlap")
    """
    
    def __init__(self, expression: str, *, name: Optional[str] = None):
        super().__init__(name)
        self.expression = expression
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        constraint_name = self.name or f"excl_{table_name}"
        return f"CONSTRAINT {constraint_name} EXCLUDE {self.expression}"


# ============================================================
# INDEX CLASS
# ============================================================

class Index:
    """Database index on one or more columns.
    
    Args:
        *columns: Columns to index
        name: Optional index name
        unique: Create unique index
        method: Index method (btree, hash, gist, gin, etc.)
        where: Partial index condition
        include: Columns to include (covering index)
    
    Example:
        Index(User.email)
        Index(User.name, unique=True)
        Index(User.status, where="status = 'active'", name="idx_active_users")
        Index(User.metadata, method="gin")  # For JSONB
    """
    
    def __init__(
        self,
        *columns: Union[Column, str],
        name: Optional[str] = None,
        unique: bool = False,
        method: Optional[str] = None,
        where: Optional[str] = None,
        include: Optional[Sequence[Union[Column, str]]] = None,
    ):
        self.columns = columns
        self.name = name
        self.unique = unique
        self.method = method
        self.where = where
        self.include = include
    
    def ddl(self, table_name: str, schema: str = "public", model_cls=None) -> str:
        col_names = _resolve_column_names(self.columns, model_cls)
        index_name = self.name or f"idx_{table_name}_{'_'.join(col_names)}"
        
        unique_sql = "UNIQUE " if self.unique else ""
        method_sql = f"USING {self.method} " if self.method else ""
        cols_sql = ", ".join(col_names)
        
        sql = (
            f"CREATE {unique_sql}INDEX IF NOT EXISTS {index_name} "
            f"ON {schema}.{table_name} {method_sql}({cols_sql})"
        )
        
        if self.include:
            include_cols = ", ".join(
                c if isinstance(c, str) else (c.name if hasattr(c, "name") else str(c))
                for c in self.include
            )
            sql += f" INCLUDE ({include_cols})"
        
        if self.where:
            sql += f" WHERE {self.where}"
        
        return sql


# ============================================================
# TABLE OPTIONS CLASS
# ============================================================

class TableOptions:
    """Additional table options.
    
    Args:
        tablespace: PostgreSQL tablespace
        inherits: Parent table(s) for inheritance
        partition_by: Partitioning strategy
        unlogged: Create unlogged table (faster but not crash-safe)
    """
    
    def __init__(
        self,
        *,
        tablespace: Optional[str] = None,
        inherits: Optional[Sequence[str]] = None,
        partition_by: Optional[str] = None,
        unlogged: bool = False,
    ):
        self.tablespace = tablespace
        self.inherits = inherits
        self.partition_by = partition_by
        self.unlogged = unlogged


# ============================================================
# TABLE DECORATOR
# ============================================================

def table(
    name: str,
    *,
    schema: Optional[str] = None,
    constraints: Optional[List[Constraint]] = None,
    indexes: Optional[List[Index]] = None,
    unique_together: Optional[List[Sequence[Union[Column, str]]]] = None,
    options: Optional[TableOptions] = None,
    comment: Optional[str] = None,
):
    """Decorator to define a database table from a PSQLModel class.
    
    Args:
        name: Table name in the database
        schema: Database schema (default: "public")
        constraints: List of Constraint objects (Unique, Check, ForeignKey, etc.)
        indexes: List of Index objects
        unique_together: Legacy support - list of column tuples for unique constraints
        options: TableOptions for additional PostgreSQL options
        comment: Table comment
    
    Example:
        @table(
            name="users",
            schema="auth",
            constraints=[
                UniqueConstraint(User.email),
                CheckConstraint("age >= 0"),
            ],
            indexes=[
                Index(User.email),
            ],
            comment="User accounts table",
        )
        class User(PSQLModel):
            id: serial = Column(primary_key=True)
            email: varchar(255) = Column(nullable=False)
            age: integer = Column()
    """
    
    def wrapper(cls):
        # Set table metadata
        cls.__tablename__ = name
        cls.__schema__ = schema or "public"
        cls.__constraints__ = constraints or []
        cls.__indexes__ = indexes or []
        cls.__table_options__ = options
        cls.__table_comment__ = comment
        cls.__columns__ = {}
        
        # Handle legacy unique_together by converting to UniqueConstraint
        if unique_together:
            for cols in unique_together:
                cls.__constraints__.append(UniqueConstraint(*cols))
        
        # Preserve original annotations for IDE support
        original_annotations = {}
        try:
            annotations = get_type_hints(cls)
            original_annotations = annotations.copy()
        except Exception:
            annotations = getattr(cls, "__annotations__", {}) or {}
            original_annotations = annotations.copy()

        from . import types as _sqltypes
        # Import local para evitar ciclos
        from .relationships import Relationship

        for attr, type_hint in annotations.items():
            if attr.startswith("_"):
                continue
            raw_value = getattr(cls, attr, None)

            # Saltar descriptores de relaciones (no son columnas de BD)
            if isinstance(raw_value, Relationship):
                continue

            if isinstance(raw_value, Column):
                col = raw_value
            else:
                if raw_value is not None:
                    col = Column(default=raw_value)
                else:
                    col = Column()

            col.model = cls
            col.attr_name = attr
            
            # Store the original type hint for IDE support
            col.python_type_hint = type_hint
            
            # Resolve type hint
            resolved_type: Any
            if isinstance(type_hint, type) and issubclass(type_hint, _sqltypes.SQLType):
                resolved_type = type_hint()
            elif isinstance(type_hint, _sqltypes.SQLType):
                resolved_type = type_hint
            else:
                if type_hint is int:
                    resolved_type = _sqltypes.integer()
                elif type_hint is float:
                    resolved_type = _sqltypes.numeric()
                elif type_hint is str:
                    resolved_type = _sqltypes.text()
                elif type_hint is bool:
                    resolved_type = _sqltypes.boolean()
                elif type_hint is dict:
                    resolved_type = _sqltypes.jsonb()
                else:
                    resolved_type = type_hint

            col.type_hint = resolved_type

            # Warning for unrecognized types
            if type_hint is not None:
                try:
                    is_sqltype_class = (
                        isinstance(type_hint, type)
                        and issubclass(type_hint, _sqltypes.SQLType)
                    )
                    is_sqltype_instance = isinstance(type_hint, _sqltypes.SQLType)
                except Exception:
                    is_sqltype_class = False
                    is_sqltype_instance = False

                is_builtin_mapped = type_hint in (int, float, str, bool, dict)

                if not (is_sqltype_class or is_sqltype_instance or is_builtin_mapped):
                    warnings.warn(
                        f"Unrecognized type annotation for {cls.__name__}.{attr}: {type_hint!r}. "
                        "Consider using psqlmodel.orm.types or builtin types.",
                        UserWarning,
                    )

            if col.name is None:
                col.name = attr

            cls.__columns__[attr] = col
            setattr(cls, attr, col)
        
        # Restore original annotations for IDE autocompletion
        # This is crucial for PyCharm, VSCode, etc.
        cls.__annotations__ = original_annotations

        return cls

    return wrapper


# ============================================================
# HELPER: Generate full DDL for a model
# ============================================================

def generate_table_ddl(model_cls) -> List[str]:
    """Generate all DDL statements for a table (CREATE TABLE + constraints + indexes).
    
    Returns a list of SQL statements.
    """
    statements: List[str] = []
    schema = getattr(model_cls, "__schema__", "public") or "public"
    table_name = getattr(model_cls, "__tablename__", model_cls.__name__.lower())
    columns = getattr(model_cls, "__columns__", {})
    constraints = getattr(model_cls, "__constraints__", [])
    indexes = list(getattr(model_cls, "__indexes__", []))  # Copy to avoid modifying original
    options = getattr(model_cls, "__table_options__", None)
    comment = getattr(model_cls, "__table_comment__", None)

    # CREATE SCHEMA (solo si no existe). El motor puede manejar esto globalmente,
    # pero lo mantenemos aquí para DDL de una sola tabla.
    statements.append(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    # Build CREATE TABLE and collect column-level indexes
    col_defs: List[str] = []
    for col_name, col in columns.items():
        try:
            col_defs.append(f"{col_name} {col.ddl()}")
        except Exception:
            # Si algo falla en una columna, la ignoramos para no romper todo el DDL
            continue
        
        # Generate index from Column(index=True) or Column(unique=True)
        if getattr(col, "index", False) or getattr(col, "unique", False):
            idx = Index(
                col,
                unique=getattr(col, "unique", False),
                method=getattr(col, "index_method", None),
            )
            indexes.append(idx)
    
    # Add table-level constraints
    constraint_defs: List[str] = []
    for constraint in constraints:
        try:
            constraint_defs.append(constraint.ddl(table_name, schema, model_cls))
        except Exception:
            continue
    
    all_defs = col_defs + constraint_defs
    
    # Handle table options
    table_prefix = "CREATE"
    if options and options.unlogged:
        table_prefix = "CREATE UNLOGGED"
    
    create_table = f"{table_prefix} TABLE IF NOT EXISTS {schema}.{table_name} ({', '.join(all_defs)})"
    
    if options:
        if options.inherits:
            create_table += f" INHERITS ({', '.join(options.inherits)})"
        if options.partition_by:
            create_table += f" PARTITION BY {options.partition_by}"
        if options.tablespace:
            create_table += f" TABLESPACE {options.tablespace}"
    
    statements.append(create_table + ";")
    
    # Add indexes
    for index in indexes:
        try:
            statements.append(index.ddl(table_name, schema, model_cls) + ";")
        except Exception:
            continue
    
    # Add table comment
    if comment:
        statements.append(f"COMMENT ON TABLE {schema}.{table_name} IS '{comment}';")
    
    return statements