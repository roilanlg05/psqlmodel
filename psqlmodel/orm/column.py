# ============================================================
# column.py – expresiones SQL + clase SQLColumn (modelo + query)
# ============================================================

# -------------------------------
# BASE DE EXPRESIONES
# -------------------------------

class SQLExpression:
    def to_sql(self):
        """Retorna solo el string SQL (legacy)."""
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        """Retorna (sql_template, params) para parametrización segura."""
        raise NotImplementedError()

    def __str__(self):
        return self.to_sql()

    # Permitir combinar expresiones con & (AND) y | (OR)
    def __and__(self, other):
        return LogicalExpression(self, "AND", other)

    def __or__(self, other):
        return LogicalExpression(self, "OR", other)

    def asc(self):
        return SortExpression(self, "ASC")

    def desc(self):
        return SortExpression(self, "DESC")


class SortExpression:
    def __init__(self, expr, direction):
        self.expr = expr
        self.direction = direction

    def to_sql(self):
        return f"{self.expr.to_sql()} {self.direction}"

    def to_sql_params(self):
        sql, params = self.expr.to_sql_params()
        return f"{sql} {self.direction}", params


class RawExpression(SQLExpression):
    def __init__(self, text):
        self.text = text

    def to_sql(self):
        return self.text

    def to_sql_params(self):
        # Si el texto contiene %s/$1, el usuario debe pasar los params manualmente
        return self.text, []

    def As(self, alias):
        """Permite aliasar una expresión cruda: RawExpression("...").As("name")."""
        return Alias(self, alias)


class BinaryExpression(SQLExpression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        left_sql, left_params = (
            self.left.to_sql_params()
            if hasattr(self.left, "to_sql_params")
            else (str(self.left), [])
        )
        right_sql, right_params = (
            self.right.to_sql_params()
            if hasattr(self.right, "to_sql_params")
            else (str(self.right), [])
        )
        # Si el operador es = y el right es un valor, usar placeholder
        if self.op in ("=", "!=", "<", "<=", ">", ">=") and not isinstance(
            self.right, SQLExpression
        ):
            right_sql = "%s"
            right_params = [self.right]
        sql = f"{left_sql} {self.op} {right_sql}"
        return sql, left_params + right_params


class LogicalExpression(SQLExpression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        left_sql, left_params = (
            self.left.to_sql_params()
            if hasattr(self.left, "to_sql_params")
            else (str(self.left), [])
        )
        right_sql, right_params = (
            self.right.to_sql_params()
            if hasattr(self.right, "to_sql_params")
            else (str(self.right), [])
        )
        sql = f"({left_sql} {self.op} {right_sql})"
        return sql, left_params + right_params


class InExpression(SQLExpression):
    """SQL IN / NOT IN expression.
    
    Examples:
        User.id.In(1, 2, 3) → id IN (1, 2, 3)
        User.id.NotIn([1, 2, 3]) → id NOT IN (1, 2, 3)
    """
    def __init__(self, column, values, negate=False):
        self.column = column
        self.values = values  # List of expressions or values
        self.negate = negate

    def to_sql_params(self):
        col_sql, col_params = self.column.to_sql_params()
        
        # Build values SQL
        values_sql = []
        values_params = []
        for val in self.values:
            if isinstance(val, SQLExpression):
                v_sql, v_params = val.to_sql_params()
                values_sql.append(v_sql)
                values_params.extend(v_params)
            else:
                values_sql.append("%s")
                values_params.append(val)
        
        op = "NOT IN" if self.negate else "IN"
        sql = f"{col_sql} {op} ({', '.join(values_sql)})"
        return sql, col_params + values_params


class Alias(SQLExpression):
    def __init__(self, expr, alias):
        self.expr = expr
        self.alias = alias

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        expr_sql, expr_params = (
            self.expr.to_sql_params()
            if hasattr(self.expr, "to_sql_params")
            else (str(self.expr), [])
        )
        sql = f"{expr_sql} AS {self.alias}"
        return sql, expr_params


# ============================================================
# SQL COLUMN – (Modelo + QueryBuilder)
# ============================================================

class Column(SQLExpression):
    def validate_value(self, value):
        """
        Valida el valor según las restricciones de longitud y valor.
        Lanza ValueError si no se cumple alguna restricción.
        """
        # Validación de longitud para tipos string/bit
        if self.max_len is not None and value is not None:
            if isinstance(value, (str, bytes, list)):
                if len(value) > self.max_len:
                    raise ValueError(
                        f"Valor para columna '{self.name}' excede max_len={self.max_len}"
                    )
            elif isinstance(value, int):
                # Para bits representados como int (opcional)
                if value.bit_length() > self.max_len:
                    raise ValueError(
                        f"Valor para columna '{self.name}' excede max_len={self.max_len} bits"
                    )
        if self.min_len is not None and value is not None:
            if isinstance(value, (str, bytes, list)):
                if len(value) < self.min_len:
                    raise ValueError(
                        f"Valor para columna '{self.name}' menor a min_len={self.min_len}"
                    )
        # Validación de valor numérico
        if self.max_value is not None and value is not None:
            if isinstance(value, (int, float)) and value > self.max_value:
                raise ValueError(
                    f"Valor para columna '{self.name}' excede max_value={self.max_value}"
                )
        if self.min_value is not None and value is not None:
            if isinstance(value, (int, float)) and value < self.min_value:
                raise ValueError(
                    f"Valor para columna '{self.name}' menor a min_value={self.min_value}"
                )
        # Puedes llamar a validate_value(value) en los métodos de inserción/actualización del modelo/ORM.

    # Esta columna sirve para:
    # - Declaración de modelos (PSQLModel)
    # - Construcción de queries (QueryBuilder)
    def __init__(
        self,
        *,
        primary_key=False,
        nullable=True,
        default=None,
        foreign_key=None,
        name=None,
        type_hint=None,
        timez=False,
        model=None,
        attr_name=None,
        index=False,
        unique=False,
        index_method=None,
        max_len=None,
        min_len=None,
        max_value=None,
        min_value=None,
        on_delete=None,
    ):
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.foreign_key = foreign_key
        self.on_delete = on_delete
        self.name = name
        self.type_hint = type_hint
        self.timez = timez
        self.model = model
        self.attr_name = attr_name
        self.index = index
        self.unique = unique
        self.index_method = index_method  # btree, hash, gin, gist, etc.
        self.max_len = max_len
        self.min_len = min_len
        self.max_value = max_value
        self.min_value = min_value
        if self.on_delete:
            self.on_delete = str(self.on_delete).strip().upper()
            valid = {"CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"}
            if self.on_delete not in valid:
                raise ValueError(
                    f"on_delete inválido: {self.on_delete}. "
                    f"Opciones: {', '.join(sorted(valid))}"
                )

    # ----------------------------------
    # SQL EXPRESSION PART (QueryBuilder)
    # ----------------------------------

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        # Incluir schema si existe
        schema = getattr(self.model, "__schema__", None)
        table = self.model.__tablename__

        if schema and schema != "public":
            full_table = f"{schema}.{table}"
        else:
            full_table = table

        return f"{full_table}.{self.name}", []

    def As(self, alias):
        return Alias(self, alias)

    # binary ops
    def _bin(self, other, op):
        if isinstance(other, SQLExpression):
            return BinaryExpression(self, op, other)
        return BinaryExpression(self, op, RawExpression(repr(other)))

    def __eq__(self, other):
        return self._bin(other, "=")

    def __ne__(self, other):
        return self._bin(other, "!=")

    def __lt__(self, other):
        return self._bin(other, "<")

    def __le__(self, other):
        return self._bin(other, "<=")

    def __gt__(self, other):
        return self._bin(other, ">")

    def __ge__(self, other):
        return self._bin(other, ">=")

    # Arithmetic operators
    def __add__(self, other):
        return ArithmeticExpression(self, "+", other)

    def __sub__(self, other):
        return ArithmeticExpression(self, "-", other)

    def __mul__(self, other):
        return ArithmeticExpression(self, "*", other)

    def __truediv__(self, other):
        return ArithmeticExpression(self, "/", other)

    def __radd__(self, other):
        return ArithmeticExpression(other, "+", self)

    def __rsub__(self, other):
        return ArithmeticExpression(other, "-", self)

    def __rmul__(self, other):
        return ArithmeticExpression(other, "*", self)

    def __rtruediv__(self, other):
        return ArithmeticExpression(other, "/", self)

    # IN / NOT IN operators
    def In(self, *values):
        """SQL IN operator: column IN (values...)
        
        Examples:
            User.id.In(1, 2, 3)  # id IN (1, 2, 3)
            User.id.In([1, 2, 3])  # id IN (1, 2, 3)
            User.id.In(subquery)  # id IN (SELECT ...)
        """
        # Import here to avoid circular dependency
        from psqlmodel.query.builder import SelectQuery
        
        # Handle single argument
        if len(values) == 1:
            val = values[0]
            if isinstance(val, (list, tuple)):
                values = val
            elif isinstance(val, SelectQuery):
                # Subquery
                return BinaryExpression(self, "IN", SubqueryValue(val))
        
        # Multiple values or unpacked list
        val_expressions = [
            v if isinstance(v, SQLExpression) else v
            for v in values
        ]
        return InExpression(self, val_expressions, negate=False)
    
    def NotIn(self, *values):
        """SQL NOT IN operator: column NOT IN (values...)"""
        from psqlmodel.query.builder import SelectQuery
        
        # Handle single argument
        if len(values) == 1:
            val = values[0]
            if isinstance(val, (list, tuple)):
                values = val
            elif isinstance(val, SelectQuery):
                # Subquery
                return BinaryExpression(self, "NOT IN", SubqueryValue(val))
        
        # Multiple values or unpacked list
        val_expressions = [
            v if isinstance(v, SQLExpression) else v
            for v in values
        ]
        return InExpression(self, val_expressions, negate=True)

    # ----------------------------------
    # METADATA (migraciones)
    # ----------------------------------

    def ddl(self):
        # Guardar contra type_hint ausente o no compatible
        if not self.type_hint:
            type_sql = "TEXT"
        else:
            try:
                base_type = self.type_hint.ddl()
            except Exception:
                base_type = str(self.type_hint)

            # Manejo de max_len/min_len para tipos que lo soportan
            if base_type in ("VARCHAR", "CHAR") and self.max_len:
                type_sql = f"{base_type}({self.max_len})"
            elif base_type in ("BIT", "BIT VARYING") and self.max_len:
                type_sql = f"{base_type}({self.max_len})"
            else:
                type_sql = base_type

        # Si el tipo es timestamp y la columna pide time zone, emitir WITH TIME ZONE
        try:
            from .types import timestamp as _ts_class

            if isinstance(self.type_hint, _ts_class) or getattr(
                self.type_hint, "__class__", None
            ) is _ts_class:
                if getattr(self, "timez", False):
                    type_sql = "TIMESTAMP WITH TIME ZONE"
        except Exception:
            pass

        parts = [type_sql]
        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")

        # Detectar si es un tipo UUID
        try:
            from .types import uuid as _uuid_class

            is_uuid = (
                isinstance(self.type_hint, _uuid_class)
                or getattr(self.type_hint, "__class__", None) is _uuid_class
                or (isinstance(self.type_hint, type) and issubclass(self.type_hint, _uuid_class))
            )
        except Exception:
            is_uuid = False

        # Para UUID: usar gen_random_uuid() siempre (con o sin default Python)
        if is_uuid:
            parts.append("DEFAULT gen_random_uuid()")
        elif self.default is not None:
            if callable(self.default):
                # Callables: evaluar y usar el valor
                val = self.default()
                parts.append(f"DEFAULT '{val}'")
            else:
                # Valor literal
                parts.append(f"DEFAULT '{self.default}'")

        # Foreign key inline (acepta Column, schema.table.col o table.col)
        if self.foreign_key:
            ref_table = None
            ref_col = None

            # Caso 1: se pasó un objeto Column
            if isinstance(self.foreign_key, Column):
                fk_col = self.foreign_key
                fk_model = getattr(fk_col, "model", None)
                fk_schema = (
                    getattr(fk_model, "__schema__", "public") if fk_model else "public"
                )
                fk_table = getattr(fk_model, "__tablename__", None) or (
                    fk_model.__name__.lower() if fk_model else None
                )
                ref_col = getattr(fk_col, "name", None)
                if not (fk_table and ref_col):
                    raise ValueError(
                        "foreign_key Column debe tener model y name definidos."
                    )
                ref_table = f"{fk_schema}.{fk_table}" if fk_schema else fk_table

            else:
                ref_parts = str(self.foreign_key).split(".")
                if len(ref_parts) < 2:
                    raise ValueError(
                        "foreign_key debe ser Column, 'tabla.columna' "
                        "o 'schema.tabla.columna'. "
                        f"Recibido: {self.foreign_key}"
                    )
                if len(ref_parts) == 2:
                    # table.col (no forzamos schema para evitar prefijos incorrectos)
                    fk_table, ref_col = ref_parts
                    ref_table = fk_table
                else:
                    # schema.table.col (o con schema.table compuestos)
                    fk_schema = ref_parts[0]
                    fk_table = ".".join(ref_parts[1:-1])
                    ref_col = ref_parts[-1]
                    ref_table = f"{fk_schema}.{fk_table}" if fk_schema else fk_table

            parts.append(f"REFERENCES {ref_table}({ref_col})")
            if self.on_delete:
                parts.append(f"ON DELETE {self.on_delete}")
        return " ".join(parts)


# ============================================================
# CASE WHEN
# ============================================================

class CaseExpression(SQLExpression):
    def __init__(self):
        self.whens = []
        self.else_value = None
        self.alias = None

    def When(self, cond, val):
        self.whens.append((cond, val))
        return self

    def Else(self, val):
        self.else_value = val
        return self

    def As(self, alias):
        self.alias = alias
        return self

    def _val_to_sql(self, val):
        """Convert a value to SQL string."""
        if hasattr(val, "to_sql"):
            return val.to_sql()
        elif isinstance(val, str):
            return f"'{val}'"
        elif val is None:
            return "NULL"
        else:
            return str(val)

    def _val_to_sql_params(self, val):
        """Convert a value to SQL with params."""
        if hasattr(val, "to_sql_params"):
            return val.to_sql_params()
        elif hasattr(val, "to_sql"):
            return val.to_sql(), []
        elif isinstance(val, str):
            return "%s", [val]
        elif val is None:
            return "NULL", []
        else:
            return "%s", [val]

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        parts = ["CASE"]
        params = []

        for cond, val in self.whens:
            # Condition
            if hasattr(cond, "to_sql_params"):
                cond_sql, cond_params = cond.to_sql_params()
                params.extend(cond_params)
            else:
                cond_sql = str(cond)

            # Value
            val_sql, val_params = self._val_to_sql_params(val)
            params.extend(val_params)

            parts.append(f"WHEN {cond_sql} THEN {val_sql}")

        if self.else_value is not None:
            else_sql, else_params = self._val_to_sql_params(self.else_value)
            params.extend(else_params)
            parts.append(f"ELSE {else_sql}")

        parts.append("END")
        result = " ".join(parts)

        if self.alias:
            result += f" AS {self.alias}"

        return result, params


# Shortcut
def Case():
    return CaseExpression()


# ============================================================
# AGGREGATES (simple y window)
# ============================================================

class AggregateOrWindow(SQLExpression):
    """Representa una función de agregado que puede usarse como simple
    (SUM(col)) o como window function (SUM(col) OVER (...)).
    """

    def __init__(self, func_name, col_sql):
        # col_sql es una cadena SQL ya renderizada (p.ej. col.to_sql())
        self.func_name = func_name
        self.col_sql = col_sql
        self.partition = []
        self.order = []
        self.alias = None
        self.use_window = False
        self.filter_cond = None

    # ---- Window API ----
    def Over(self):
        self.use_window = True
        return self

    def PartitionBy(self, *cols):
        self.partition.extend(cols)
        return self

    def OrderBy(self, *cols):
        self.order.extend(cols)
        return self

    def As(self, alias):
        self.alias = alias
        return self

    def Filter(self, cond):
        """Postgres-style FILTER (WHERE cond) for aggregates.

        Example:
            Count(Order.id).Filter(Order.total > 100)
        """
        self.filter_cond = cond
        return self

    def _base_sql(self):
        return f"{self.func_name}({self.col_sql})"

    def to_sql(self):
        sql = self._base_sql()
        if self.filter_cond is not None:
            sql += f" FILTER (WHERE {self.filter_cond.to_sql()})"
        if self.use_window:
            parts = []
            if self.partition:
                parts.append(
                    "PARTITION BY " + ", ".join(c.to_sql() for c in self.partition)
                )
            if self.order:
                parts.append("ORDER BY " + ", ".join(c.to_sql() for c in self.order))
            window = " OVER (" + " ".join(parts) + ")" if parts else " OVER ()"
            sql += window
        if self.alias:
            sql += f" AS {self.alias}"
        return sql

    def to_sql_params(self):
        # Por compatibilidad: actualmente no se propagan params en aggregates.
        # Devolvemos el SQL completo como string y lista vacía de parámetros.
        return self.to_sql(), []

    # ---- Operadores binarios (>, <, ==, etc.) ----
    def _bin(self, other, op):
        if isinstance(other, SQLExpression):
            return BinaryExpression(self, op, other)
        return BinaryExpression(self, op, RawExpression(repr(other)))

    def __eq__(self, other):
        return self._bin(other, "=")

    def __ne__(self, other):
        return self._bin(other, "!=")

    def __lt__(self, other):
        return self._bin(other, "<")

    def __le__(self, other):
        return self._bin(other, "<=")

    def __gt__(self, other):
        return self._bin(other, ">")

    def __ge__(self, other):
        return self._bin(other, ">=")


def _col_sql(col):
    # helper para aceptar Column, SQLExpression o texto crudo en agregados
    if isinstance(col, SQLExpression):
        return col.to_sql()
    return str(col)


def Sum(col):
    """SUM(col) que puede usarse como agregado simple o window.

    Ejemplos:
        Sum(Order.total) > 500          # agregado simple
        Sum(Order.total).Over().PartitionBy(Order.user_id)  # window
    """
    return AggregateOrWindow("SUM", _col_sql(col))


def Avg(col):
    return AggregateOrWindow("AVG", _col_sql(col))


def Count(col):
    return AggregateOrWindow("COUNT", _col_sql(col))


def RowNumber():
    """ROW_NUMBER() solo como window function."""
    return AggregateOrWindow("ROW_NUMBER", "")


# ============================================================
# EXISTS
# ============================================================

class ExistsExpression(SQLExpression):
    def __init__(self, query, negate=False):
        self.query = query
        self.negate = negate

    def to_sql(self):
        prefix = "NOT EXISTS" if self.negate else "EXISTS"
        return f"{prefix} ({self.query.to_sql()})"

    def to_sql_params(self):
        prefix = "NOT EXISTS" if self.negate else "EXISTS"
        if hasattr(self.query, "to_sql_params"):
            inner_sql, inner_params = self.query.to_sql_params()
        else:
            inner_sql, inner_params = self.query.to_sql(), []
        return f"{prefix} ({inner_sql})", inner_params


def Exists(q):
    return ExistsExpression(q)


def NotExists(q):
    return ExistsExpression(q, negate=True)


# ============================================================
# SQL FUNCTIONS (NOW, COALESCE, etc.)
# ============================================================

class FuncExpression(SQLExpression):
    """Represents a SQL function call like NOW(), COALESCE(), etc."""

    def __init__(self, func_name: str, *args):
        self.func_name = func_name
        self.args = args
        self._alias = None

    def As(self, alias: str):
        self._alias = alias
        return self

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        parts = []
        params = []
        for arg in self.args:
            if hasattr(arg, "to_sql_params"):
                arg_sql, arg_params = arg.to_sql_params()
                parts.append(arg_sql)
                params.extend(arg_params)
            elif hasattr(arg, "to_sql"):
                parts.append(arg.to_sql())
            else:
                # Literal value - use placeholder
                parts.append("%s")
                params.append(arg)

        sql = f"{self.func_name}({', '.join(parts)})"
        if self._alias:
            sql = f"{sql} AS {self._alias}"
        return sql, params

    # Binary ops for comparisons
    def _bin(self, other, op):
        if isinstance(other, SQLExpression):
            return BinaryExpression(self, op, other)
        return BinaryExpression(self, op, RawExpression(repr(other)))

    def __eq__(self, other):
        return self._bin(other, "=")

    def __ne__(self, other):
        return self._bin(other, "!=")

    def __lt__(self, other):
        return self._bin(other, "<")

    def __le__(self, other):
        return self._bin(other, "<=")

    def __gt__(self, other):
        return self._bin(other, ">")

    def __ge__(self, other):
        return self._bin(other, ">=")

    # Arithmetic ops
    def __add__(self, other):
        return ArithmeticExpression(self, "+", other)

    def __sub__(self, other):
        return ArithmeticExpression(self, "-", other)

    def __mul__(self, other):
        return ArithmeticExpression(self, "*", other)

    def __truediv__(self, other):
        return ArithmeticExpression(self, "/", other)

    def __radd__(self, other):
        return ArithmeticExpression(other, "+", self)

    def __rsub__(self, other):
        return ArithmeticExpression(other, "-", self)

    def __rmul__(self, other):
        return ArithmeticExpression(other, "*", self)

    def __rtruediv__(self, other):
        return ArithmeticExpression(other, "/", self)


# Common SQL functions
def Now():
    """NOW() - current timestamp."""
    return FuncExpression("NOW")


def Coalesce(*args):
    """COALESCE(a, b, c, ...) - return first non-null."""
    return FuncExpression("COALESCE", *args)


def Nullif(a, b):
    """NULLIF(a, b) - return NULL if a = b."""
    return FuncExpression("NULLIF", a, b)


def Greatest(*args):
    """GREATEST(a, b, ...) - return largest value."""
    return FuncExpression("GREATEST", *args)


def Least(*args):
    """LEAST(a, b, ...) - return smallest value."""
    return FuncExpression("LEAST", *args)


def Lower(expr):
    """LOWER(text) - lowercase."""
    return FuncExpression("LOWER", expr)


def Upper(expr):
    """UPPER(text) - uppercase."""
    return FuncExpression("UPPER", expr)


def Length(expr):
    """LENGTH(text) - string length."""
    return FuncExpression("LENGTH", expr)


def Concat(*args):
    """CONCAT(a, b, ...) - concatenate strings."""
    return FuncExpression("CONCAT", *args)


def Func(name: str, *args):
    """Generic function: Func('my_func', arg1, arg2)."""
    return FuncExpression(name, *args)


# ============================================================
# JSONB FUNCTIONS
# ============================================================

def JsonbBuildObject(*pairs):
    """jsonb_build_object(key1, val1, key2, val2, ...).

    Usage:
        JsonbBuildObject('ip', '192.168.1.1', 'device', 'iPhone')
    """
    return FuncExpression("jsonb_build_object", *pairs)


def JsonbAgg(expr):
    """jsonb_agg(expr) - aggregate into JSONB array."""
    return FuncExpression("jsonb_agg", expr)


def ToJsonb(expr):
    """to_jsonb(expr) - convert to JSONB."""
    return FuncExpression("to_jsonb", expr)


def JsonbExtract(col, key):
    """Extract from JSONB: col->'key' or col->>'key'."""
    return RawExpression(
        f"{col.to_sql() if hasattr(col, 'to_sql') else col}->'{key}'"
    )


def JsonbExtractText(col, key):
    """Extract text from JSONB: col->>'key'."""
    return RawExpression(
        f"{col.to_sql() if hasattr(col, 'to_sql') else col}->>'{key}'"
    )


# ============================================================
# ARITHMETIC EXPRESSIONS
# ============================================================

class ArithmeticExpression(SQLExpression):
    """Represents arithmetic: a + b, a * b, etc."""

    def __init__(self, left, op: str, right):
        self.left = left
        self.op = op
        self.right = right
        self._alias = None

    def As(self, alias: str):
        self._alias = alias
        return self

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        # Left
        if hasattr(self.left, "to_sql_params"):
            left_sql, left_params = self.left.to_sql_params()
        elif hasattr(self.left, "to_sql"):
            left_sql, left_params = self.left.to_sql(), []
        else:
            left_sql, left_params = "%s", [self.left]

        # Right
        if hasattr(self.right, "to_sql_params"):
            right_sql, right_params = self.right.to_sql_params()
        elif hasattr(self.right, "to_sql"):
            right_sql, right_params = self.right.to_sql(), []
        else:
            right_sql, right_params = "%s", [self.right]

        sql = f"({left_sql} {self.op} {right_sql})"
        if self._alias:
            sql = f"{sql} AS {self._alias}"
        return sql, left_params + right_params

    # Chain arithmetic
    def __add__(self, other):
        return ArithmeticExpression(self, "+", other)

    def __sub__(self, other):
        return ArithmeticExpression(self, "-", other)

    def __mul__(self, other):
        return ArithmeticExpression(self, "*", other)

    def __truediv__(self, other):
        return ArithmeticExpression(self, "/", other)

    def __radd__(self, other):
        return ArithmeticExpression(other, "+", self)

    def __rsub__(self, other):
        return ArithmeticExpression(other, "-", self)

    def __rmul__(self, other):
        return ArithmeticExpression(other, "*", self)

    def __rtruediv__(self, other):
        return ArithmeticExpression(other, "/", self)

    # Comparisons
    def _bin(self, other, op):
        if isinstance(other, SQLExpression):
            return BinaryExpression(self, op, other)
        return BinaryExpression(self, op, RawExpression(repr(other)))

    def __eq__(self, other):
        return self._bin(other, "=")

    def __ne__(self, other):
        return self._bin(other, "!=")

    def __lt__(self, other):
        return self._bin(other, "<")

    def __le__(self, other):
        return self._bin(other, "<=")

    def __gt__(self, other):
        return self._bin(other, ">")

    def __ge__(self, other):
        return self._bin(other, ">=")


# ============================================================
# SUBQUERY AS VALUE
# ============================================================

class SubqueryValue(SQLExpression):
    """Wrap a SELECT query to use as a scalar value in INSERT/UPDATE."""

    def __init__(self, query):
        self.query = query
        self._alias = None

    def As(self, alias: str):
        self._alias = alias
        return self

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        if hasattr(self.query, "to_sql_params"):
            inner_sql, inner_params = self.query.to_sql_params()
        else:
            inner_sql, inner_params = self.query.to_sql(), []

        sql = f"({inner_sql})"
        if self._alias:
            sql = f"{sql} AS {self._alias}"
        return sql, inner_params

    # Arithmetic operations for subquery results
    def __add__(self, other):
        return ArithmeticExpression(self, "+", other)

    def __sub__(self, other):
        return ArithmeticExpression(self, "-", other)

    def __mul__(self, other):
        return ArithmeticExpression(self, "*", other)

    def __truediv__(self, other):
        return ArithmeticExpression(self, "/", other)


def Scalar(query):
    """Wrap a query to use as scalar value: Scalar(Select(User.id).Where(...))"""
    return SubqueryValue(query)


# ============================================================
# EXCLUDED (for UPSERT DO UPDATE)
# ============================================================

class ExcludedColumn(SQLExpression):
    """Reference to EXCLUDED.column_name in ON CONFLICT DO UPDATE."""

    def __init__(self, column):
        self.column = column

    def to_sql(self):
        col_name = self.column.name if hasattr(self.column, "name") else str(self.column)
        return f"EXCLUDED.{col_name}"

    def to_sql_params(self):
        return self.to_sql(), []


def Excluded(column):
    """Reference EXCLUDED.col in ON CONFLICT DO UPDATE SET."""
    return ExcludedColumn(column)


# ============================================================
# VALUES expression for INSERT ... SELECT FROM VALUES
# ============================================================

class ValuesExpression(SQLExpression):
    """VALUES clause: VALUES ('a', 1), ('b', 2), ..."""

    def __init__(self, *rows):
        self.rows = rows
        self._alias = None
        self._col_aliases = None

    def As(self, alias: str, *col_names):
        """Alias the VALUES clause: AS v(col1, col2, ...)."""
        self._alias = alias
        self._col_aliases = col_names if col_names else None
        return self

    def to_sql(self):
        sql, _ = self.to_sql_params()
        return sql

    def to_sql_params(self):
        row_parts = []
        params = []
        for row in self.rows:
            placeholders = []
            for val in row:
                if hasattr(val, "to_sql_params"):
                    val_sql, val_params = val.to_sql_params()
                    placeholders.append(val_sql)
                    params.extend(val_params)
                else:
                    placeholders.append("%s")
                    params.append(val)
            row_parts.append(f"({', '.join(placeholders)})")

        sql = f"VALUES {', '.join(row_parts)}"
        if self._alias:
            sql = f"({sql}) AS {self._alias}"
            if self._col_aliases:
                sql += f"({', '.join(self._col_aliases)})"
        return sql, params


def Values(*rows):
    """Create VALUES expression: Values(('a', 1), ('b', 2))"""
    return ValuesExpression(*rows)
