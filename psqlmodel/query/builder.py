# ============================================================
# query_builder.py – Select, Join, Query (compatible con column.py + PSQLModel)
# ============================================================

from ..orm.column import (
    SQLExpression,
    Column,
    Alias,
    RawExpression,
)


# ============================================================
# QUERY BASE
# ============================================================

class Query(SQLExpression):
    """
    Base de todas las consultas (Select, Insert, Update, Delete, etc.)
    """

    def to_sql_params(self):
        raise NotImplementedError()

    # Helpers de conveniencia para ejecutar una query con un Engine.
    def execute(self, engine, *params):
        """Ejecutar esta query en modo síncrono usando un Engine."""
        return engine.execute(self, *params)

    async def execute_async(self, engine, *params):
        """Ejecutar esta query en modo asíncrono usando un Engine."""
        return await engine.execute_async(self, *params)


# ============================================================
# JOIN OBJECT
# ============================================================

class Join:
    def __init__(self, parent_query, model):
        self.parent_query = parent_query
        self.model = model
        self.condition = None
        self.kind = "INNER"  # INNER, LEFT, RIGHT, FULL, CROSS

    def On(self, cond):
        # Guardamos la condición en el propio Join y devolvemos
        # el SelectQuery padre para permitir cadenas fluidas como
        # Select(...).Join(Model).On(cond).DistinctOn(...)
        self.condition = cond
        return self.parent_query

    # Variantes de JOIN
    def Left(self):
        self.kind = "LEFT"
        return self

    def Right(self):
        self.kind = "RIGHT"
        return self

    def Inner(self):
        self.kind = "INNER"
        return self

    def Full(self):
        self.kind = "FULL"
        return self

    def Cross(self):
        self.kind = "CROSS"
        return self


def _detect_relationship(owner_model, target_model):
    """Helper to detect relationship between two models using __relations__."""
    relations = getattr(owner_model, "__relations__", {}) if hasattr(owner_model, "__relations__") else {}
    for attr_name, rel in relations.items():
        try:
            target = rel._resolve_target()
        except Exception:
            target = None
        if target is not target_model:
            continue
        try:
            rel._detect_relationship_type()
        except Exception:
            pass
        rel_type = getattr(rel, "_relationship_type", None)
        fk_name = getattr(rel, "_foreign_key", None)
        return {"type": rel_type, "fk_name": fk_name, "attr_name": attr_name, "rel": rel}

    # Fallback heuristic
    def to_singular(name: str) -> str:
        return name[:-1] if name.endswith("s") else name

    owner_table = getattr(owner_model, "__tablename__", owner_model.__name__.lower())
    target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())

    owner_fk_name = f"{to_singular(target_table)}_id"
    if owner_fk_name in getattr(owner_model, "__columns__", {}):
        col = owner_model.__columns__[owner_fk_name]
        is_unique = getattr(col, "unique", False)
        return {
            "type": "one_to_one" if is_unique else "many_to_one",
            "fk_name": owner_fk_name,
            "attr_name": to_singular(target_table),
            "rel": None,
        }

    target_fk_name = f"{to_singular(owner_table)}_id"
    if target_fk_name in getattr(target_model, "__columns__", {}):
        return {
            "type": "one_to_many",
            "fk_name": target_fk_name,
            "attr_name": target_table,
            "rel": None,
        }

    return None


# ============================================================
# SELECT
# ============================================================

class SelectQuery(Query):
    def __init__(self, *columns):
        if not columns:
            raise ValueError("Select requiere al menos una columna, modelo o expresión")

        # Flags / metadata
        self.select_all = False
        self.base_model = None
        self._from_model = None  # Para Include/relaciones
        self._from = None        # Fuente explícita de FROM (modelo, str, subquery)

        # WHERE / JOIN / GROUP / ORDER / etc.
        self.where = []
        self.joins = []
        self.group_by = []
        self.having = None
        self.order_by = []
        self.limit = None
        self.offset = None
        self.distinct = False
        self.distinct_on = None
        self.includes = []      # Para eager loading de relaciones
        self._last_column = None  # Para Like/ILike/NotLike/In/NotIn

        # -------------------------------
        # 1) Determinar base_model
        # -------------------------------
        candidates = []
        for c in columns:
            expr = c.expr if isinstance(c, Alias) else c
            if isinstance(expr, Column) and getattr(expr, "model", None) is not None:
                candidates.append(expr.model)
            elif hasattr(expr, "__tablename__"):
                candidates.append(expr)

        if candidates:
            self.base_model = candidates[0]
            self._from_model = self.base_model

        # -------------------------------
        # 2) Modo SELECT * FROM model
        # -------------------------------
        if len(columns) == 1 and hasattr(columns[0], "__tablename__"):
            # Select(User) → SELECT * FROM users
            self.select_all = True
            self.columns = []
            if self.base_model is None:
                self.base_model = columns[0]
                self._from_model = self.base_model
            return

        # -------------------------------
        # 3) Normalizar columnas
        #     - str  → RawExpression
        #     - int/float/bool → RawExpression
        #     - Model → se expande a todas sus columnas
        # -------------------------------
        normalized = []
        for c in columns:
            expr = c.expr if isinstance(c, Alias) else c
            if isinstance(expr, str):
                # str como columna cruda
                normalized.append(RawExpression(expr))
            elif isinstance(expr, (int, float, bool)):
                # Números y booleanos → RawExpression
                normalized.append(RawExpression(str(expr)))
            else:
                normalized.append(c)

        expanded = []
        for c in normalized:
            expr = c.expr if isinstance(c, Alias) else c
            if hasattr(expr, "__tablename__") and hasattr(expr, "__columns__"):
                # Modelo dentro del SELECT → expandir a todas sus columnas
                model = expr
                cols_map = getattr(model, "__columns__", {})
                for name, col in cols_map.items():
                    bound_col = getattr(model, name, col)
                    expanded.append(bound_col)
                # Asegurar base_model si aún no estaba
                if self.base_model is None:
                    self.base_model = model
                    self._from_model = model
            else:
                expanded.append(c)

        self.columns = tuple(expanded)

    # ------------------------
    # FROM
    # ------------------------
    def From(self, source):
        """Setea la fuente explícita de FROM: modelo, nombre de tabla/CTE (str) o subquery."""
        self._from = source
        if self.base_model is None and hasattr(source, "__tablename__"):
            self.base_model = source
            self._from_model = source
        return self

    # ------------------------
    # DISTINCT
    # ------------------------
    def Distinct(self):
        self.distinct = True
        return self

    def DistinctOn(self, *cols):
        self.distinct_on = cols
        return self

    # ------------------------
    # WHERE SYSTEM
    # ------------------------
    def Where(self, cond):
        # Primera condición usa WHERE; las siguientes WHERE se consideran AND
        # Si cond es una columna, se guarda para usar con Like/ILike/NotLike
        if not self.where:
            self.where.append(("WHERE", cond))
        else:
            self.where.append(("AND", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def And(self, cond):
        self.where.append(("AND", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def Or(self, cond):
        self.where.append(("OR", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def Like(self, pattern: str):
        """SQL LIKE pattern matching después de Where.

        Usage:
            Select(User).Where(User.email).Like("user@%")
        """
        if not self._last_column:
            raise ValueError("Like() debe usarse después de Where(columna)")
        from ..orm.column import RawExpression
        # Reemplazar la última condición con LIKE
        if self.where and self.where[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where[-1]
            like_expr = RawExpression(f"{self._last_column.to_sql()} LIKE {repr(pattern)}")
            self.where[-1] = (op, like_expr)
        return self

    def ILike(self, pattern: str):
        """Case-insensitive LIKE (PostgreSQL) después de Where.

        Usage:
            Select(User).Where(User.email).ILike("user@%")
        """
        if not self._last_column:
            raise ValueError("ILike() debe usarse después de Where(columna)")
        from ..orm.column import RawExpression
        if self.where and self.where[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where[-1]
            ilike_expr = RawExpression(f"{self._last_column.to_sql()} ILIKE {repr(pattern)}")
            self.where[-1] = (op, ilike_expr)
        return self

    def NotLike(self, pattern: str):
        """SQL NOT LIKE pattern matching después de Where.

        Usage:
            Select(User).Where(User.email).NotLike("test%")
        """
        if not self._last_column:
            raise ValueError("NotLike() debe usarse después de Where(columna)")
        from ..orm.column import RawExpression
        if self.where and self.where[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where[-1]
            notlike_expr = RawExpression(f"{self._last_column.to_sql()} NOT LIKE {repr(pattern)}")
            self.where[-1] = (op, notlike_expr)
        return self

    def In(self, values):
        """SQL IN después de Where.

        Usage:
            Select(User).Where(User.id).In([1, 2, 3])
            Select(User).Where(User.role).In(Select(Role.name))
        """
        if not self._last_column:
            raise ValueError("In() debe usarse después de Where(columna)")
        from ..orm.column import RawExpression

        # Si values es una Query (subconsulta)
        if hasattr(values, "to_sql"):
            in_expr = RawExpression(f"{self._last_column.to_sql()} IN ({values.to_sql()})")
        else:
            vals = ", ".join(repr(v) for v in values)
            in_expr = RawExpression(f"{self._last_column.to_sql()} IN ({vals})")

        if self.where and self.where[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where[-1]
            self.where[-1] = (op, in_expr)
        return self

    def NotIn(self, values):
        """SQL NOT IN después de Where.

        Usage:
            Select(User).Where(User.id).NotIn([1, 2, 3])
            Select(User).Where(User.role).NotIn(Select(Role.name))
        """
        if not self._last_column:
            raise ValueError("NotIn() debe usarse después de Where(columna)")
        from ..orm.column import RawExpression

        # Si values es una Query (subconsulta)
        if hasattr(values, "to_sql"):
            notin_expr = RawExpression(f"{self._last_column.to_sql()} NOT IN ({values.to_sql()})")
        else:
            vals = ", ".join(repr(v) for v in values)
            notin_expr = RawExpression(f"{self._last_column.to_sql()} NOT IN ({vals})")

        if self.where and self.where[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where[-1]
            self.where[-1] = (op, notin_expr)
        return self

    # ------------------------
    # JOIN SYSTEM
    # ------------------------
    def Join(self, model):
        j = Join(self, model)
        self.joins.append(j)
        return j

    def LeftJoin(self, model):
        j = Join(self, model)
        j.kind = "LEFT"
        self.joins.append(j)
        return j

    def RightJoin(self, model):
        j = Join(self, model)
        j.kind = "RIGHT"
        self.joins.append(j)
        return j

    def InnerJoin(self, model):
        j = Join(self, model)
        j.kind = "INNER"
        self.joins.append(j)
        return j

    def FullJoin(self, model):
        j = Join(self, model)
        j.kind = "FULL"
        self.joins.append(j)
        return j

    def CrossJoin(self, model):
        j = Join(self, model)
        j.kind = "CROSS"
        self.joins.append(j)
        # CROSS JOIN no requiere .On(), devolver el SelectQuery para seguir encadenando
        return self

    # ------------------------
    # EAGER VIA JOIN RELACIONADO
    # ------------------------
    def JoinRelated(self, target_model, kind: str = "INNER"):
        """Agrega un JOIN automático usando metadata de relaciones entre base_model y target_model.

        Útil para eager loading basado en JOIN sin escribir condiciones a mano.
        Soporta many-to-one, one-to-one y one-to-many. Para many-to-many requiere secondary.
        """
        if self.base_model is None:
            raise ValueError("JoinRelated requiere un base_model en el SELECT")

        rel_info = _detect_relationship(self.base_model, target_model)
        if not rel_info:
            raise ValueError(f"No se detectó relación entre {self.base_model} y {target_model}")

        rel_type = rel_info["type"]
        fk_name = rel_info["fk_name"]
        rel = rel_info["rel"]

        # Resolver primary keys
        def _pk(model_cls):
            for name, col in getattr(model_cls, "__columns__", {}).items():
                if getattr(col, "primary_key", False):
                    return name
            return None

        if rel_type in ("many_to_one", "one_to_one"):
            fk_col = getattr(self.base_model, fk_name)
            target_pk_name = _pk(target_model)
            if target_pk_name is None:
                raise ValueError(f"No PK en {target_model}")
            target_pk_col = getattr(target_model, target_pk_name)
            cond = fk_col == target_pk_col
            j = Join(self, target_model)
            j.kind = kind.upper()
            j.condition = cond
            self.joins.append(j)
            return j

        if rel_type == "one_to_many":
            base_pk_name = _pk(self.base_model)
            if base_pk_name is None:
                raise ValueError(f"No PK en {self.base_model}")
            base_pk_col = getattr(self.base_model, base_pk_name)
            fk_col = getattr(target_model, fk_name)
            cond = fk_col == base_pk_col
            j = Join(self, target_model)
            j.kind = kind.upper()
            j.condition = cond
            self.joins.append(j)
            return j

        if rel_type == "many_to_many" and rel and rel.secondary:
            # JOIN con tabla intermedia y luego con target
            owner_table = getattr(self.base_model, "__tablename__", None)
            target_pk_name = _pk(target_model)
            base_pk_name = _pk(self.base_model)
            if not target_pk_name or not base_pk_name:
                raise ValueError("PK no encontrada para many_to_many")

            # nombres de columnas FK en junction
            owner_fk = f"{owner_table[:-1] if owner_table.endswith('s') else owner_table}_id"
            target_table = getattr(target_model, "__tablename__", target_model.__name__.lower())
            target_fk = f"{target_table[:-1] if target_table.endswith('s') else target_table}_id"

            # JOIN junction
            junction_table = rel.secondary
            from ..orm.column import RawExpression
            cond_junction = RawExpression(
                f"{junction_table}.{owner_fk} = {self.base_model.__tablename__}.{base_pk_name}"
            )
            j1 = Join(self, junction_table)
            j1.kind = kind.upper()
            j1.condition = cond_junction
            self.joins.append(j1)

            # JOIN target on junction.target_fk = target.pk
            cond_target = RawExpression(
                f"{junction_table}.{target_fk} = {target_model.__tablename__}.{target_pk_name}"
            )
            j2 = Join(self, target_model)
            j2.kind = kind.upper()
            j2.condition = cond_target
            self.joins.append(j2)
            return j2

        raise ValueError(f"Tipo de relación '{rel_type}' no soportado para JoinRelated")

    # ------------------------
    # EAGER LOADING - INCLUDE
    # ------------------------
    def Include(self, target):
        """Eager load de relaciones automáticas.

        El sistema detecta automáticamente la relación entre el modelo principal
        y el modelo/columna/subconsulta proporcionada.
        """
        self.includes.append(target)
        return self

    # ------------------------
    # GROUP BY / HAVING
    # ------------------------
    def GroupBy(self, *cols):
        self.group_by.extend(cols)
        return self

    def Having(self, cond):
        # Permite pasar una sola condición o una LogicalExpression
        # construida con & / |, igual que en Where.
        self.having = cond
        return self

    # ------------------------
    # ORDER BY
    # ------------------------
    def OrderBy(self, *cols):
        # Guardamos solo las columnas; la dirección se aplicará luego
        # con Asc()/Desc() sobre el último grupo añadido.
        if not cols:
            return self
        self.order_by.append({"cols": list(cols), "direction": None})
        return self

    def Asc(self):
        if not self.order_by:
            return self
        self.order_by[-1]["direction"] = "ASC"
        return self

    def Desc(self):
        if not self.order_by:
            return self
        self.order_by[-1]["direction"] = "DESC"
        return self

    # ------------------------
    # LIMIT / OFFSET
    # ------------------------
    def Limit(self, n):
        self.limit = n
        return self

    def Offset(self, n):
        self.offset = n
        return self

    # ------------------------
    # EXEC HELPERS (SessionManager)
    # ------------------------
    def Exec(self):
        """
        Ejecuta este SELECT usando la Session síncrona actual
        registrada en el SessionManager.

        Uso:
            Select(User).Where(User.id == 1).Exec().first()
            Select(User).Where(...).all()   # alias directo
        """
        from ..core.session import SessionManager  # import local → evita ciclos
        session = SessionManager.require_current()
        return session.exec(self)

    def all(self):
        """
        Ejecuta este SELECT en la Session actual y devuelve .all().

        Uso:
            users = Select(User).Where(User.is_active == True).all()
        """
        return self.Exec().all()

    def first(self):
        """
        Ejecuta este SELECT en la Session actual y devuelve .first().

        Uso:
            user = Select(User).Where(User.id == 1).first()
        """
        return self.Exec().first()

    def one(self):
        """
        Ejecuta este SELECT en la Session actual y devuelve .one().

        Lanza ValueError si no hay resultados o hay más de uno.
        """
        return self.Exec().one()

    def one_or_none(self):
        """
        Ejecuta este SELECT en la Session actual y devuelve .one_or_none().

        Devuelve None si no hay resultados y lanza si hay más de uno.
        """
        return self.Exec().one_or_none()

    async def all_async(self):
        """
        Versión asíncrona usando AsyncSession actual via SessionManager.

        Uso:
            async with AsyncSession(engine) as s:
                users = await Select(User).Where(...).all_async()
        """
        from ..core.session import SessionManager  # import local → evita ciclos
        session = SessionManager.require_current_async()
        return await session.exec(self).all()

    async def first_async(self):
        """
        Versión asíncrona: devuelve el primer resultado o None.
        """
        from ..core.session import SessionManager
        session = SessionManager.require_current_async()
        return await session.exec(self).first()

    async def one_async(self):
        """
        Versión asíncrona: devuelve exactamente un resultado o lanza error.
        """
        from ..core.session import SessionManager
        session = SessionManager.require_current_async()
        return await session.exec(self).one()

    async def one_or_none_async(self):
        """
        Versión asíncrona: devuelve uno, None o lanza si hay más de uno.
        """
        from ..core.session import SessionManager
        session = SessionManager.require_current_async()
        return await session.exec(self).one_or_none()

    # ============================================================
    # SQL GENERATION
    # ============================================================

    def to_sql_params(self):
        sql = ["SELECT"]
        params = []

        # DISTINCT
        if self.distinct_on:
            sql.append(
                "DISTINCT ON (" +
                ", ".join(c.to_sql() for c in self.distinct_on) +
                ")"
            )
        elif self.distinct:
            sql.append("DISTINCT")

        # COLUMNS
        if self.select_all:
            # SELECT * FROM table
            sql.append("*")
        else:
            if not self.columns:
                raise ValueError("SELECT requiere al menos una columna cuando no se usa Select(Model) directo")
            sql.append(", ".join(c.to_sql() for c in self.columns))

        # FROM
        if self._from is not None:
            src = self._from
            # Subquery
            if hasattr(src, "to_sql_params"):
                from_sql, from_params = src.to_sql_params()
                sql.append(f"FROM ({from_sql})")
                params.extend(from_params)
            # Modelo
            elif hasattr(src, "__tablename__"):
                schema = getattr(src, "__schema__", None)
                table = src.__tablename__
                if schema and schema != "public":
                    full_table = f"{schema}.{table}"
                else:
                    full_table = table
                sql.append(f"FROM {full_table}")
            # Nombre crudo (tabla/CTE)
            else:
                sql.append(f"FROM {src}")
        else:
            # FROM por base_model
            if self.base_model is None:
                raise ValueError(
                    "No se pudo determinar la tabla base para SELECT; "
                    "usa Select(Model) o Select(...).From('tabla')"
                )
            schema = getattr(self.base_model, "__schema__", None)
            table = self.base_model.__tablename__
            if schema and schema != "public":
                full_table = f"{schema}.{table}"
            else:
                full_table = table
            sql.append(f"FROM {full_table}")

        # JOINS
        for j in self.joins:
            join_sql, join_params = j.condition.to_sql_params() if j.condition else ("", [])
            # Qualify table with schema if present
            j_schema = getattr(j.model, "__schema__", None)
            j_table = getattr(j.model, "__tablename__", None) if hasattr(j.model, "__tablename__") else None

            if j_table is None:
                # Por compatibilidad, si el join.model no es un modelo, lo usamos crudo
                j_full_table = str(j.model)
            else:
                j_full_table = f"{j_schema}.{j_table}" if j_schema and j_schema != "public" else j_table

            if j.kind == "CROSS":
                sql.append(f"CROSS JOIN {j_full_table}")
            else:
                join_kw = {
                    "INNER": "JOIN",
                    "LEFT": "LEFT JOIN",
                    "RIGHT": "RIGHT JOIN",
                    "FULL": "FULL JOIN",
                }.get(j.kind, "JOIN")
                if join_sql:
                    sql.append(f"{join_kw} {j_full_table} ON {join_sql}")
                else:
                    sql.append(f"{join_kw} {j_full_table}")
            params.extend(join_params)

        # WHERE
        if self.where:
            where_fragments = []
            for op, cond in self.where:
                if hasattr(cond, "to_sql_params"):
                    cond_sql, cond_params = cond.to_sql_params()
                else:
                    cond_sql, cond_params = str(cond), []
                where_fragments.append(f"{op} {cond_sql}")
                params.extend(cond_params)
            sql.append(" ".join(where_fragments))

        # GROUP BY
        if self.group_by:
            sql.append("GROUP BY " + ", ".join(c.to_sql() for c in self.group_by))

        # HAVING
        if self.having:
            if hasattr(self.having, "to_sql_params"):
                having_sql, having_params = self.having.to_sql_params()
            else:
                having_sql, having_params = str(self.having), []
            sql.append("HAVING " + having_sql)
            params.extend(having_params)

        # ORDER BY
        if self.order_by:
            order_fragments = []
            for entry in self.order_by:
                cols = entry["cols"]
                direction = entry["direction"] or "ASC"
                for col in cols:
                    if hasattr(col, "to_sql"):
                        order_fragments.append(f"{col.to_sql()} {direction}")
                    else:
                        order_fragments.append(f"{str(col)} {direction}")
            sql.append("ORDER BY " + ", ".join(order_fragments))

        # LIMIT
        if self.limit is not None:
            sql.append("LIMIT %s")
            params.append(self.limit)

        # OFFSET
        if self.offset is not None:
            sql.append("OFFSET %s")
            params.append(self.offset)

        return "\n".join(sql), params


def Select(*columns):
    """Convenience function that returns a SelectQuery instance so callers
    can use `Select(...)` as a function and chain `.From()`, `.Where()`, etc.

    Ejemplos:
        Select(User)                          # SELECT * FROM users
        Select(User.id, User.email)           # columnas de un modelo
        Select(User, Driver, Order.id)        # todas las columnas de User y Driver + Order.id
        Select("id", "email").From("auth.users")
        Select(RawExpression("*")).From("active_users")  # CTE
    """
    return SelectQuery(*columns)


# ============================================================
# DELETE
# ============================================================

class DeleteQuery(Query):
    """DELETE query builder with WHERE support.

    Usage:
        Delete(User).Where(User.id == 5).execute(engine)
        Delete(User).Where(User.is_active == False)
    """

    def __init__(self, model):
        self.model = model
        self.where_clauses = []
        self._returning = None
        self._last_column = None  # Para In/NotIn

    def Where(self, cond):
        if not self.where_clauses:
            self.where_clauses.append(("WHERE", cond))
        else:
            self.where_clauses.append(("AND", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def And(self, cond):
        self.where_clauses.append(("AND", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def Or(self, cond):
        self.where_clauses.append(("OR", cond))
        self._last_column = cond if hasattr(cond, "to_sql") else None
        return self

    def In(self, *values):
        """SQL IN después de Where. Soporta .In([1, 2]) y .In(1, 2)."""
        if not self._last_column:
            raise ValueError("In() debe usarse después de Where(columna)")
        from psqlmodel.orm.column import RawExpression

        # Aplanar valores si se pasa una lista única
        if len(values) == 1 and isinstance(values[0], (list, tuple, set)) and not isinstance(values[0], (str, bytes)):
            actual_values = values[0]
        else:
            actual_values = values

        # Si values es una Query (subconsulta)
        if hasattr(actual_values, "to_sql"):
             in_expr = RawExpression(f"{self._last_column.to_sql()} IN ({actual_values.to_sql()})")
        else:
            vals = ", ".join(repr(v) for v in actual_values)
            in_expr = RawExpression(f"{self._last_column.to_sql()} IN ({vals})")

        if self.where_clauses and self.where_clauses[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where_clauses[-1]
            self.where_clauses[-1] = (op, in_expr)
        return self

    def NotIn(self, *values):
        """SQL NOT IN después de Where. Soporta .NotIn([1, 2]) y .NotIn(1, 2)."""
        if not self._last_column:
            raise ValueError("NotIn() debe usarse después de Where(columna)")
        from psqlmodel.column import RawExpression

        # Aplanar valores si se pasa una lista única
        if len(values) == 1 and isinstance(values[0], (list, tuple, set)) and not isinstance(values[0], (str, bytes)):
            actual_values = values[0]
        else:
            actual_values = values

        # Si values es una Query (subconsulta)
        if hasattr(actual_values, "to_sql"):
            notin_expr = RawExpression(f"{self._last_column.to_sql()} NOT IN ({actual_values.to_sql()})")
        else:
            vals = ", ".join(repr(v) for v in actual_values)
            notin_expr = RawExpression(f"{self._last_column.to_sql()} NOT IN ({vals})")

        if self.where_clauses and self.where_clauses[-1][0] in ("WHERE", "AND", "OR"):
            op, _ = self.where_clauses[-1]
            self.where_clauses[-1] = (op, notin_expr)
        return self


    def Or(self, cond):
        self.where_clauses.append(("OR", cond))
        return self

    def Returning(self, *cols):
        """Add RETURNING clause to get deleted rows back."""
        self._returning = cols if cols else None
        return self

    def to_sql_params(self):
        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")

        sql = [f"DELETE FROM {schema}.{table}"]
        params = []

        # WHERE
        if self.where_clauses:
            where_parts = []
            for op, cond in self.where_clauses:
                if hasattr(cond, "to_sql_params"):
                    cond_sql, cond_params = cond.to_sql_params()
                else:
                    cond_sql, cond_params = str(cond), []
                where_parts.append(f"{op} {cond_sql}")
                params.extend(cond_params)
            sql.append(" ".join(where_parts))

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.to_sql() if hasattr(c, "to_sql") else str(c) for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def Delete(model):
    """Create a DELETE query for the given model."""
    return DeleteQuery(model)


# ============================================================
# UPDATE (Bulk/WHERE-based)
# ============================================================

class UpdateQuery(Query):
    """UPDATE query builder with SET and WHERE support.

    Usage:
        Update(User).Set(name="NewName").Where(User.id == 5).execute(engine)
        Update(User).Set(is_active=False).Where(User.last_login < cutoff).execute(engine)
    """

    def __init__(self, model):
        self.model = model
        self.set_clauses = []  # list of (column, value)
        self.where_clauses = []
        self._returning = None

    def Set(self, **kwargs):
        """
        Set columns to values using keyword arguments only.
        Example: .Set(email="foo", name="bar")
        This enables auto-completion of model attributes in most editors.
        """
        columns = getattr(self.model, "__columns__", {})
        for col_name, value in kwargs.items():
            if col_name in columns:
                col = columns[col_name]
                self.set_clauses.append((col, value))
            else:
                raise ValueError(f"Column '{col_name}' not found in model {self.model}")
        return self

    def SetMany(self, **kwargs):
        """Set multiple columns at once using keyword arguments.

        Usage: Update(User).SetMany(name="Alice", age=30).Where(...)
        """
        columns = getattr(self.model, "__columns__", {})
        for col_name, value in kwargs.items():
            if col_name in columns:
                col = columns[col_name]
                self.set_clauses.append((col, value))
        return self

    def Where(self, cond):
        if not self.where_clauses:
            self.where_clauses.append(("WHERE", cond))
        else:
            self.where_clauses.append(("AND", cond))
        return self

    def And(self, cond):
        self.where_clauses.append(("AND", cond))
        return self

    def Or(self, cond):
        self.where_clauses.append(("OR", cond))
        return self

    def Returning(self, *cols):
        """Add RETURNING clause to get updated rows back."""
        self._returning = cols if cols else None
        return self

    def to_sql_params(self):
        if not self.set_clauses:
            raise ValueError("UPDATE query requires at least one SET clause")

        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")

        sql = [f"UPDATE {schema}.{table}"]
        params = []

        # SET - support expressions (functions, subqueries, arithmetic, etc.)
        set_parts = []
        for col, value in self.set_clauses:
            col_name = col.name if hasattr(col, "name") else str(col)
            if hasattr(value, "to_sql_params"):
                val_sql, val_params = value.to_sql_params()
                set_parts.append(f"{col_name} = {val_sql}")
                params.extend(val_params)
            elif hasattr(value, "to_sql"):
                set_parts.append(f"{col_name} = {value.to_sql()}")
            else:
                set_parts.append(f"{col_name} = %s")
                params.append(value)
        sql.append("SET " + ", ".join(set_parts))

        # WHERE
        if self.where_clauses:
            where_parts = []
            for op, cond in self.where_clauses:
                if hasattr(cond, "to_sql_params"):
                    cond_sql, cond_params = cond.to_sql_params()
                else:
                    cond_sql, cond_params = str(cond), []
                where_parts.append(f"{op} {cond_sql}")
                params.extend(cond_params)
            sql.append(" ".join(where_parts))

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.to_sql() if hasattr(c, "to_sql") else str(c)
                for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def Update(model):
    """Create an UPDATE query for the given model."""
    return UpdateQuery(model)


# ============================================================
# INSERT SELECT PROXY (fluent API helper)
# ============================================================

class InsertSelectProxy:
    """Proxy that allows chaining SelectQuery methods while maintaining InsertQuery context.
    
    Example:
        Insert(Archive)
            .Select(User.id, User.name)
            .Where(User.active == False)
            .And(NotExists(...))
            .Returning(Archive.id)
    """
    def __init__(self, insert_query, select_query):
        self.insert_query = insert_query
        self.select_query = select_query
    
    # Proxy SelectQuery methods
    def Where(self, cond):
        self.select_query.Where(cond)
        return self
    
    def And(self, cond):
        self.select_query.And(cond)
        return self
    
    def Or(self, cond):
        self.select_query.Or(cond)
        return self
    
    def OrderBy(self, *cols):
        self.select_query.OrderBy(*cols)
        return self
    
    def Limit(self, n):
        self.select_query.Limit(n)
        return self
    
    def Offset(self, n):
        self.select_query.Offset(n)
        return self
    
    def In(self, values):
        self.select_query.In(values)
        return self
    
    def NotIn(self, values):
        self.select_query.NotIn(values)
        return self
    
    # InsertQuery methods remain available
    def Returning(self, *cols):
        self.insert_query.Returning(*cols)
        return self
    
    def OnConflict(self, conflict_column, *, do_update=None, do_nothing=False):
        self.insert_query.OnConflict(conflict_column, do_update=do_update, do_nothing=do_nothing)
        return self
    
    # Execution methods
    def to_sql_params(self):
        return self.insert_query.to_sql_params()
    
    def execute(self, engine, *params):
        return self.insert_query.execute(engine, *params)
    
    async def execute_async(self, engine, *params):
        return await self.insert_query.execute_async(engine, *params)


# ============================================================
# INSERT (with RETURNING and ON CONFLICT / UPSERT)
# ============================================================

class InsertQuery(Query):
    """INSERT query builder with RETURNING and ON CONFLICT (UPSERT) support.

    Usage:
        # Simple insert
        Insert(User).Values(name="Alice", age=30).execute(engine)

        # Insert with RETURNING
        Insert(User).Values(name="Alice", age=30).Returning(User.id).execute(engine)

        # UPSERT (insert or update on conflict)
        Insert(User).Values(email="a@b.com", name="Alice").OnConflict(
            User.email,
            do_update={"name": "Alice Updated"}
        ).execute(engine)
    """

    def __init__(self, model):
        self.model = model
        self.values_dict = {}
        self.select_query = None  # NEW: for INSERT...SELECT
        self._returning = None
        self._on_conflict_column = None
        self._on_conflict_do_update = None
        self._on_conflict_do_nothing = False

    def Values(self, **kwargs):
        """
        Set column values for the insert, validating against model columns.
        Example: Insert(User).Values(name="Alice", email="a@b.com")
        Only model attributes are allowed (auto-completion in IDEs).
        """
        if self.select_query:
            raise ValueError("Cannot use .Values() with .Select() - choose one")
        columns = getattr(self.model, "__columns__", {})
        for col_name, value in kwargs.items():
            if col_name in columns:
                self.values_dict[col_name] = value
            else:
                raise ValueError(f"Column '{col_name}' not found in model {self.model}")
        return self

    def Select(self, *columns):
        """
        Use INSERT...SELECT instead of INSERT...VALUES.
        
        Usage:
            Insert(Archive)
                .Select(User.id, User.name)
                .Where(User.active == False)
                .Returning(Archive.id)
        
        Returns: SelectQuery that can be chained with .Where(), .And(), etc.
        """
        if self.values_dict:
            raise ValueError("Cannot use .Select() with .Values() - choose one")
        
        # Create a SelectQuery with these columns
        self.select_query = SelectQuery(*columns)
        # Return self but with SelectQuery methods available via proxy
        return InsertSelectProxy(self, self.select_query)

    def Returning(self, *cols):
        """Add RETURNING clause to get inserted row(s) back."""
        self._returning = cols if cols else None
        return self

    def OnConflict(self, conflict_column, *, do_update=None, do_nothing=False):
        """Handle conflicts (UPSERT).

        Args:
            conflict_column: The column that defines the conflict (usually unique/pk)
            do_update: Dict of {column_name: new_value} to update on conflict
            do_nothing: If True, do nothing on conflict (ignore)
        """
        self._on_conflict_column = conflict_column
        self._on_conflict_do_update = do_update
        self._on_conflict_do_nothing = do_nothing
        return self

    def to_sql_params(self):
        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")

        # INSERT...SELECT path
        if self.select_query:
            # Extract column names from SelectQuery columns
            col_names = []
            for c in self.select_query.columns:
                if hasattr(c, "name"):
                    col_names.append(c.name)
                elif hasattr(c, "expr") and hasattr(c.expr, "name"):  # Alias
                    col_names.append(c.expr.name)
                else:
                    # For RawExpression or unknown, skip or use a generic name
                    # This might need refinement based on actual usage
                    pass
            
            # Get SELECT SQL
            select_sql, params = self.select_query.to_sql_params()
            
            sql = [f"INSERT INTO {schema}.{table}"]
            if col_names:
                sql.append(f"({', '.join(col_names)})")
            sql.append(select_sql)
            
            # ON CONFLICT (same logic as VALUES)
            if self._on_conflict_column is not None:
                conflict_col_name = (
                    self._on_conflict_column.name
                    if hasattr(self._on_conflict_column, "name")
                    else str(self._on_conflict_column)
                )
                if self._on_conflict_do_nothing:
                    sql.append(f"ON CONFLICT ({conflict_col_name}) DO NOTHING")
                elif self._on_conflict_do_update:
                    update_parts = []
                    for col_name, value in self._on_conflict_do_update.items():
                        if hasattr(value, "to_sql"):
                            update_parts.append(f"{col_name} = {value.to_sql()}")
                        else:
                            update_parts.append(f"{col_name} = %s")
                            params.append(value)
                    sql.append(
                        f"ON CONFLICT ({conflict_col_name}) DO UPDATE SET "
                        + ", ".join(update_parts)
                    )
            
            # RETURNING
            if self._returning:
                returning_cols = ", ".join(
                    c.name if hasattr(c, "name") else str(c) for c in self._returning
                )
                sql.append(f"RETURNING {returning_cols}")
            
            return "\n".join(sql), params
        
        # INSERT...VALUES path (original logic)
        if not self.values_dict:
            raise ValueError("INSERT query requires at least one value or SELECT")

        col_names = list(self.values_dict.keys())
        placeholders = []
        params = []

        for col_name in col_names:
            value = self.values_dict[col_name]
            # Handle expressions (functions, subqueries, etc.)
            if hasattr(value, "to_sql_params"):
                val_sql, val_params = value.to_sql_params()
                placeholders.append(val_sql)
                params.extend(val_params)
            elif hasattr(value, "to_sql"):
                placeholders.append(value.to_sql())
            else:
                placeholders.append("%s")
                params.append(value)

        sql = [f"INSERT INTO {schema}.{table} ({', '.join(col_names)})"]
        sql.append(f"VALUES ({', '.join(placeholders)})")

        # ON CONFLICT
        if self._on_conflict_column is not None:
            conflict_col_name = (
                self._on_conflict_column.name
                if hasattr(self._on_conflict_column, "name")
                else str(self._on_conflict_column)
            )

            if self._on_conflict_do_nothing:
                sql.append(f"ON CONFLICT ({conflict_col_name}) DO NOTHING")
            elif self._on_conflict_do_update:
                update_parts = []
                for col_name, value in self._on_conflict_do_update.items():
                    # Support EXCLUDED references and expressions
                    if hasattr(value, "to_sql_params"):
                        val_sql, val_params = value.to_sql_params()
                        update_parts.append(f"{col_name} = {val_sql}")
                        params.extend(val_params)
                    elif hasattr(value, "to_sql"):
                        update_parts.append(f"{col_name} = {value.to_sql()}")
                    else:
                        update_parts.append(f"{col_name} = %s")
                        params.append(value)
                sql.append(
                    f"ON CONFLICT ({conflict_col_name}) DO UPDATE SET "
                    + ", ".join(update_parts)
                )

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.name if hasattr(c, "name") else str(c) for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def Insert(model):
    """Create an INSERT query for the given model."""
    return InsertQuery(model)


# ============================================================
# BULK OPERATIONS
# ============================================================

class BulkInsertQuery(Query):
    """Bulk INSERT query for inserting multiple rows at once.

    Usage:
        BulkInsert(User, [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]).execute(engine)
    """

    def __init__(self, model, rows: list):
        self.model = model
        self.rows = rows
        self._returning = None
        self._on_conflict_column = None
        self._on_conflict_do_update = None
        self._on_conflict_do_nothing = False

    def Returning(self, *cols):
        self._returning = cols if cols else None
        return self

    def OnConflict(self, conflict_column, *, do_update=None, do_nothing=False):
        self._on_conflict_column = conflict_column
        self._on_conflict_do_update = do_update
        self._on_conflict_do_nothing = do_nothing
        return self

    def to_sql_params(self):
        if not self.rows:
            raise ValueError("BulkInsert requires at least one row")

        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")

        # Get column names from first row
        col_names = list(self.rows[0].keys())

        # Build VALUES clause for all rows
        values_clauses = []
        params = []
        for row in self.rows:
            placeholders = ["%s"] * len(col_names)
            values_clauses.append(f"({', '.join(placeholders)})")
            for col_name in col_names:
                params.append(row.get(col_name))

        sql = [f"INSERT INTO {schema}.{table} ({', '.join(col_names)})"]
        sql.append("VALUES " + ", ".join(values_clauses))

        # ON CONFLICT
        if self._on_conflict_column is not None:
            conflict_col_name = (
                self._on_conflict_column.name
                if hasattr(self._on_conflict_column, "name")
                else str(self._on_conflict_column)
            )

            if self._on_conflict_do_nothing:
                sql.append(f"ON CONFLICT ({conflict_col_name}) DO NOTHING")
            elif self._on_conflict_do_update:
                update_parts = []
                for col_name, value in self._on_conflict_do_update.items():
                    update_parts.append(f"{col_name} = %s")
                    params.append(value)
                sql.append(
                    f"ON CONFLICT ({conflict_col_name}) DO UPDATE SET "
                    + ", ".join(update_parts)
                )

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.name if hasattr(c, "name") else str(c) for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def BulkInsert(model, rows: list):
    """Create a bulk INSERT query for the given model and rows."""
    return BulkInsertQuery(model, rows)


class BulkUpdateQuery(Query):
    """Bulk UPDATE query using CASE WHEN for updating multiple rows efficiently.

    Usage:
        # Update specific rows by primary key
        BulkUpdate(User, User.id, [
            {"id": 1, "name": "Alice Updated"},
            {"id": 2, "name": "Bob Updated"},
        ]).execute(engine)
    """

    def __init__(self, model, pk_column, rows: list):
        self.model = model
        self.pk_column = pk_column
        self.rows = rows
        self._returning = None

    def Returning(self, *cols):
        self._returning = cols if cols else None
        return self

    def to_sql_params(self):
        if not self.rows:
            raise ValueError("BulkUpdate requires at least one row")

        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")
        pk_name = (
            self.pk_column.name if hasattr(self.pk_column, "name") else str(self.pk_column)
        )

        # Get all column names to update (excluding PK)
        update_cols = set()
        pk_values = []
        for row in self.rows:
            pk_values.append(row.get(pk_name))
            for key in row.keys():
                if key != pk_name:
                    update_cols.add(key)

        # Build CASE WHEN statements for each column
        set_parts = []
        params = []

        for col_name in update_cols:
            case_parts = []
            for row in self.rows:
                if col_name in row:
                    case_parts.append(f"WHEN {pk_name} = %s THEN %s")
                    params.append(row.get(pk_name))
                    params.append(row.get(col_name))
            if case_parts:
                set_parts.append(
                    f"{col_name} = CASE {' '.join(case_parts)} ELSE {col_name} END"
                )

        if not set_parts:
            raise ValueError(
                "BulkUpdate requires at least one column to update (besides PK)"
            )

        sql = [f"UPDATE {schema}.{table}"]
        sql.append("SET " + ", ".join(set_parts))

        # WHERE pk IN (...)
        in_placeholders = ", ".join(["%s"] * len(pk_values))
        sql.append(f"WHERE {pk_name} IN ({in_placeholders})")
        params.extend(pk_values)

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.name if hasattr(c, "name") else str(c) for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def BulkUpdate(model, pk_column, rows: list):
    """Create a bulk UPDATE query for the given model using CASE WHEN."""
    return BulkUpdateQuery(model, pk_column, rows)


class BulkDeleteQuery(Query):
    """Bulk DELETE query for deleting multiple rows by primary key.

    Usage:
        BulkDelete(User, User.id, [1, 2, 3]).execute(engine)
    """

    def __init__(self, model, pk_column, pk_values: list):
        self.model = model
        self.pk_column = pk_column
        self.pk_values = pk_values
        self._returning = None

    def Returning(self, *cols):
        self._returning = cols if cols else None
        return self

    def to_sql_params(self):
        if not self.pk_values:
            raise ValueError("BulkDelete requires at least one primary key value")

        schema = getattr(self.model, "__schema__", "public") or "public"
        table = getattr(self.model, "__tablename__")
        pk_name = (
            self.pk_column.name if hasattr(self.pk_column, "name") else str(self.pk_column)
        )

        in_placeholders = ", ".join(["%s"] * len(self.pk_values))
        sql = [f"DELETE FROM {schema}.{table}"]
        sql.append(f"WHERE {pk_name} IN ({in_placeholders})")
        params = list(self.pk_values)

        # RETURNING
        if self._returning:
            returning_cols = ", ".join(
                c.name if hasattr(c, "name") else str(c) for c in self._returning
            )
            sql.append(f"RETURNING {returning_cols}")

        return "\n".join(sql), params


def BulkDelete(model, pk_column, pk_values: list):
    """Create a bulk DELETE query for the given model by primary key values."""
    return BulkDeleteQuery(model, pk_column, pk_values)


# ============================================================
# BULK OPERATIONS
# ============================================================

class CTEQuery(Query):
    """CTE (WITH clause) query builder.

    Usage:
        # Simple CTE
        With("active_users", Select(User.id, User.name).Where(User.active == True)) \
            .Then(Select(RawExpression("*")).From("active_users")) \
            .execute(engine)

        # Multiple CTEs
        With("rich_users", Select(User.id).Where(User.balance > 1000)) \
            .With("active_users", Select(User.id).Where(User.active == True)) \
            .Then(
                InsertFromSelect(
                    Transactions, ["user_id", "type"],
                    Select(RawExpression("id"), RawExpression("'bonus'")).From("rich_users")
                )
            ).execute(engine)
    """

    def __init__(self, name: str, query):
        """
        Args:
            name: CTE alias name
            query: The query for this CTE
        """
        self.ctes = [(name, query)]
        self.final_query = None

    def With(self, name: str, query):
        """Add another CTE to the WITH clause."""
        self.ctes.append((name, query))
        return self

    def Then(self, query):
        """Set the final query that uses the CTEs.
        
        Smart FROM inference: If the final query is a SelectQuery with exactly
        one CTE and no explicit .From(), automatically set FROM to the CTE name.
        """
        self.final_query = query
        
        # Smart FROM inference for SelectQuery
        if isinstance(query, SelectQuery):
            # Auto-infer FROM only if:
            # 1. Exactly one CTE registered
            # 2. No explicit FROM clause set
            if len(self.ctes) == 1 and query._from is None:
                query._from = self.ctes[0][0]  # Set FROM to CTE name
        
        return self

    def to_sql_params(self):
        if not self.final_query:
            raise ValueError("CTE requires a final query via .Then()")

        params = []
        cte_parts = []

        for name, cte_query in self.ctes:
            cte_sql, cte_params = cte_query.to_sql_params()
            cte_parts.append(f"{name} AS (\n{cte_sql}\n)")
            params.extend(cte_params)

        final_sql, final_params = self.final_query.to_sql_params()
        params.extend(final_params)

        sql = "WITH " + ",\n".join(cte_parts) + "\n" + final_sql
        return sql, params


def With(name: str, query):
    """Create a CTE (WITH clause) query."""
    return CTEQuery(name, query)
