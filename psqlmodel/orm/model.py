from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, TYPE_CHECKING

from .column import Column
from ..query.crud import DirtyTrackingMixin

if TYPE_CHECKING:
    from ..core.session import Session, AsyncSession


class PSQLModel(DirtyTrackingMixin):
    """
    Modelo base ligero para el ORM.

    Requisitos para que el Engine y otros módulos lo detecten correctamente:
    - __tablename__:   nombre de la tabla
    - __schema__:      nombre del schema (opcional, por defecto 'public')
    - __columns__:     dict[str, Column] con el mapeo de columnas
    - __unique_together__: lista opcional para constraints de unicidad
    """

    __tablename__: ClassVar[str]
    __schema__: ClassVar[Optional[str]]
    __columns__: ClassVar[Dict[str, Column]]
    __unique_together__: ClassVar[List[str]]

    # ============================================================
    # Hook de clase – añade anotaciones dinámicas para autocompletado
    # ============================================================
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Hook llamado automáticamente al declarar una subclase.

        - Asegura que todas las columnas en __columns__ tengan una anotación
          de tipo en __annotations__ (respetando las ya definidas por el usuario).
        - Esto mejora el autocompletado en el IDE y no afecta en runtime.
        """
        super().__init_subclass__(**kwargs)

        cols: Dict[str, Column] = getattr(cls, "__columns__", {}) or {}
        if not cols:
            return

        # Garantizar que __annotations__ exista
        if not hasattr(cls, "__annotations__"):
            cls.__annotations__ = {}

        annotations = cls.__annotations__

        for col_name, col in cols.items():
            # Si ya hay anotación explícita, se respeta
            if col_name in annotations:
                continue

            # Inferir tipo desde Column.type_hint si existe
            python_type: Any = getattr(col, "type_hint", Any)

            # Si la columna es nullable, el tipo pasa a Optional
            if getattr(col, "nullable", False):
                python_type = Optional[python_type]

            annotations[col_name] = python_type

    # ============================================================
    # Inicialización de instancia
    # ============================================================
    def __init__(self, **kwargs: Any) -> None:
        """
        Inicializa una instancia del modelo a partir de kwargs.

        Reglas:
        - Solo acepta columnas definidas en `__columns__`.
        - Aplica `default` si existe en Column.
        - Valida `nullable` y lanza ValueError si falta un valor no-nullable
          (excepto para primary_key, que puede omitirse para autogenerarse).
        """
        # Inicializar mixin de tracking (no pasamos kwargs para no romper su API)
        try:
            super().__init__()  # DirtyTrackingMixin.__init__
        except TypeError:
            # Por si el mixin no define __init__ o espera otros args
            try:
                super().__init__  # acceso para mypy/linters
            except Exception:
                pass

        cls = self.__class__
        cols: Dict[str, Column] = getattr(cls, "__columns__", {}) or {}

        # Detectar claves inválidas
        unknown_keys = set(kwargs.keys()) - set(cols.keys())
        if unknown_keys:
            bad = ", ".join(sorted(unknown_keys))
            raise TypeError(f"Unknown fields for {cls.__name__}: {bad}")

        for name, col in cols.items():
            if name in kwargs:
                value = kwargs[name]
            else:
                default = getattr(col, "default", None)
                if default is not None:
                    value = default() if callable(default) else default
                else:
                    value = None

            # Validar no-null (excepto primary key)
            if (
                value is None
                and not getattr(col, "nullable", False)
                and not getattr(col, "primary_key", False)
            ):
                raise ValueError(
                    f"Field '{name}' is not nullable and no value provided"
                )

            # Usamos setattr para que DirtyTrackingMixin pueda interceptar
            setattr(self, name, value)

    # ============================================================
    # Serialización
    # ============================================================
    def to_dict(self) -> Dict[str, Any]:
        """Serializa la instancia a dict usando únicamente las columnas definidas."""
        cols = getattr(self.__class__, "__columns__", {}) or {}
        return {name: getattr(self, name, None) for name in cols.keys()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PSQLModel":
        """
        Crea una instancia desde un dict filtrando solo las columnas válidas.

        No hace validación adicional; delega en __init__.
        """
        cols = getattr(cls, "__columns__", {}) or {}
        filtered = {k: data[k] for k in cols.keys() if k in data}
        return cls(**filtered)

    def update_from_dict(
        self,
        data: Dict[str, Any],
        *,
        ignore_unknown: bool = True,
        partial: bool = True,
    ) -> "PSQLModel":
        """
        Actualiza campos desde un dict respetando el tracking de DirtyTrackingMixin.

        - Usa `setattr`, por lo que cualquier lógica de tracking se dispara normalmente.
        - Si `ignore_unknown=False`, lanza TypeError ante claves no mapeadas a columnas.
        - Si `partial=False`, exige que todos los campos no-nullables (no PK) estén presentes
          en `data`, y si no, lanza ValueError.
        """
        cols = self.__class__.columns()

        # Validar unknowns
        if not ignore_unknown:
            unknown = set(data.keys()) - set(cols.keys())
            if unknown:
                raise TypeError(
                    f"Unknown fields for {self.__class__.__name__}: {', '.join(sorted(unknown))}"
                )

        # Asignar solo columnas conocidas
        for name, value in data.items():
            if name in cols:
                setattr(self, name, value)

        if not partial:
            for name, col in cols.items():
                if getattr(col, "primary_key", False):
                    continue
                if getattr(col, "nullable", False):
                    continue
                if name not in data:
                    raise ValueError(
                        f"Field '{name}' is required (non-nullable) but not present in update_from_dict() data"
                    )

        return self

    # ============================================================
    # Validación lógica (runtime)
    # ============================================================
    def validate(self) -> None:
        """
        Valida los valores actuales de la instancia contra las restricciones básicas
        de Column (nullable / primary_key).

        No toca constraints complejos ni tipos; eso se delega al motor de BD.
        """
        cols = getattr(self.__class__, "__columns__", {}) or {}
        for name, col in cols.items():
            val = getattr(self, name, None)
            if (
                val is None
                and not getattr(col, "nullable", False)
                and not getattr(col, "primary_key", False)
            ):
                raise ValueError(
                    f"Field '{name}' is not nullable and has no value"
                )

    # ============================================================
    # Representación y acceso dinámico
    # ============================================================
    def __repr__(self) -> str:
        data = self.to_dict()
        items = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"{self.__class__.__name__}({items})"

    def __getattr__(self, name: str) -> Any:
        """
        Permite acceso a columnas no inicializadas sin lanzar AttributeError.

        - Si `name` es una columna declarada en __columns__, devuelve None.
        - Si no es columna, lanza AttributeError normal.

        Esto mejora el autocompletado del IDE y evita errores molestos cuando
        se accede a una columna antes de que la instancia haya sido poblada
        completamente (por ejemplo, al crearla a partir de un SELECT parcial).
        """
        cols = getattr(self.__class__, "__columns__", {}) or {}
        if name in cols:
            return None
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    # ============================================================
    # Helpers de clase
    # ============================================================
    @classmethod
    def columns(cls) -> Dict[str, Column]:
        """Devuelve el dict de columnas del modelo (puede ser vacío)."""
        return getattr(cls, "__columns__", {}) or {}

    @classmethod
    def column_names(cls) -> List[str]:
        """Devuelve la lista de nombres de columnas declaradas."""
        return list(cls.columns().keys())

    @classmethod
    def primary_key_column(cls) -> Optional[str]:
        """Devuelve el nombre de la columna primary_key, o None si no existe."""
        for name, col in cls.columns().items():
            if getattr(col, "primary_key", False):
                return name
        return None

    # ============================================================
    # Integración con Session / AsyncSession
    # ============================================================
    def _get_bound_session_sync(self) -> Optional["Session"]:
        """
        Intenta resolver una Session sync asociada:

        1. A través de self.__session__ (Session.add la setea).
        2. A través de SessionManager.current() si estamos dentro de un contexto.
        """
        # 1) __session__ explícita
        session = getattr(self, "__session__", None)
        engine = getattr(session, "engine", None)
        config = getattr(engine, "config", None)
        if session is not None and config is not None and not getattr(config, "async_", False):
            return session  # type: ignore[return-value]

        # 2) ContextVar global
        try:
            from ..core.session import SessionManager
            current = SessionManager.current()
        except Exception:
            current = None

        engine = getattr(current, "engine", None)
        config = getattr(engine, "config", None)
        if current is not None and config is not None and not getattr(config, "async_", False):
            return current  # type: ignore[return-value]
        return None

    def _get_bound_session_async(self) -> Optional["AsyncSession"]:
        """
        Intenta resolver una AsyncSession asociada:

        1. A través de self.__session__.
        2. A través de SessionManager.current_async().
        """
        session = getattr(self, "__session__", None)
        engine = getattr(session, "engine", None)
        config = getattr(engine, "config", None)
        if session is not None and config is not None and getattr(config, "async_", False):
            return session  # type: ignore[return-value]

        try:
            from ..core.session import SessionManager
            current = SessionManager.current_async()
        except Exception:
            current = None

        engine = getattr(current, "engine", None)
        config = getattr(engine, "config", None)
        if current is not None and config is not None and getattr(config, "async_", False):
            return current  # type: ignore[return-value]
        return None

    def _get_dirty_field_names(self) -> List[str]:
        """
        Intenta obtener los nombres de campos sucios desde DirtyTrackingMixin.

        Soporta varias formas:
        - método get_dirty_fields() → iterable de nombres
        - atributo _dirty_fields → iterable de nombres
        - dict {nombre: bool} → claves con True
        Si no se puede determinar, devuelve [] (el caller decide el fallback).
        """
        cols = self.__class__.columns()
        col_names = set(cols.keys())

        names: List[str] = []

        getter = getattr(self, "get_dirty_fields", None)
        dirty_source: Any = None

        if callable(getter):
            try:
                dirty_source = getter()  # type: ignore[misc]
            except TypeError:
                try:
                    dirty_source = getter(self)  # type: ignore[misc]
                except Exception:
                    dirty_source = None
        else:
            dirty_source = getattr(self, "_dirty_fields", None)

        if isinstance(dirty_source, dict):
            names = [k for k, v in dirty_source.items() if v]
        elif isinstance(dirty_source, (list, tuple, set)):
            names = list(dirty_source)
        elif isinstance(dirty_source, str):
            names = [dirty_source]

        return [n for n in names if n in col_names]

    def _mark_clean_if_possible(self) -> None:
        """Intenta marcar la instancia como 'clean' después de un save/delete directo."""
        cleaner = getattr(self, "mark_clean", None)
        if callable(cleaner):
            try:
                cleaner()  # type: ignore[misc]
            except TypeError:
                try:
                    cleaner(self)  # type: ignore[misc]
                except Exception:
                    pass

    # ============================================================
    # Persistencia síncrona (psycopg) + integración Session
    # ============================================================
    def save(self, conn=None, dsn: Optional[str] = None) -> None:
        """
        Inserta o actualiza la fila correspondiente en la base de datos (modo sync).

        Prioridades:
        1. Si hay Session activa (self.__session__ o SessionManager.current()):
           - Se delega en la Session/Transaction:
               session.add(self)
               session.flush()
           - La TX permanece abierta (commit/rollback lo maneja el caller).
        2. Si no hay Session, usa psycopg con DSN/conn como antes.

        Si pasas `conn`, puede ser:
            - una conexión psycopg (conn.cursor() disponible)
            - un cursor psycopg directamente
        - Si `conn` es None, se abrirá una conexión temporal usando `dsn`.
        - Para UPSERT:
            - Siempre inserta todas las columnas.
            - Pero la parte de UPDATE en ON CONFLICT usa solo campos sucios
              (si DirtyTrackingMixin los reporta; si no, usa todas).
        """
        # 1) Intentar usar Session/Transaction para respetar pipeline y tracking
        session = self._get_bound_session_sync()
        if session is not None:
            session.add(self)
            # save() se interpreta como "persistir ahora dentro de la TX actual"
            session.flush()
            return

        # 2) Fallback: psycopg directo como en tu versión original
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - import error runtime
            raise RuntimeError("psycopg is required for sync save") from exc

        close_conn = False
        cursor = None

        if conn is None:
            if not dsn:
                raise ValueError("Provide a psycopg connection or a dsn")
            conn = psycopg.connect(dsn)
            close_conn = True

        # Aceptar tanto conexión como cursor
        cursor = conn.cursor() if hasattr(conn, "cursor") else conn

        cls = self.__class__
        table = cls.__tablename__
        cols = cls.columns()
        pk = cls.primary_key_column()
        data = self.to_dict()

        column_names = list(cols.keys())
        values = [data.get(c) for c in column_names]

        placeholders = ",".join(["%s"] * len(values)) if values else ""
        cols_sql = ",".join(column_names)

        # Campos sucios (si los hay)
        dirty_fields = self._get_dirty_field_names()

        if pk:
            # Construir SET para update (excluyendo la PK) usando solo campos sucios si se conocen
            if dirty_fields:
                set_exprs = [
                    f"{c}=EXCLUDED.{c}"
                    for c in column_names
                    if c != pk and c in dirty_fields
                ]
                # Si por alguna razón no queda nada, caemos a todas excepto PK
                if not set_exprs:
                    set_exprs = [f"{c}=EXCLUDED.{c}" for c in column_names if c != pk]
            else:
                # Sin tracking disponible → usar todas excepto PK
                set_exprs = [f"{c}=EXCLUDED.{c}" for c in column_names if c != pk]

            set_sql = ",".join(set_exprs) if set_exprs else None

            sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"
            if set_sql:
                sql += f" ON CONFLICT ({pk}) DO UPDATE SET {set_sql} RETURNING {pk}"
            else:
                sql += f" ON CONFLICT ({pk}) DO NOTHING RETURNING {pk}"
        else:
            sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"

        cursor.execute(sql, values)

        # Intentar recuperar PK devuelta
        try:
            row = cursor.fetchone()
            if row and pk:
                setattr(self, pk, row[0])
        except Exception:
            # Sin RETURNING o sin fila
            pass

        if close_conn:
            conn.commit()
            conn.close()

        # Marcar como "clean" después del save directo
        self._mark_clean_if_possible()

    def delete(self, conn=None, dsn: Optional[str] = None) -> None:
        """
        Elimina la fila correspondiente por primary key (modo sync, psycopg).

        Prioridades:
        1. Si hay Session activa → delega en Session.delete(self).
        2. Si no hay Session → ejecuta DELETE directo con psycopg.

        Requisitos:
        - El modelo debe definir una primary key.
        - La instancia debe tener un valor para esa PK.
        """
        # 1) Usar Session si está disponible
        session = self._get_bound_session_sync()
        if session is not None:
            session.delete(self)
            return

        # 2) Fallback psycopg
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - import error runtime
            raise RuntimeError("psycopg is required for sync delete") from exc

        close_conn = False

        if conn is None:
            if not dsn:
                raise ValueError("Provide a psycopg connection or a dsn")
            conn = psycopg.connect(dsn)
            close_conn = True

        cursor = conn.cursor() if hasattr(conn, "cursor") else conn

        cls = self.__class__
        pk = cls.primary_key_column()
        if not pk:
            raise ValueError("No primary key defined for model; cannot delete")

        pk_val = getattr(self, pk, None)
        if pk_val is None:
            raise ValueError(
                "Primary key value is not set on the instance; cannot delete"
            )

        sql = f"DELETE FROM {cls.__tablename__} WHERE {pk} = %s"
        cursor.execute(sql, (pk_val,))

        if close_conn:
            conn.commit()
            conn.close()

        self._mark_clean_if_possible()

    # ============================================================
    # Persistencia asíncrona (asyncpg) + integración AsyncSession
    # ============================================================
    async def save_async(self, conn=None, dsn: Optional[str] = None) -> None:
        """
        Inserta o actualiza la fila correspondiente en la base de datos (modo async).

        Prioridades:
        1. Si hay AsyncSession activa → usa Transaction async:
               session.add(self)
               await session.flush()
        2. Si no hay AsyncSession → usa asyncpg con DSN/conn, como antes.

        - Usa UPSERT basado en la primary key si existe.
        - La parte de UPDATE de ON CONFLICT usa solo campos sucios
          si DirtyTrackingMixin los reporta.
        """
        # 1) Intentar usar AsyncSession/Transaction
        session = self._get_bound_session_async()
        if session is not None:
            session.add(self)
            await session.flush()
            return

        # 2) Fallback asyncpg
        try:
            import asyncpg  # type: ignore
        except Exception as exc:  # pragma: no cover - import error runtime
            raise RuntimeError("asyncpg is required for async save") from exc

        close_conn = False
        if conn is None:
            if not dsn:
                raise ValueError("Provide an asyncpg connection or a dsn")
            conn = await asyncpg.connect(dsn)
            close_conn = True

        cls = self.__class__
        table = cls.__tablename__
        cols = cls.columns()
        pk = cls.primary_key_column()
        data = self.to_dict()

        column_names = list(cols.keys())
        values = [data.get(c) for c in column_names]

        # asyncpg usa $1, $2, ... como placeholders
        placeholders = (
            ",".join(f"${i + 1}" for i in range(len(values))) if values else ""
        )
        cols_sql = ",".join(column_names)

        dirty_fields = self._get_dirty_field_names()

        if pk:
            if dirty_fields:
                set_exprs = [
                    f"{c}=EXCLUDED.{c}"
                    for c in column_names
                    if c != pk and c in dirty_fields
                ]
                if not set_exprs:
                    set_exprs = [f"{c}=EXCLUDED.{c}" for c in column_names if c != pk]
            else:
                set_exprs = [f"{c}=EXCLUDED.{c}" for c in column_names if c != pk]

            set_sql = ",".join(set_exprs) if set_exprs else None

            sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"
            if set_sql:
                sql += f" ON CONFLICT ({pk}) DO UPDATE SET {set_sql} RETURNING {pk}"
            else:
                sql += f" ON CONFLICT ({pk}) DO NOTHING RETURNING {pk}"
        else:
            sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"

        row = await conn.fetchrow(sql, *values)
        if row and pk:
            # asyncpg permite acceder por nombre de columna
            setattr(self, pk, row[pk])

        if close_conn:
            await conn.close()

        self._mark_clean_if_possible()

    async def delete_async(self, conn=None, dsn: Optional[str] = None) -> None:
        """
        Elimina la fila correspondiente por primary key (modo async, asyncpg).

        Prioridades:
        1. Si hay AsyncSession activa → usa AsyncSession.delete(self).
        2. Si no hay AsyncSession → ejecuta DELETE directo con asyncpg.

        Requisitos:
        - El modelo debe definir una primary key.
        - La instancia debe tener un valor para esa PK.
        """
        # 1) Usar AsyncSession si está disponible
        session = self._get_bound_session_async()
        if session is not None:
            await session.delete(self)
            return

        try:
            import asyncpg  # type: ignore
        except Exception as exc:  # pragma: no cover - import error runtime
            raise RuntimeError("asyncpg is required for async delete") from exc

        close_conn = False
        if conn is None:
            if not dsn:
                raise ValueError("Provide an asyncpg connection or a dsn")
            conn = await asyncpg.connect(dsn)
            close_conn = True

        cls = self.__class__
        pk = cls.primary_key_column()
        if not pk:
            raise ValueError("No primary key defined for model; cannot delete")

        pk_val = getattr(self, pk, None)
        if pk_val is None:
            raise ValueError(
                "Primary key value is not set on the instance; cannot delete"
            )

        sql = f"DELETE FROM {cls.__tablename__} WHERE {pk} = $1"
        await conn.execute(sql, pk_val)

        if close_conn:
            await conn.close()

        self._mark_clean_if_possible()
