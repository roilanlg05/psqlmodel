"""CRUD helpers and dirty tracking integration (versión inicial).

Esta versión define un mixin para PSQLModel que permite marcar instancias
como "dirty" cuando se modifican atributos, y helpers mínimos para
INSERT/UPDATE basados en Column metadata. La integración completa con
Transaction Manager se hará en pasos posteriores.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..orm.column import Column


class DirtyTrackingMixin:
    """Mixin para modelos que soportan seguimiento de cambios (dirty tracking)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self.__original_values: Dict[str, Any] = {
            name: getattr(self, name)
            for name, col in getattr(self, "__columns__", {}).items()
        }
        self.__dirty_fields: Dict[str, Tuple[Any, Any]] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        # Permitir que PSQLModel y otras bases inicialicen primero
        super().__setattr__(name, value)
        columns = getattr(self, "__columns__", {})
        if name in columns:
            orig = getattr(self, "__original_values", {}).get(name, None)
            if orig != value:
                if not hasattr(self, "__dirty_fields"):
                    self.__dirty_fields = {}
                self.__dirty_fields[name] = (orig, value)

    @property
    def dirty_fields(self) -> Dict[str, Tuple[Any, Any]]:
        return getattr(self, "__dirty_fields", {})

    def clear_dirty(self) -> None:
        self.__original_values = {
            name: getattr(self, name)
            for name, col in getattr(self, "__columns__", {}).items()
        }
        self.__dirty_fields = {}


def build_insert_sql(model: Any, style: str = "psycopg") -> Tuple[str, list]:
    """
    Construye la SQL INSERT y la lista de valores para un modelo.

    Args:
        model: Instancia del modelo a insertar.
        style: Estilo de placeholder ("psycopg" usa %s, "asyncpg" usa $1, $2, ...)
    """
    from ..orm.types import serial, bigserial, smallserial
    
    cols: Dict[str, Column] = getattr(model, "__columns__", {})
    table = getattr(model, "__tablename__")
    schema = getattr(model, "__schema__", "public") or "public"

    names = []
    placeholders = []
    values = []
    idx = 1
    for name, col in cols.items():
        val = getattr(model, name)
        # Excluir columnas serial/autoincrement si el valor es None
        type_hint = getattr(col, "type_hint", None)
        is_serial = isinstance(type_hint, (serial, bigserial, smallserial)) if type_hint else False
        if is_serial and val is None:
            continue
        names.append(name)
        if style == "asyncpg":
            placeholders.append(f"${idx}")
            idx += 1
        else:
            placeholders.append("%s")
        values.append(val)

    sql = (
        f"INSERT INTO {schema}.{table} (" + ", ".join(names) + ") VALUES (" + ", ".join(placeholders) + ")"
    )
    return sql, values


def build_update_sql(model: Any, dirty_fields: Dict[str, Tuple[Any, Any]], style: str = "psycopg") -> Tuple[str, list]:
    """
    Construye la SQL UPDATE basándose únicamente en los dirty_fields.

    Args:
        model: Instancia del modelo a actualizar.
        dirty_fields: dict {name: (old_val, new_val)} de campos cambiados.
        style: Estilo de placeholder ("psycopg" usa %s, "asyncpg" usa $1, $2, ...)
    """
    cols: Dict[str, Column] = getattr(model, "__columns__", {})
    table = getattr(model, "__tablename__")
    schema = getattr(model, "__schema__", "public") or "public"

    pk_col = None
    pk_name = None
    for name, col in cols.items():
        if getattr(col, "primary_key", False):
            pk_col = col
            pk_name = name
            break

    if pk_col is None or pk_name is None:
        raise ValueError("No primary key column defined on model")

    set_parts = []
    values = []
    idx = 1
    for name, (_old, new) in dirty_fields.items():
        if name == pk_name:
            continue
        if style == "asyncpg":
            set_parts.append(f"{name} = ${idx}")
            idx += 1
        else:
            set_parts.append(f"{name} = %s")
        values.append(new)

    if not set_parts:
        return "", []

    values.append(getattr(model, pk_name))
    if style == "asyncpg":
        sql = f"UPDATE {schema}.{table} SET " + ", ".join(set_parts) + f" WHERE {pk_name} = ${idx}"
    else:
        sql = f"UPDATE {schema}.{table} SET " + ", ".join(set_parts) + f" WHERE {pk_name} = %s"
    return sql, values
