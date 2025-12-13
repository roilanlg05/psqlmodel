"""Type stubs for PSQLModel - provides IDE autocompletion support."""

from typing import ClassVar, Dict, List, Any, Optional, Type, TypeVar, overload
from psqlmodel.column import Column

_T = TypeVar('_T', bound='PSQLModel')

class PSQLModel:
    __tablename__: ClassVar[str]
    __schema__: ClassVar[Optional[str]]
    __columns__: ClassVar[Dict[str, Column]]
    __unique_together__: ClassVar[List[str]]
    
    def __init__(self, **kwargs: Any) -> None: ...
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls: Type[_T], data: Dict[str, Any]) -> _T: ...
    
    def validate(self) -> None: ...
    
    def __repr__(self) -> str: ...
    
    @classmethod
    def columns(cls) -> Dict[str, Column]: ...
    
    @classmethod
    def column_names(cls) -> List[str]: ...
    
    @classmethod
    def primary_key_column(cls) -> Optional[str]: ...
    
    def save(self, conn: Any = None, dsn: Optional[str] = None) -> None: ...
    
    def delete(self, conn: Any = None, dsn: Optional[str] = None) -> None: ...
    
    async def save_async(self, conn: Any = None, dsn: Optional[str] = None) -> None: ...
    
    async def delete_async(self, conn: Any = None, dsn: Optional[str] = None) -> None: ...
    
    # Métodos heredados de DirtyTrackingMixin
    def mark_as_clean(self) -> None: ...
    def get_dirty_fields(self) -> Dict[str, Any]: ...
    def is_dirty(self) -> bool: ...
    def is_field_dirty(self, field_name: str) -> bool: ...
    
    # Clase estática para init
    @staticmethod
    def init(
        dsn: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: Optional[str] = None,
        async_: bool = False,
        pool_size: int = 20,
        auto_adjust_pool_size: bool = False,
        max_pool_size: Optional[int] = None,
        connection_timeout: Optional[float] = None,
        ensure_database: bool = True,
        ensure_tables: bool = True,
        models_path: Optional[str] = None,
        debug: bool = False,
        health_check_enabled: bool = False,
        health_check_interval: float = 30.0,
        health_check_retries: int = 1,
        health_check_timeout: float = 5.0
    ) -> Any: ...
