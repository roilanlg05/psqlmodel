# Engine Design

Documento de diseño del motor (Engine) del ORM. Este archivo es la referencia viva para la arquitectura del Engine, su API pública y su relación con el QueryBuilder, los modelos (`@table`), el CRUD y los Triggers.

---

## 1. Objetivos Generales

- Punto de entrada unificado `create_engine(...)`.
- Soporte para Postgres **sincrono** y **asíncrono**.
- Pool de conexiones configurable, con opción de **auto-ajuste** dinámico.
- Manejo de **transacciones** a nivel de Engine, pero **orquestadas desde fuera** (por ejemplo desde el QueryBuilder o capas superiores).
- Soporte para un **Execution Pipeline** (pasos de validación, logging, ejecución, post-procesado).
- **Autovalidación** de operaciones (tipos, estado del modelo, constraints básicas) antes de tocar la base de datos.
- Detección automática de modelos decorados con `@table` (`EnsureDatabaseTables`) y creación de **schemas** y **tablas** si no existen.
- Capacidad de **crear la base de datos** si no existe (recibiendo `db_name`).
- Diseño preparado para:
  - Manager de transacciones (Transaction Manager) a nivel de Engine.
  - Sistema de **Triggers** declarativos (archivo `triggers.py` + decorador `@Trigger(...)`).
  - Manejo de “transacciones de dominio” tipo `user.id = 10` (no solo `INSERT/UPDATE` SQL explícitos).
  - Ejecución de queries en paralelo cuando las características de concurrencia de Python 3.14+ estén disponibles.

---

## 2. API Pública del Engine ✅

### 2.1. `create_engine(...)`

Firma orientativa (sujeta a ajustes al implementar):

```python
def create_engine(
    dsn: str | None = None,
    *,
    username: str | None = None,
    password: str | None = None,
    host: str = "localhost",
    port: int = 5432,
    database: str | None = None,
    async_: bool = False,
    pool_size: int = 20,
    auto_adjust_pool_size: bool = False,
    max_pool_size: int | None = None,
    connection_timeout: float | None = None,
    # flags de comportamiento
    ensure_database: bool = True,
    ensure_tables: bool = True,
) -> "Engine":
    ...
```

Características:

- Puede recibir una **cadena de conexión Postgres** (`dsn`) o parámetros sueltos (`username`, `host`, `port`, `database`, etc.).
- `async_`: si `True`, el Engine se inicializa en modo asíncrono (por ejemplo internamente usando `asyncpg`); si `False`, modo síncrono (por ejemplo `psycopg2`).
- `pool_size`: tamaño inicial del pool.
- `auto_adjust_pool_size`: si `True`, el Engine podrá abrir más conexiones cuando la carga lo requiera.
- `max_pool_size`: límite máximo duro cuando se auto-ajusta el pool (si es `None`, se decide un valor por defecto razonable).
- `connection_timeout`: timeout para adquirir una conexión del pool.
- `ensure_database`: si `True`, el Engine intentará crear la base de datos si no existe.
- `ensure_tables`: si `True`, el Engine ejecutará automáticamente `EnsureDatabaseTables` en la inicialización.

La función devolverá una instancia de `Engine` (clase descrita abajo).

### 2.2. Clase `Engine`

Responsabilidades principales:

- Mantener configuración de conexión (DSN, modo sync/async, opciones de pool).
- Mantener y administrar el **pool de conexiones** (incluyendo auto-ajuste y timeouts).
- Proveer un **Transaction Manager** usable desde fuera (QueryBuilder, modelos, etc.).
- Proveer un **Execution Pipeline** para ejecutar consultas/operaciones con pasos comunes (logging, validación, traducción de QueryBuilder → SQL, ejecución, parseo de resultados).
- Exponer puntos de integración para **Triggers** y validaciones automáticas.
- Encapsular la lógica de `EnsureDatabaseTables`.

Métodos clave (borrador):

```python
class Engine:
    def __init__(...):
        ...

    # --- Gestión de conexiones ---
    async def acquire(self): ...  # versión async
    async def release(self, conn): ...

    def acquire_sync(self): ...   # versión sync
    def release_sync(self, conn): ...

    def connection(self): ...     # context manager sync (acquire + release automático)
    async def connection_async(self): ...  # async context manager

    # --- Pipeline de ejecución ---
    def execute(self, query_or_sql, *params, **kwargs): ...
    async def execute_async(self, query_or_sql, *params, **kwargs): ...

    # --- Transacciones ---
    def transaction(self): ...           # devuelve Transaction (sync)
    async def transaction_async(self): ...  # devuelve Transaction (async)

    # --- Auto-setup ---
    def ensure_database(self): ...
    def ensure_tables(self): ...  # llama internamente a EnsureDatabaseTables
```

> Nota: el **release de conexiones será automático** usando context managers (`with` / `async with`), para evitar que el usuario tenga que hacer `acquire`/`release` manuales.

---

## 3. Pool de Conexiones y Concurrencia ✅ (versión inicial sync)

### 3.1. Pool básico

- Pool interno con:
  - `pool_size` inicial.
  - Posible `max_pool_size`.
  - Timeout de adquisición (`connection_timeout`).
- `auto_adjust_pool_size=True` permite que, si todas las conexiones están ocupadas y no hay slots libres, el Engine abra nuevas conexiones **hasta** `max_pool_size`.
- Si se supera el timeout al intentar adquirir una conexión, se lanza una excepción de timeout específica.

### 3.2. Adquisición y liberación inteligentes

- El Engine es el responsable de:
  - Decidir cuándo tomar una conexión del pool (`acquire`).
  - Liberarla automáticamente (`release`) una vez completada la operación, **sin exigir al usuario** gestionar esto manualmente.
- Se expondrán context managers tipo:

  ```python
  with engine.connection() as conn:
      ...  # uso sync

  async with engine.connection_async() as conn:
      ...  # uso async
  ```

### 3.3. Seguridad en presencia de hilos (Thread Safe)

- El Engine y su pool deben ser **Thread Safe**:
  - Uso de locks / primitivas de sincronización adecuados alrededor de las estructuras de datos del pool.
  - Asegurarse de que el `auto_adjust_pool_size` no causa condiciones de carrera al crear/desechar conexiones.

### 3.4. Paralelismo (Python 3.14+)

- Diseño preparado para, en el futuro, soportar **ejecución de queries en paralelo**:
  - Posible integración con nuevas primitivas de concurrencia de Python 3.14+.
  - Interfaz planeada, pero implementación real se pospone:
    - `engine.parallel_execute([...queries...])` (futuro).

---

## 4. EnsureDatabaseTables (Auto-creación de DB, Schemas y Tablas) ✅ (plan DDL en memoria)

### 4.1. Comportamiento general

- Función `EnsureDatabaseTables(engine)` (o método `engine.ensure_tables()`) que se ejecuta **automáticamente** al crear el Engine (si `ensure_tables=True`).
- Responsabilidades:
  1. Recorrer **todos los archivos del proyecto** en busca de modelos decorados con `@table`,
     o bien, si se pasa `models_path` a `create_engine`, limitar el escaneo a esa ruta
     (archivo o directorio) donde viven todos los modelos.
  2. Cargar/importar esos módulos para que las clases modelo queden registradas.
  3. A partir de los modelos (`PSQLModel` + decorador `@table`):
     - Determinar el **schema** (`schema` en el decorador, o `public` por defecto si no se especifica).
     - Construir el DDL para crear schemas y tablas si no existen.
  4. Ejecutar los `CREATE SCHEMA IF NOT EXISTS ...` y `CREATE TABLE IF NOT EXISTS ...` para cada modelo.

### 4.2. Detección de modelos `@table`

- Estrategia:
  - Recorrer el árbol de ficheros del proyecto (por ejemplo con `os.walk`).
  - Encontrar todos los `.py` (excluyendo `__pycache__`, virtualenvs, etc.).
  - Importar dinámicamente los módulos (o usar un registro explícito de modelos mantenido por el decorador `@table`).
  - Filtrar las clases que:
    - Subclasen `PSQLModel`.
    - Tengan atributos `__tablename__` y meta-información generada por `@table` (`__schema__`, `__columns__`, etc.).

### 4.3. Creación de base de datos

- Cuando se usa `create_engine(..., ensure_database=True, database="mydb")`:
  - El Engine intentará conectarse a Postgres a una DB “administrativa” (ej. `postgres`).
  - Si la base de datos `mydb` no existe, ejecutará `CREATE DATABASE mydb`.
  - Luego creará el pool ya apuntando a `mydb`.

---

## 5. Transacciones

### 5.1. Principios

- Las transacciones se **disparan y manejan desde fuera del Engine**, por ejemplo desde el QueryBuilder o desde una capa de dominio, pero el Engine proporciona las primitivas.
- El Engine ofrece un **Transaction Manager** que:
  - Administra `BEGIN` / `COMMIT` / `ROLLBACK`.
  - Maneja nested transactions / savepoints (en el futuro si es necesario).
- Importante: aquí “Transaction Manager” **no** significa solo `INSERT/UPDATE/DELETE` SQL, sino manejo de **transacciones de estado del modelo**, por ejemplo:
  - `user.id = 10`
  - `user.email = "a@b.com"`
  Estas operaciones se verán como parte de una transacción de dominio que luego se traduce a operaciones SQL en el pipeline.

### 5.2. API `Transaction`

- Habrá una función/objeto de alto nivel `Transaction` (expuesta desde el Engine o un módulo común) para iniciar un bloque transaccional:

  ```python
  from orm.engine import Transaction

  with Transaction(engine) as tx:
      user.id = 10
      user.email = "test@example.com"
      # otras mutaciones de modelos
      # el commit/rollback se maneja por el Transaction Manager
  ```

- En modo async:

  ```python
  async with Transaction(engine) as tx:
      ...
  ```

- El Transaction Manager decidirá **cuándo** enviar al DBMS las operaciones acumuladas (ej. en el exit del context manager) y cómo mapear las mutaciones de objetos a `INSERT/UPDATE/DELETE` reales (parte que se relaciona con el archivo de CRUD).

### 5.3. Transaction Manager interno

- Componente interno del Engine que:
  - Rastrea el estado de las entidades (dirty tracking: qué atributos han cambiado).
  - Agrupa cambios en una unidad de trabajo (Unit of Work) durante la vida de la transacción.
  - Coordina con el módulo de CRUD para materializar cambios en SQL.
  - Ejecuta `BEGIN` / `COMMIT` / `ROLLBACK` sobre una conexión tomada del pool.

  ### 5.4. Sesiones síncronas y asíncronas (Session / AsyncSession)

  Además del `Transaction` de bajo nivel, el ORM expondrá una API de **sesiones** al estilo de otros ORMs, pensada para usarse tanto directamente como inyectada como dependencia en frameworks tipo FastAPI.

  #### 5.4.1. Objetivos de `Session` / `AsyncSession`

  - Encapsular un `Transaction` y una conexión del pool bajo una interfaz más familiar:
    - `session.add(model)`
    - `session.flush()` (opcional)
    - `session.commit()` (opcional)
    - `session.refresh(model)` (opcional)
  - Integrar con el Transaction Manager existente, de forma que:
    - `Session` use internamente `Transaction(engine)` (modo sync).
    - `AsyncSession` use internamente `Transaction(engine)` en modo async.
  - Proporcionar una forma cómoda de usarlas como **dependencias** en frameworks web:

  ```python
  def get_session() -> Iterator[Session]:
    with Session(engine) as session:
      yield session

  async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
      yield session
  ```

  #### 5.4.2. API prevista `Session` (sync)

  ```python
  from psqlmodel.transactions import Transaction

  class Session:
    def __init__(self, engine: Engine):
      self.engine = engine
      self._tx: Transaction | None = None

    def __enter__(self) -> "Session":
      # Crea y entra en una Transaction sync interna
      self._tx = Transaction(self.engine)
      self._tx.__enter__()
      return self

    def __exit__(self, exc_type, exc, tb) -> None:
      # Delega commit/rollback al Transaction interno
      if self._tx is not None:
        self._tx.__exit__(exc_type, exc, tb)

    # --- API de trabajo con modelos ---
    def add(self, model: "PSQLModel") -> None:
      # Registra el modelo en la transacción interna
      self._tx.register(model)

    def flush(self) -> None:
      # Versión inicial: opcional; podría forzar un flush inmediato
      # (ejecutar los INSERT/UPDATE de los modelos registrados hasta ahora)
      ...

    def commit(self) -> None:
      # Versión inicial: opcional; en el diseño base el commit se hace al
      # salir del contexto. Se deja prevista esta API para usos futuros.
      ...

    def refresh(self, model: "PSQLModel") -> None:
      # Versión inicial: opcional; haría un SELECT por PK y actualizaría
      # los atributos del modelo con los valores en BD.
      ...
  ```

  Uso típico:

  ```python
  engine = create_engine(...)

  with Session(engine) as session:
    user = User(name="Alice")
    session.add(user)
    user.name = "Alice Updated"
    # Al salir del contexto: BEGIN + INSERT/UPDATE + COMMIT
  ```

  #### 5.4.3. API prevista `AsyncSession` (async)

  ```python
  class AsyncSession:
    def __init__(self, engine: Engine):
      self.engine = engine
      self._tx: Transaction | None = None

    async def __aenter__(self) -> "AsyncSession":
      self._tx = Transaction(self.engine)
      await self._tx.__aenter__()
      return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
      if self._tx is not None:
        await self._tx.__aexit__(exc_type, exc, tb)

    def add(self, model: "PSQLModel") -> None:
      self._tx.register(model)

    async def flush(self) -> None:
      ...

    async def commit(self) -> None:
      ...

    async def refresh(self, model: "PSQLModel") -> None:
      ...
  ```

  Uso típico asíncrono:

  ```python
  engine = create_engine(..., async_=True)

  async with AsyncSession(engine) as session:
    user = User(name="Bob")
    session.add(user)
    user.age = 30
    # Al salir: BEGIN + INSERT/UPDATE + COMMIT en modo async
  ```

  #### 5.4.4. Uso como dependencia en FastAPI y otros frameworks

  La API de `Session`/`AsyncSession` está pensada para integrarse fácilmente como *dependency*:

  ```python
  from typing import Annotated
  from fastapi import Depends, FastAPI

  engine = create_engine(...)

  def get_session() -> Iterator[Session]:
    with Session(engine) as session:
      yield session

  SessionDep = Annotated[Session, Depends(get_session)]

  app = FastAPI()

  @app.get("/users/{user_id}")
  def read_user(user_id: int, session: SessionDep):
    # Aquí se combinarían QueryBuilder + Session/Engine
    ...
  ```

  Para async:

  ```python
  from typing import AsyncIterator, Annotated
  from fastapi import Depends, FastAPI

  engine = create_engine(..., async_=True)

  async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSession(engine) as session:
      yield session

  AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]

  @app.get("/users/{user_id}")
  async def read_user(user_id: int, session: AsyncSessionDep):
    ...
  ```

---

## 6. Triggers

### 6.1. Archivo `triggers.py` y clase `Trigger`

- Se definirá un archivo separado `triggers.py` que contendrá la clase `Trigger`.
- `Trigger` representará la definición de un trigger SQL con métodos para expresar su contenido en Python:

  ```python
  class Trigger:
      def __init__(self, name, timing, event, function, when=None):
          ...

      def Begin(self): ...  # y otros métodos para construir la definición
      # Ej.: BeforeInsert, AfterUpdate, ForEachRow, When(cond), etc.
  ```

### 6.2. Decorador `@Trigger(...)` en modelos

- Los triggers se asociarán a tablas mediante un decorador en los modelos, importando la clase `Trigger` desde `triggers.py`.

  ```python
  from triggers import Trigger

  user_audit_trigger = Trigger(...)

  @Trigger(user_audit_trigger)
  @table(name="users", schema="auth")
  class User(PSQLModel):
      ...
  ```

- Esto permite mantener **modelos, triggers y lógica de dominio modularizados**.

### 6.3. Integración con Engine

- El Engine (y/o `EnsureDatabaseTables`) deberá:
  - Detectar triggers definidos y asociados a modelos.
  - Generar y ejecutar el DDL correspondiente (`CREATE TRIGGER ...`).
- El diseño soportará en el futuro otras capacidades como:
  - LISTEN/NOTIFY.
  - Hooks en modelos vinculados a triggers.

---

## 7. CRUD y Mutaciones de Modelo

### 7.1. Archivo de CRUD separado

- Las operaciones `INSERT`, `UPDATE`, `DELETE`, `CREATE` se mantendrán en un **archivo diferente** (por ejemplo `crud.py`).
- Excepciones:
  - Se podrá usar una primitiva `Insert` para operaciones complejas que involucren subqueries avanzadas generadas por el QueryBuilder.

### 7.2. Mutaciones tipo `user.id = 10`

- En la práctica, el flujo recomendado será:
  - Mutar atributos del modelo en Python (`user.id = 10`, `user.email = ...`).
  - El Transaction Manager marcará esos modelos como “dirty”.
  - Al terminar la transacción (en `Transaction.__exit__`), el CRUD generará los SQL pertinentes.

- Esto separa claramente:
  - **Engine + Transaction Manager** (gestión de transacciones y conexiones).
  - **CRUD** (traducción de cambios de objetos a SQL).
  - **QueryBuilder** (construcción declarativa de queries complejas).

---

## 8. Execution Pipeline y Login

### 8.1. Execution Pipeline

- El Engine tendrá un **pipeline de ejecución** para todas las operaciones:

  1. **Login / Autenticación** (si aplica; p. ej. credenciales, tokens, multi-tenant).
  2. **Autovalidación** (tipos, constraints básicas, estado de los modelos).
  3. **Traducción** de QueryBuilder / operaciones de dominio a SQL + parámetros.
  4. **Planificación** (opcional futuro: orden, paralelismo, retries).
  5. **Ejecución** contra el DBMS usando una conexión del pool.
  6. **Post-proceso** (parseo de filas a modelos, manejo de errores, logging).

### 8.2. Login

- “Login” aquí se refiere a la etapa donde:
  - Se validan credenciales de conexión y/o contexto de seguridad de la app.
  - Se puede inyectar lógica de autenticación/autorización.
- Diseño: exponer hooks en Engine para que el usuario registre funciones de login/autenticación si lo desea.

---

## 9. Autovalidación

- El Engine y el Transaction Manager deberán soportar **autovalidación** antes de ejecutar:
  - Tipos de columnas (compatibles con `psqlmodel.types`).
  - Campos requeridos (`nullable=False`).
  - Constraints simples (ej. primary key presente cuando sea obligatorio).
  - Estado de la transacción (no permitir operaciones en transacción cerrada, etc.).

---

## 10. Resumen de Requisitos Cubiertos

Lista de lo que el diseño contempla según la descripción del usuario:

- `create_engine` con:
  - dsn o parámetros (`username`, `host`, `port`, `database`, `async_`, `pool_size`, `auto_adjust_pool_size`, `connection_timeout`). ✅
- Engine sync/async, con pool de conexiones y **Thread Safe**. ✅
- Auto-ajuste del pool (`auto_adjust_pool_size`) y `max_pool_size`. ✅
- Timeout al adquirir conexiones. ✅
- Adquisición y liberación automáticas de conexiones (`connection()` / `connection_async()`). ✅
- Transacciones manejadas desde fuera (p.ej. QueryBuilder) vía `Transaction`, con Transaction Manager interno. ✅
- Distinción entre transacciones de dominio (`user.id = 10`, etc.) y CRUD explícito separado en otro archivo. ✅
- Soporte de `Insert` para queries complejas como excepción en el CRUD. ✅
- Archivo `triggers.py` con clase `Trigger`, y decorador `@Trigger(...)` para asociar triggers a modelos. ✅
- Integración de triggers en la creación de tablas (DDL de `CREATE TRIGGER`). ✅
- `EnsureDatabaseTables` que:
  - Recorre todos los archivos del proyecto.
  - Busca modelos con `@table`.
  - Usa `schema` del decorador o `public` por defecto.
  - Crea schemas y tablas si no existen. ✅
- Creación de la DB si no existe (`ensure_database`, `database` name). ✅
- Execution Pipeline con pasos de Login, autovalidación, traducción, ejecución, post-proceso. ✅
- Soporte futuro para ejecución en paralelo (Python 3.14+). ✅

Este documento se irá actualizando a medida que implementemos cada módulo (`engine.py`, `crud.py`, `triggers.py`, integración con QueryBuilder, etc.).

---

## 11. Roadmap: Pendientes para Producción

### 11.1. Migrations (Versionado de Schema) ❌

**Estado**: No implementado

**Requisitos**:
- Sistema de migraciones automáticas tipo Alembic
- Detección de cambios en modelos (diff entre modelo Python y schema real)
- Generación automática de scripts de migración
- Versionado con timestamps o números secuenciales
- Comandos CLI: `migrate`, `rollback`, `history`, `autogenerate`
- Soporte para migraciones manuales (SQL raw)
- Transacciones por migración (rollback si falla)

**Diseño propuesto**:
```python
# CLI
psqlmodel migrate init          # Crear carpeta migrations/
psqlmodel migrate autogenerate  # Detectar cambios y generar script
psqlmodel migrate up            # Aplicar migraciones pendientes
psqlmodel migrate down          # Revertir última migración
psqlmodel migrate history       # Ver historial

# Archivo de migración generado
# migrations/20251205_001_create_users.py
class Migration:
    version = "20251205_001"
    depends_on = None
    
    def up(self, engine):
        engine.execute("CREATE TABLE users (...)")
    
    def down(self, engine):
        engine.execute("DROP TABLE users")
```

**Tablas de control**:
```sql
CREATE TABLE _psqlmodel_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    checksum VARCHAR(64)
);
```

---

### 11.2. Relationships (Lazy/Eager Loading) ✅

**Estado**: Implementado

**Requisitos**:
- ✅ Definición declarativa de relaciones (OneToOne, OneToMany, ManyToMany)
- ✅ Lazy loading (cargar relacionados al acceder)
- ✅ Eager loading (cargar con JOIN en query inicial) - Base implementada
- ✅ Backref automático (relación inversa)
- ⚠️ Cascade delete/update - Definido pero requiere integración con Session

**Implementación**:
```python
from psqlmodel import Relationship, OneToMany, ManyToOne, ManyToMany, joinedload

@table(name="users")
class User(PSQLModel):
    id: uuid = Column(primary_key=True)
    name: varchar = Column()
    
    # Relación uno a muchos
    posts: OneToMany["Post"] = Relationship(back_populates="author")

@table(name="posts")
class Post(PSQLModel):
    id: serial = Column(primary_key=True)
    title: varchar = Column()
    author_id: uuid = Column(foreign_key="users.id")
    
    # Relación muchos a uno
    author: ManyToOne[User] = Relationship(back_populates="posts")

# Uso con lazy loading
user = session.get(User, user_id)
for post in user.posts:  # Query ejecutada aquí (cuando esté integrado)
    print(post.title)

# Uso con eager loading (sintaxis preparada)
users = Select(User).options(joinedload(User.posts))
```

**Estrategias de carga implementadas**:
| Estrategia | Descripción | Estado |
|------------|-------------|--------|
| `lazy` | Query separada al acceder (default) | ⚠️ Estructura lista, requiere contexto de session |
| `joined` | JOIN en query principal | ✅ Helpers creados (joinedload) |
| `subquery` | Subquery después de query principal | ✅ Helper creado |
| `selectin` | SELECT ... WHERE id IN (...) | ✅ Helper creado |

**Pendientes para completar**:
- Integrar con Session para ejecutar las queries lazy automáticamente
- Implementar JOIN real en QueryBuilder para eager loading
- Activar cascade operations en Transaction Manager
- Agregar soporte para tabla intermedia automática en ManyToMany

---

### 11.3. Query Execution Completo ✅

**Estado**: Implementado

**Lo que funciona**:
- SELECT con execute()/execute_async()
- INSERT/UPDATE via Session flush (básico)
- DELETE completo con WHERE conditions
- UPDATE masivo (UPDATE ... SET ... WHERE)
- INSERT ... RETURNING (obtener ID generado)
- UPSERT (INSERT ... ON CONFLICT)
- Bulk operations (BulkInsert, BulkUpdate, BulkDelete)
- Raw SQL con parámetros seguros
- session.exec(query) para ejecutar cualquier query builder
- session.exec_one() y session.exec_scalar() para conveniencia

**API implementada**:
```python
# INSERT con RETURNING
query = Insert(User).Values(name="Alice", email="a@b.com").Returning(User.id)
result = session.exec(query)  # [{'id': 1}]

# UPDATE masivo
Update(User).Set(User.is_active, False).Where(User.last_login < cutoff).execute(engine)

# DELETE con WHERE
Delete(User).Where(User.is_active == False).execute(engine)

# UPSERT
Insert(User).values(email="a@b.com", name="Alice").on_conflict(
    User.email, 
    do_update={"name": "Alice Updated"}
).execute(engine)

# Bulk insert
session.add_all([User(name="A"), User(name="B"), User(name="C")])
```

---

### 11.4. Connection Lifecycle ✅

**Estado**: Implementado

**Lo que funciona**:
- Pool de conexiones sync/async
- Acquire/release
- Timeout al adquirir
- **Reconnect automático** cuando conexión muere (health monitor)
- **Health checks** periódicos del pool (start_health_monitor/stop_health_monitor)
- **Connection validation** antes de usar (_repair_sync_pool, _repair_async_pool)
- **Graceful shutdown** del pool (dispose)
- **Retry logic** para operaciones fallidas (RetryMiddleware)

**Diseño propuesto**:
```python
engine = create_engine(
    ...,
    # Nuevos parámetros
    pool_pre_ping=True,           # Validar conexión antes de usar
    pool_recycle=3600,            # Reciclar conexiones cada hora
    max_retries=3,                # Reintentos en operaciones
    retry_delay=0.5,              # Delay entre reintentos
    health_check_interval=30,     # Check cada 30 segundos
)

# Health check endpoint
engine.health_check()  # -> {"status": "healthy", "pool_size": 20, "active": 5}

# Graceful shutdown
await engine.dispose()  # Cierra todas las conexiones limpiamente
```

**Estados de conexión a manejar**:
```
┌─────────────────────────────────────────────────────────┐
│  Pool                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ healthy │  │ healthy │  │  stale  │  │  dead   │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
│       ↓            ↓            ↓            ↓          │
│     usar        usar       reciclar     descartar      │
└─────────────────────────────────────────────────────────┘
```

---

### 11.5. Parametrización Segura (SQL Injection) ✅

**Estado**: Implementado

**Solución implementada**:
- Usar **placeholders** (%s para psycopg2, $1..$n para asyncpg) en todas las queries
- Pasar valores como **parámetros separados** al driver
- Nunca interpolar valores directamente en SQL
- to_sql_params() retorna (sql_template, params) en todos los query builders
- AsyncSession convierte automáticamente %s a $1, $2, ... para asyncpg

**Solución requerida**:
- Usar **placeholders** ($1, $2 o %s) en todas las queries
- Pasar valores como **parámetros separados** al driver
- Nunca interpolar valores directamente en SQL

**Diseño propuesto**:
```python
# ANTES (inseguro):
f"WHERE name = '{value}'"  # ❌ SQL Injection posible

# DESPUÉS (seguro):
("WHERE name = %s", [value])  # ✅ Parametrizado

# Cambios en Query.to_sql():
class Query:
    def to_sql(self) -> tuple[str, list]:
        """Retorna (sql_template, params) en lugar de solo sql."""
        ...

# Ejemplo:
query = Select(User).Where(User.name == "Alice")
sql, params = query.to_sql()
# sql = "SELECT ... WHERE users.name = $1"
# params = ["Alice"]

engine.execute(sql, *params)  # Seguro
```

**Reglas de escape por tipo**:
| Tipo Python | Placeholder |
|-------------|-------------|
| str | $1 (texto) |
| int/float | $1 (numérico) |
| None | NULL |
| bool | TRUE/FALSE |
| list | ANY($1) |
| dict (jsonb) | $1::jsonb |
| datetime | $1::timestamp |

---

### 11.6. Prioridades de Implementación

| Prioridad | Feature | Complejidad | Impacto | Estado |
|-----------|---------|-------------|---------|--------|
| 🟢 Completado | Parametrización segura | Media | Seguridad crítica | ✅ |
| 🟢 Completado | Connection lifecycle | Media | Estabilidad en producción | ✅ |
| 🟢 Completado | Query execution completo | Media | Funcionalidad CRUD | ✅ |
| 🟡 Media | Migrations | Alta | Mantenibilidad | ❌ |
| 🟢 Baja | Relationships | Alta | Developer experience | ❌ |

---

## 12. Changelog

| Fecha | Cambio |
|-------|--------|
| 2024-XX-XX | Diseño inicial del Engine |
| 2024-XX-XX | Implementación pool sync/async |
| 2024-XX-XX | EnsureDatabaseTables + DDL |
| 2024-XX-XX | Transaction Manager + Session |
| 2024-XX-XX | Dirty tracking + CRUD básico |
| 2025-12-05 | Añadido roadmap de pendientes para producción |
| 2025-12-05 | Implementado health checks y auto-reconnect para pool |
| 2025-12-05 | Implementado middleware pipeline con prioridades y timeouts |
| 2025-12-05 | Añadidos middlewares de ejemplo: Validation, Metrics, Audit, Logging, Retry |
| 2025-12-05 | Implementado Query Execution completo: Insert, Update, Delete, BulkOps, UPSERT |
| 2025-12-05 | Implementado session.exec(), exec_one(), exec_scalar() para ejecutar queries |
| 2025-12-05 | Parametrización segura completa (%s sync, $n async) |
