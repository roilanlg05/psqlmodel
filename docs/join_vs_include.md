# Join vs JoinRelated vs Include

Guía práctica para saber cuándo usar cada mecanismo de carga de relaciones en PSQLModel.

## 1) Join / LeftJoin / CrossJoin (manual)
- Tú escribes el `JOIN` y la condición `.On(...)`.
- Control total sobre el tipo de JOIN y el predicate.
- No usa metadata de `Relation`; es 100% manual.
- El SELECT solo devuelve las columnas que incluyas; si quieres columnas del relacionado, debes listarlas.

Ejemplo:
```python
q = (
    Select(User.id, User.name, Post.title)
    .LeftJoin(Post).On(Post.user_id == User.id)
    .Where(User.is_active == True)
)
sql, params = q.to_sql_params()
```

## 2) JoinRelated (JOIN automático usando metadata)
- Usa la metadata de `Relation` para detectar la relación y construir la condición FK/PK (incluye many-to-many con la junction).
- Devuelve el `SelectQuery`, así puedes encadenar `.AddRelatedColumns(...)`.
- Útil para eager por JOIN sin escribir la condición a mano.
- No asigna objetos relacionados en memoria; solo construye el SQL con JOIN.

Ejemplos:
```python
# many-to-one: Book.publisher_id → Publisher.id
q = (
    Select(Book)                # SELECT explícito; AddRelatedColumns añade lo que falte
    .JoinRelated(Publisher)     # JOIN automático FK/PK
    .AddRelatedColumns(Publisher.id, Publisher.name)
)

# many-to-many con secondary (explícita o auto)
q = (
    Select(Author)
    .JoinRelated(Book, kind="LEFT")  # genera JOIN junction + JOIN book
    .AddRelatedColumns(Book.title)
)
```

## 3) Include (eager por subconsultas posteriores)
- Ejecuta la consulta base, luego hace consultas adicionales para traer los relacionados y los asigna a los atributos (`order.order_customer`, listas, etc.).
- No modifica el SQL de la consulta principal.
- Devuelve instancias con las relaciones ya pobladas (sin duplicar filas).

Ejemplo:
```python
orders = session.exec(
    Select(Order).Include(Driver)  # carga Driver en una segunda consulta y lo asigna
).all()

print(orders[0].order_customer.name)  # ya viene cargado
```

## Cuándo usar cada uno
- **Join / JoinLeft manual**: cuando necesitas control total sobre la condición o combinaciones poco convencionales.
- **JoinRelated**: para escribir menos boilerplate al hacer JOINs típicos basados en las relaciones declaradas (incluye many-to-many). Añade columnas con `AddRelatedColumns(...)`.
- **Include**: cuando quieres instancias con relaciones ya asignadas sin preocuparte por condiciones de JOIN ni duplicados; hace las consultas extra automáticamente.
