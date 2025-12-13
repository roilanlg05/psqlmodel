# PSQLModel Relationships Reference

This document provides a comprehensive technical reference for `psqlmodel/orm/relationships.py`. The ORM uses a unified `Relation` class that automatically detects relationship types based on convention.

## Overview

The relationship system is built around a single class: `Relation`. It simplifies definition by automatically inferring:
- **Type**: `OneToOne`, `OneToMany`, `ManyToOne`, or `ManyToMany`.
- **Foreign Keys**: Auto-detects `user_id`, `order_id` naming conventions.
- **Junction Tables**: Automatically handles many-to-many intermediate tables.

---

## 1. Defining Relations

### Syntax

```python
class User(PSQLModel):
    # Forward reference as string
    posts: Relation["Post"] = Relation("Post")

class Post(PSQLModel):
    user_id: int = Column(foreign_key="users.id")
    # Direct class reference
    user: Relation["User"] = Relation(User)
```

### `Relation` Arguments
- **`table_name`** (Required): The target model class or its name as a string (to avoid circular imports).
- **`secondary`** (Optional):
    - `None`: Standard relationship.
    - `"auto"`: Automatically generates a junction table name for Many-to-Many.
    - `"custom_table_name"`: Uses a specific table for Many-to-Many.

### Auto-Detection Logic
The `Relation` descriptor inspects both models to decide the type:
1.  **ManyToMany**: If `secondary` is provided.
2.  **ManyToOne**: If the **Owner** model has a Foreign Key to the **Target**.
3.  **OneToOne**: If the **Owner** model has a Foreign Key to the **Target** AND that column has a `UNIQUE` constraint.
4.  **OneToMany**: If the **Target** model has a Foreign Key to the **Owner**.

---

## 2. Using Relations

### Access Patterns

#### Lazy Loading (Sync only)
When you access a relationship attribute, the ORM executes a query on the fly (N+1 query pattern).
> **Requirement**: The instance must be attached to an active `Session`.

```python
user = session.get(User, 1)
print(user.posts)  # SELECT * FROM posts WHERE user_id = 1
```

#### Eager Loading (Recommended)
Use `.Include()` in your query to fetch related data in a single round-trip (or optimized batch).

```python
# Fetches User + Posts efficiently
user = session.exec(
    Select(User).Where(User.id == 1).Include(Post)
).first()

# No extra query here
print(user.posts)
```

---

## 3. Collections & Filtering (`RelationProxy`)

For `OneToMany` and `ManyToMany` relationships, accessing the attribute returns a `RelationProxy`. This object acts like a list but supports additional query operations.

### `RelationProxy` Features
- **Iterable**: behave like a standard list.
- **`.filter(condition)`**: Returns a subset of related items.
    - **In-Memory**: Pass a lambda/function.
    - **Database**: Pass a SQL expression.

```python
# Database Filter (Executes SQL: SELECT ... WHERE user_id=? AND rating > 4)
high_rated_posts = user.posts.filter(Post.rating > 4).all()

# In-Memory Filter (No SQL)
recent_posts = user.posts.filter(lambda p: p.created_at > yesterday)
```

### `FilterResult`
The result of a `.filter()` operation.
- **`.all()`**: Returns list.
- **`.first()`**: Returns first item or None.
- **`.limit(n)`**: Slices the result.
- **`.order_by(column)`**: Sorts the result.

---

## 4. Many-to-Many

To create a M2M relationship, simply use `secondary="auto"` on one side.

```python
class Student(PSQLModel):
    courses: Relation["Course"] = Relation("Course", secondary="auto")

class Course(PSQLModel):
    students: Relation["Student"] = Relation("Student", secondary="auto")
```

The system will automatically manage the hidden junction table `course_student_junction`.

### Internal Mechanics
1.  **DDL Generation**: The `Engine` detects these relationships and creates the junction table automatically during `ensure_tables()`.
2.  **Querying**: Lazy loading executes a 2-step process or a JOIN (depending on optimization) to fetch targets via the junction table.

---

## 5. Async Support

**Important**: Lazy Loading (`user.posts`) is **NOT** supported in Async mode because Python's properties (`__get__`) cannot be `await`ed transparently.

### Async Best Practice
Always use **Eager Loading** with `Include()`.

```python
# Correct Async Usage
users = await session.exec(
    Select(User).Include(Post)
)
for user in users:
    print(user.posts)  # Safe, data is already loaded
```
