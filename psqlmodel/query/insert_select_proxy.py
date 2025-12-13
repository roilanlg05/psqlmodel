# InsertSelectProxy class for fluent chaining
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
