"""
psqlmodel/middlewares.py – Example middleware implementations for the Execution Pipeline.

Provides ready-to-use middlewares for:
- Validation (checks Column constraints before execution)
- Metrics collection (timing, query counting)
- Audit logging (records all executed queries)

Usage:
    from psqlmodel.middlewares import ValidationMiddleware, MetricsMiddleware, AuditMiddleware
    
    engine.add_middleware_sync(ValidationMiddleware(model=User).sync, priority=10)
    engine.add_middleware_sync(MetricsMiddleware().sync, priority=20)
    engine.add_middleware_sync(AuditMiddleware().sync, priority=30)
"""

from __future__ import annotations

import time
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict
from datetime import datetime


# ============================================================
# VALIDATION MIDDLEWARE
# ============================================================

class ValidationMiddleware:
    """Middleware that validates parameter values against Column constraints.
    
    This middleware inspects INSERT/UPDATE queries and validates the parameters
    against the model's Column definitions (max_len, min_len, max_value, min_value).
    
    Args:
        model: Optional PSQLModel class to validate against. If None, validation
               is skipped (passthrough).
        strict: If True, raises on validation failure. If False, logs warnings.
    
    Usage:
        validator = ValidationMiddleware(model=User, strict=True)
        engine.add_middleware_sync(validator.sync, priority=10)
        engine.add_middleware_async(validator.async_, priority=10)
    """
    
    def __init__(self, model=None, strict: bool = True):
        self.model = model
        self.strict = strict
        self._logger = logging.getLogger("psqlmodel.validation")
    
    def _validate_params(self, sql: str, params: Any) -> None:
        """Validate parameters against model Column constraints."""
        if self.model is None or not params:
            return
        
        # Get column definitions from model
        columns = getattr(self.model, "__columns__", {})
        if not columns:
            return
        
        # Try to match params to columns based on SQL structure
        # This is a simplified approach - for INSERT we match by position
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith("INSERT"):
            # Extract column names from INSERT statement
            match = re.search(r'INSERT\s+INTO\s+\S+\s*\(([^)]+)\)', sql, re.IGNORECASE)
            if match:
                col_names = [c.strip().strip('"') for c in match.group(1).split(',')]
                param_list = list(params) if params else []
                
                for i, col_name in enumerate(col_names):
                    if i >= len(param_list):
                        break
                    
                    col = columns.get(col_name)
                    if col is None:
                        continue
                    
                    value = param_list[i]
                    try:
                        col.validate_value(value)
                    except ValueError as e:
                        if self.strict:
                            raise
                        else:
                            self._logger.warning(f"Validation warning: {e}")
    
    def sync(self, sql: str, params: Any, next_: Callable) -> Any:
        """Synchronous middleware entry point."""
        self._validate_params(sql, params)
        return next_(sql, params)
    
    async def async_(self, sql: str, params: Any, next_: Callable) -> Any:
        """Asynchronous middleware entry point."""
        self._validate_params(sql, params)
        return await next_(sql, params)


# ============================================================
# METRICS MIDDLEWARE
# ============================================================

@dataclass
class QueryMetrics:
    """Container for collected query metrics."""
    total_queries: int = 0
    total_time_ms: float = 0.0
    queries_by_type: Dict[str, int] = field(default_factory=dict)
    slowest_query_ms: float = 0.0
    slowest_query_sql: str = ""
    last_query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.total_time_ms / max(1, self.total_queries), 3),
            "queries_by_type": dict(self.queries_by_type),
            "slowest_query_ms": round(self.slowest_query_ms, 3),
            "slowest_query_sql": self.slowest_query_sql[:100] + "..." if len(self.slowest_query_sql) > 100 else self.slowest_query_sql,
            "last_query_time_ms": round(self.last_query_time_ms, 3),
        }


class MetricsMiddleware:
    """Middleware that collects execution metrics for queries.
    
    Tracks:
    - Total query count
    - Total execution time
    - Queries by type (SELECT, INSERT, UPDATE, DELETE)
    - Slowest query
    - Last query execution time
    
    Args:
        slow_query_threshold_ms: Log warning for queries slower than this (default: 1000ms)
    
    Usage:
        metrics = MetricsMiddleware(slow_query_threshold_ms=500)
        engine.add_middleware_sync(metrics.sync, priority=20)
        
        # Later: access metrics
        print(metrics.get_metrics().to_dict())
        metrics.reset()
    """
    
    def __init__(self, slow_query_threshold_ms: float = 1000.0):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self._metrics = QueryMetrics()
        self._logger = logging.getLogger("psqlmodel.metrics")
    
    def _get_query_type(self, sql: str) -> str:
        """Extract query type from SQL statement."""
        sql_upper = sql.strip().upper()
        for qtype in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "BEGIN", "COMMIT", "ROLLBACK"):
            if sql_upper.startswith(qtype):
                return qtype
        return "OTHER"
    
    def _record_metrics(self, sql: str, elapsed_ms: float) -> None:
        """Record metrics for a query execution."""
        self._metrics.total_queries += 1
        self._metrics.total_time_ms += elapsed_ms
        self._metrics.last_query_time_ms = elapsed_ms
        
        qtype = self._get_query_type(sql)
        self._metrics.queries_by_type[qtype] = self._metrics.queries_by_type.get(qtype, 0) + 1
        
        if elapsed_ms > self._metrics.slowest_query_ms:
            self._metrics.slowest_query_ms = elapsed_ms
            self._metrics.slowest_query_sql = sql
        
        if elapsed_ms > self.slow_query_threshold_ms:
            self._logger.warning(f"Slow query detected ({elapsed_ms:.2f}ms): {sql[:100]}")
    
    def get_metrics(self) -> QueryMetrics:
        """Get the current metrics snapshot."""
        return self._metrics
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self._metrics = QueryMetrics()
    
    def sync(self, sql: str, params: Any, next_: Callable) -> Any:
        """Synchronous middleware entry point."""
        start = time.perf_counter()
        try:
            result = next_(sql, params)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_metrics(sql, elapsed_ms)
    
    async def async_(self, sql: str, params: Any, next_: Callable) -> Any:
        """Asynchronous middleware entry point."""
        start = time.perf_counter()
        try:
            result = await next_(sql, params)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_metrics(sql, elapsed_ms)


# ============================================================
# AUDIT MIDDLEWARE
# ============================================================

@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: datetime
    sql: str
    params: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "sql": self.sql,
            "params": str(self.params)[:200] if self.params else None,
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
            "error": self.error,
        }


class AuditMiddleware:
    """Middleware that logs all query executions for audit purposes.
    
    Maintains an in-memory log of all queries with:
    - Timestamp
    - SQL statement
    - Parameters (truncated)
    - Duration
    - Success/failure status
    
    Args:
        max_entries: Maximum audit entries to keep in memory (default: 1000)
        log_to_logger: If True, also log to Python logger (default: False)
        sensitive_patterns: List of regex patterns to redact from logs
    
    Usage:
        audit = AuditMiddleware(max_entries=500, log_to_logger=True)
        engine.add_middleware_sync(audit.sync, priority=5)  # Early priority for timing
        
        # Later: get audit log
        for entry in audit.get_entries():
            print(entry.to_dict())
        
        audit.clear()
    """
    
    def __init__(
        self, 
        max_entries: int = 1000, 
        log_to_logger: bool = False,
        sensitive_patterns: Optional[List[str]] = None
    ):
        self.max_entries = max_entries
        self.log_to_logger = log_to_logger
        self.sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in (sensitive_patterns or [])]
        self._entries: List[AuditEntry] = []
        self._logger = logging.getLogger("psqlmodel.audit")
    
    def _redact_sensitive(self, text: str) -> str:
        """Redact sensitive information from text."""
        for pattern in self.sensitive_patterns:
            text = pattern.sub("[REDACTED]", text)
        return text
    
    def _add_entry(self, entry: AuditEntry) -> None:
        """Add an audit entry, removing oldest if at capacity."""
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(entry)
        
        if self.log_to_logger:
            sql_redacted = self._redact_sensitive(entry.sql)
            if entry.success:
                self._logger.info(f"[AUDIT] {entry.duration_ms:.2f}ms: {sql_redacted[:100]}")
            else:
                self._logger.error(f"[AUDIT] FAILED ({entry.error}): {sql_redacted[:100]}")
    
    def get_entries(self, limit: Optional[int] = None) -> List[AuditEntry]:
        """Get audit entries (most recent first)."""
        entries = list(reversed(self._entries))
        if limit:
            entries = entries[:limit]
        return entries
    
    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()
    
    def sync(self, sql: str, params: Any, next_: Callable) -> Any:
        """Synchronous middleware entry point."""
        start = time.perf_counter()
        error_msg = None
        success = True
        
        try:
            result = next_(sql, params)
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            entry = AuditEntry(
                timestamp=datetime.now(),
                sql=self._redact_sensitive(sql),
                params=params,
                duration_ms=elapsed_ms,
                success=success,
                error=error_msg,
            )
            self._add_entry(entry)
    
    async def async_(self, sql: str, params: Any, next_: Callable) -> Any:
        """Asynchronous middleware entry point."""
        start = time.perf_counter()
        error_msg = None
        success = True
        
        try:
            result = await next_(sql, params)
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            entry = AuditEntry(
                timestamp=datetime.now(),
                sql=self._redact_sensitive(sql),
                params=params,
                duration_ms=elapsed_ms,
                success=success,
                error=error_msg,
            )
            self._add_entry(entry)


# ============================================================
# LOGGING MIDDLEWARE (Simple)
# ============================================================

class LoggingMiddleware:
    """Simple middleware that logs all queries to a Python logger.
    
    Args:
        logger_name: Name of the logger to use (default: "psqlmodel.queries")
        level: Logging level (default: logging.DEBUG)
        include_params: Whether to include parameters in log (default: False for security)
    
    Usage:
        logging_mw = LoggingMiddleware(level=logging.INFO, include_params=False)
        engine.add_middleware_sync(logging_mw.sync, priority=50)
    """
    
    def __init__(
        self, 
        logger_name: str = "psqlmodel.queries",
        level: int = logging.DEBUG,
        include_params: bool = False
    ):
        self._logger = logging.getLogger(logger_name)
        self.level = level
        self.include_params = include_params
    
    def sync(self, sql: str, params: Any, next_: Callable) -> Any:
        """Synchronous middleware entry point."""
        if self.include_params:
            self._logger.log(self.level, f"Executing: {sql} | Params: {params}")
        else:
            self._logger.log(self.level, f"Executing: {sql}")
        return next_(sql, params)
    
    async def async_(self, sql: str, params: Any, next_: Callable) -> Any:
        """Asynchronous middleware entry point."""
        if self.include_params:
            self._logger.log(self.level, f"Executing (async): {sql} | Params: {params}")
        else:
            self._logger.log(self.level, f"Executing (async): {sql}")
        return await next_(sql, params)


# ============================================================
# RETRY MIDDLEWARE
# ============================================================

class RetryMiddleware:
    """Middleware that automatically retries failed queries.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Base delay between retries in seconds (default: 0.5)
        exponential_backoff: If True, doubles delay on each retry (default: True)
        retryable_exceptions: Tuple of exception types to retry (default: common DB errors)
    
    Usage:
        retry = RetryMiddleware(max_retries=3, retry_delay=0.5)
        engine.add_middleware_sync(retry.sync, priority=5)  # Early for wrapping
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        exponential_backoff: bool = True,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        # Default retryable exceptions (connection errors, deadlocks, etc.)
        self.retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            OSError,
        )
        self._logger = logging.getLogger("psqlmodel.retry")
    
    def sync(self, sql: str, params: Any, next_: Callable) -> Any:
        """Synchronous middleware entry point with retry logic."""
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return next_(sql, params)
            except self.retryable_exceptions as e:
                last_error = e
                if attempt < self.max_retries:
                    self._logger.warning(f"Query failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    if self.exponential_backoff:
                        delay *= 2
        
        raise last_error  # type: ignore
    
    async def async_(self, sql: str, params: Any, next_: Callable) -> Any:
        """Asynchronous middleware entry point with retry logic."""
        import asyncio
        
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await next_(sql, params)
            except self.retryable_exceptions as e:
                last_error = e
                if attempt < self.max_retries:
                    self._logger.warning(f"Query failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    if self.exponential_backoff:
                        delay *= 2
        
        raise last_error  # type: ignore
