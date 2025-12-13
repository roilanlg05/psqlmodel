"""
HTTP Integration for Triggers

Provides HttpPost() wrapper for making HTTP requests from triggers.
Supports both plpython3u (using requests) and plpgsql (using pg_http extension).

Usage:
    from psqlmodel.http import HttpPost
    from psqlmodel.trigger_functions import Json
    
    def on_user_created():
        body = Json(
            event="user_created",
            id=New.id,
            email=New.email
        )
        HttpPost(
            "https://api.example.com/hooks/user",
            body=body,
            content_type="application/json",
            timeout=2
        )
"""

from typing import Any, Dict, Optional, Union
import json


class HttpPost:
    """
    Wrapper for HTTP POST requests from triggers.
    
    In plpython3u: uses requests library
    In plpgsql: uses pg_http extension
    
    Usage:
        HttpPost(
            "https://api.example.com/endpoint",
            body={"event": "something"},
            content_type="application/json",
            timeout=2
        )
    """
    
    def __init__(
        self,
        url: str,
        body: Optional[Union[Dict, str, Any]] = None,
        content_type: str = "application/json",
        timeout: int = 5,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize HTTP POST request.
        
        Args:
            url: Target URL
            body: Request body (will be JSON-serialized if dict)
            content_type: Content-Type header
            timeout: Request timeout in seconds
            headers: Additional headers
        """
        self.url = url
        self.body = body
        self.content_type = content_type
        self.timeout = timeout
        self.headers = headers or {}
    
    def to_plpython(self) -> str:
        """
        Generate plpython code using requests library.
        
        Returns:
            Python code string to execute HTTP POST
        """
        # Serialize body if needed
        if self.body is None:
            body_str = "None"
        elif isinstance(self.body, str):
            body_str = repr(self.body)
        elif hasattr(self.body, '_is_json') or isinstance(self.body, dict):
            body_str = f"json.dumps({dict(self.body)!r}, default=str)"
        else:
            body_str = repr(str(self.body))
        
        headers = {
            "Content-Type": self.content_type,
            **self.headers
        }
        
        code = f"""
try:
    import requests
    import json
    requests.post(
        "{self.url}",
        data={body_str},
        headers={headers!r},
        timeout={self.timeout}
    )
except Exception as e:
    plpy.notice(f"HTTP POST failed: {{e}}")
"""
        return code.strip()
    
    def to_plpgsql(self) -> str:
        """
        Generate plpgsql code using pg_http extension.
        
        Returns:
            PL/pgSQL code string to execute HTTP POST
        """
        # Serialize body
        if self.body is None:
            body_literal = "''"
        elif isinstance(self.body, str):
            body_literal = f"'{self.body}'"
        elif hasattr(self.body, '_is_json') or isinstance(self.body, dict):
            body_json = json.dumps(self.body, default=str).replace("'", "''")
            body_literal = f"'{body_json}'"
        else:
            body_literal = f"'{str(self.body)}'"
        
        # Build headers JSON
        headers = {
            "Content-Type": self.content_type,
            **self.headers
        }
        headers_json = json.dumps(headers).replace("'", "''")
        
        code = f"""
PERFORM http_post(
    '{self.url}',
    {body_literal},
    'application/json',
    '{headers_json}'
);
"""
        return code.strip()
    
    def __repr__(self):
        return f"HttpPost(url='{self.url}', timeout={self.timeout})"


__all__ = ['HttpPost']
