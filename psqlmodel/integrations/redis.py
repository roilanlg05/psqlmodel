"""
Redis Integration for Triggers

Provides RedisPublish() wrapper for publishing to Redis channels from triggers.
Requires plpython3u and redis-py library.

Usage:
    from psqlmodel.redis_integration import RedisPublish
    from psqlmodel.trigger_functions import Json
    
    def on_user_login():
        payload = Json(
            event="user_login",
            user_id=New.user_id,
            ip=New.ip_address
        )
        RedisPublish("realtime:user_login", payload)
"""

from typing import Any, Dict, Optional, Union
import json


class RedisPublish:
    """
    Wrapper for Redis PUBLISH command from triggers.
    
    Requires plpython3u with redis-py installed.
    
    Usage:
        RedisPublish("channel_name", {"event": "something"})
    """
    
    def __init__(
        self,
        channel: str,
        payload: Union[Dict, str, Any],
        redis_url: Optional[str] = None
    ):
        """
        Initialize Redis publish.
        
        Args:
            channel: Redis channel name
            payload: Message to publish (will be JSON-serialized if dict)
            redis_url: Redis connection URL (default: redis://localhost:6379/0)
        """
        self.channel = channel
        self.payload = payload
        self.redis_url = redis_url or "redis://localhost:6379/0"
    
    def to_plpython(self) -> str:
        """
        Generate plpython code using redis-py library.
        
        Returns:
            Python code string to execute Redis PUBLISH
        """
        # Serialize payload
        if isinstance(self.payload, str):
            payload_str = repr(self.payload)
        elif hasattr(self.payload, '_is_json') or isinstance(self.payload, dict):
            payload_str = f"json.dumps({dict(self.payload)!r}, default=str)"
        else:
            payload_str = repr(str(self.payload))
        
        code = f"""
try:
    import redis
    import json
    r = redis.from_url("{self.redis_url}")
    r.publish("{self.channel}", {payload_str})
    r.close()
except Exception as e:
    plpy.notice(f"Redis publish failed: {{e}}")
"""
        return code.strip()
    
    def __repr__(self) -> str:
        return f"RedisPublish(channel='{self.channel}')"


__all__ = ['RedisPublish']
