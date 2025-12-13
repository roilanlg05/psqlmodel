"""
Kafka Integration for Triggers

Provides KafkaProduce() wrapper for producing messages to Kafka from triggers.
Requires plpython3u and kafka-python library.

Usage:
    from psqlmodel.kafka_integration import KafkaProduce
    from psqlmodel.trigger_functions import Json
    
    def on_order_status_change():
        payload = Json(
            event="order_status_changed",
            order_id=New.id,
            old_status=Old.status,
            new_status=New.status
        )
        KafkaProduce("orders.events", key=New.id, value=payload)
"""

from typing import Any, Dict, Optional, Union
import json


class KafkaProduce:
    """
    Wrapper for Kafka producer from triggers.
    
    Requires plpython3u with kafka-python installed.
    
    Usage:
        KafkaProduce("topic_name", key="key", value={"event": "something"})
    """
    
    def __init__(
        self,
        topic: str,
        key: Optional[Any] = None,
        value: Optional[Union[Dict, str, Any]] = None,
        bootstrap_servers: Optional[str] = None
    ):
        """
        Initialize Kafka produce.
        
        Args:
            topic: Kafka topic name
            key: Message key (optional)
            value: Message value (will be JSON-serialized if dict)
            bootstrap_servers: Kafka bootstrap servers (default: localhost:9092)
        """
        self.topic = topic
        self.key = key
        self.value = value
        self.bootstrap_servers = bootstrap_servers or "localhost:9092"
    
    def to_plpython(self) -> str:
        """
        Generate plpython code using kafka-python library.
        
        Returns:
            Python code string to execute Kafka produce
        """
        # Serialize key
        if self.key is None:
            key_str = "None"
        elif isinstance(self.key, str):
            key_str = f"b'{self.key}'"
        else:
            key_str = f"str({self.key!r}).encode('utf-8')"
        
        # Serialize value
        if self.value is None:
            value_str = "None"
        elif isinstance(self.value, str):
            value_str = f"b'{self.value}'"
        elif hasattr(self.value, '_is_json') or isinstance(self.value, dict):
            value_str = f"json.dumps({dict(self.value)!r}, default=str).encode('utf-8')"
        else:
            value_str = f"str({self.value!r}).encode('utf-8')"
        
        code = f"""
try:
    from kafka import KafkaProducer
    import json
    producer = KafkaProducer(
        bootstrap_servers='{self.bootstrap_servers}',
        value_serializer=lambda v: v if isinstance(v, bytes) else str(v).encode('utf-8')
    )
    producer.send(
        '{self.topic}',
        key={key_str},
        value={value_str}
    )
    producer.flush()
    producer.close()
except Exception as e:
    plpy.notice(f"Kafka produce failed: {{e}}")
"""
        return code.strip()
    
    def __repr__(self) -> str:
        return f"KafkaProduce(topic='{self.topic}', key={self.key})"


__all__ = ['KafkaProduce']
