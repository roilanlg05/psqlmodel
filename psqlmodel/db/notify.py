import json
import re
import zlib
import datetime
import uuid
import decimal
import psycopg
import asyncpg


class Notify:
    """
    Advanced NOTIFY builder compatible with:
        - Subscribe()
        - PL/Python triggers (via .plpy())
        - psycopg and asyncpg execution
        - broadcast
        - compression
        - metadata injection

    Still minimal, predictable and safe.
    """

    CHANNEL_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def __init__(self, trigger_or_channel):
        if hasattr(trigger_or_channel, "trigger_name"):
            self.channel = trigger_or_channel.trigger_name
        else:
            self.channel = str(trigger_or_channel)

        self._payload = None
        self._broadcast_channels = []
        self._auto_metadata = True
        self._compression = True

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------

    def Message(self, payload):
        self._payload = payload
        return self

    def Broadcast(self, *channels):
        """Broadcast the same payload to additional channels."""
        self._broadcast_channels.extend(channels)
        return self

    def NoMetadata(self):
        """Disable auto metadata injection."""
        self._auto_metadata = False
        return self

    def NoCompression(self):
        """Disable compression for large payloads."""
        self._compression = False
        return self

    # ---------------------------------------------------------------------
    # SENDERS
    # ---------------------------------------------------------------------

    def SendSync(self, dsn=None):
        """Send NOTIFY synchronously using psycopg."""
        sql, params_list = self._multi_sql_params()
        conn = psycopg.connect(dsn or self._default_dsn(), autocommit=True)
        cur = conn.cursor()

        for sql_cmd, params in params_list:
            cur.execute(sql_cmd, params)

        cur.close()
        conn.close()

    async def SendAsync(self, dsn=None):
        """Send NOTIFY asynchronously using asyncpg."""
        sql, params_list = self._multi_sql_params()
        conn = await asyncpg.connect(dsn or self._default_dsn())

        for sql_cmd, params in params_list:
            payload = params[0]
            await conn.execute(sql_cmd.replace("%s", f"'{payload}'"))

        await conn.close()

    # ---------------------------------------------------------------------
    # SQL GENERATION (PL/PYTHON)
    # ---------------------------------------------------------------------

    def plpy(self):
        """
        Returns a string like:
            NOTIFY channel, 'json_payload';
            NOTIFY channel2, 'json_payload';
        """
        payload = self._prepare_payload()

        payload = payload.replace("'", "''")  # escape for SQL literal

        sql_commands = []

        sql_commands.append(f"NOTIFY {self.channel}, '{payload}'")

        for ch in self._broadcast_channels:
            self._validate_channel(ch)
            sql_commands.append(f"NOTIFY {ch}, '{payload}'")

        return ";\n".join(sql_commands)

    # ---------------------------------------------------------------------
    # SQL GENERATION FOR ENGINE / psycopg
    # ---------------------------------------------------------------------

    def to_sql_params(self):
        """Return SQL and payload for a single-channel notify."""
        payload = self._prepare_payload()
        sql = f"NOTIFY {self.channel}, %s"
        return sql, [payload]

    def _multi_sql_params(self):
        """
        Returns:
            [
                ("NOTIFY channel, %s", [payload]),
                ("NOTIFY channel2, %s", [payload]),
                ...
            ]
        """
        payload = self._prepare_payload()

        results = []
        results.append((f"NOTIFY {self.channel}, %s", [payload]))

        for ch in self._broadcast_channels:
            self._validate_channel(ch)
            results.append((f"NOTIFY {ch}, %s", [payload]))

        return results

    # ---------------------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------------------

    def _prepare_payload(self):
        """normalize → metadata → json → compress"""

        if self._payload is None:
            return ""

        obj = self._normalize_json(self._payload)

        if self._auto_metadata:
            obj["_meta"] = {
                "sent_at": datetime.datetime.utcnow().isoformat(),
                "channel": self.channel,
            }

        json_str = json.dumps(obj)

        if self._compression and len(json_str.encode("utf-8")) > 7900:
            compressed = zlib.compress(json_str.encode("utf-8"))
            return "__compressed__:" + compressed.hex()

        return json_str

    def _normalize_json(self, obj):
        """Convert UUID, datetime, Decimal → JSON-safe."""

        def encode(o):
            if isinstance(o, (datetime.datetime, datetime.date)):
                return o.isoformat()
            if isinstance(o, uuid.UUID):
                return str(o)
            if isinstance(o, decimal.Decimal):
                return float(o)
            return o

        return json.loads(json.dumps(obj, default=encode))

    def _validate_channel(self, ch):
        if not Notify.CHANNEL_REGEX.match(ch):
            raise ValueError(f"Invalid channel name: {ch!r}")

    @staticmethod
    def _default_dsn():
        import os
        user = os.environ.get("PSQL_USER", "hashdown")
        password = os.environ.get("PSQL_PASSWORD", "Rlg*020305")
        host = os.environ.get("PSQL_HOST", "localhost")
        port = os.environ.get("PSQL_PORT", "5433")
        db = os.environ.get("PSQL_DB", "api3602")
        return f"postgres://{user}:{password}@{host}:{port}/{db}"
