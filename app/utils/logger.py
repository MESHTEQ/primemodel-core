"""
app/utils/logger.py
--------------------
Structured JSON logger for PrimeModel AI Engine.
All services import `get_logger(__name__)` — never use print() in production code.
"""

import logging
import json
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Suitable for Railway log aggregation and Supabase Edge Function caller inspection.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed as record attributes
        extra_keys = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            )
        }
        if extra_keys:
            payload["extra"] = extra_keys
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with JSON output.
    Log level is read from the LOG_LEVEL env variable via config.
    Calling get_logger multiple times with the same name returns the same instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    # Lazily resolve log level to avoid circular import with config at module load
    import os
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    return logger
