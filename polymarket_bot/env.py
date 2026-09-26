"""Environment loading and typed accessors.

`.env` is loaded eagerly at import time, before any dataclass default is
evaluated, so configuration actually takes effect. This is the fix for the
classic failure where `load_dotenv()` runs after `Config()` defaults were
already frozen at class-definition time.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Parse a simple KEY=VALUE .env file into os.environ.

    Supports `export KEY=VALUE`, `#` comments, quoted values, and inline
    comments that follow an unquoted value.
    """
    p = Path(path) if path else Path(".env")
    if not p.is_file():
        return None
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] in {'"', "'"}:
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = value.split(" #", 1)[0].strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return p


def ensure_loaded() -> None:
    """Load the default .env once per process (idempotent)."""
    global _LOADED
    if not _LOADED:
        load_dotenv()
        _LOADED = True


def get_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def get_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
