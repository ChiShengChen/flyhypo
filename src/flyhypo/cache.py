"""Dead-simple on-disk JSON cache for neuPrint + literature queries.

A PoC cache: one JSON file per key under ``.flyhypo_cache/``. Good enough to
avoid re-hitting neuPrint / PubMed on repeated runs of the same query, and easy
to inspect or wipe (``rm -rf .flyhypo_cache``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("FLYHYPO_CACHE_DIR", ".flyhypo_cache"))


def _path(namespace: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / namespace / f"{digest}.json"


def get(namespace: str, key: str) -> Any | None:
    p = _path(namespace, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def put(namespace: str, key: str, value: Any) -> None:
    p = _path(namespace, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, default=str))
