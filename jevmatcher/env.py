"""Minimal .env loader (no dependency). Real environment variables always win."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(*paths: str | Path) -> list[Path]:
    """Load KEY=VALUE lines from the first-found .env files into os.environ, without overriding.

    Looks in the current directory, then the project root. Returns the files that were read.
    """
    read = []
    for p in paths or (Path.cwd() / ".env", ROOT / ".env"):
        p = Path(p)
        if not p.is_file() or p in read:
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.removeprefix("export ").strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1]
            os.environ.setdefault(key, val)
        read.append(p)
    return read
