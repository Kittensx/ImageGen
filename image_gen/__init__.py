"""Source-tree bootstrap for the canonical :mod:`image_gen` package.

The implementation lives in ``src/image_gen``.  This tiny bridge keeps the
uninstalled development tree importable when the project is launched directly
from its root.  A future editable/installed package can import ``image_gen``
from ``src`` without this bridge.
"""
from __future__ import annotations

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "image_gen"
if not _SOURCE_PACKAGE.is_dir():
    raise ImportError(f"Canonical image_gen package not found: {_SOURCE_PACKAGE}")

# Point package discovery at the canonical src-layout package.
__path__ = [str(_SOURCE_PACKAGE)]

_source_init = _SOURCE_PACKAGE / "__init__.py"
exec(compile(_source_init.read_text(encoding="utf-8"), str(_source_init), "exec"), globals(), globals())
