"""Legacy compatibility launcher for the canonical txt2img CLI.

New scripts and documentation should invoke::

    python -m modules.txt2img.cli run --interactive
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.txt2img.cli import main


if __name__ == "__main__":
    warnings.warn(
        "app/run.py is a compatibility launcher. Use run.bat or "
        "python -m modules.txt2img.cli run --interactive.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(["run", "--interactive"]))
