"""Central ImageGen application identity and source-build metadata.

Application versions identify intentional alpha/beta/stable releases. Git commit
identity identifies the exact source revision when ImageGen is run from a Git
checkout. The first public alpha commit remains recorded separately as a stable
baseline and must not be confused with a later development build.
"""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCT_NAME = "ImageGen"
APPLICATION_VERSION = "0.1.0-alpha.1"
RELEASE_CHANNEL = "alpha"
METADATA_SCHEMA_VERSION = "1"

# First ImageGen alpha commit. This is a historical baseline, not a value that
# should be manually rewritten for every ordinary development commit.
ALPHA_BASELINE_COMMIT_FULL = "594c8bfc0a89499a915fa0b3370212cafb2980cc"
ALPHA_BASELINE_COMMIT_SHORT = "594c8bf"
ALPHA_BASELINE_LABEL = "first initial alpha release"

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _normalize_commit(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if _COMMIT_RE.fullmatch(token) else ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ""
    return str(completed.stdout or "").strip()


@lru_cache(maxsize=4)
def resolve_build_identity(project_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve the best available source-build identity.

    Resolution order:
    1. Explicit build environment variable (useful for packaged/CI builds).
    2. Current Git HEAD from the running checkout.
    3. The recorded first-alpha baseline as a truthful fallback.

    A dirty Git checkout is intentionally marked as not being an exact commit
    snapshot because its working files differ from HEAD.
    """

    root = Path(project_root).resolve() if project_root is not None else _project_root()

    env_commit = _normalize_commit(
        os.environ.get("IMAGEGEN_BUILD_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or ""
    )
    if env_commit:
        full = env_commit
        short = full[:7]
        source = "environment"
        dirty = False
        exact = len(full) == 40
    else:
        git_commit = _normalize_commit(_run_git(root, "rev-parse", "HEAD"))
        if git_commit:
            full = git_commit
            short = full[:7]
            source = "git_head"
            dirty = bool(_run_git(root, "status", "--porcelain", "--untracked-files=no"))
            exact = len(full) == 40 and not dirty
        else:
            full = ALPHA_BASELINE_COMMIT_FULL
            short = ALPHA_BASELINE_COMMIT_SHORT
            source = "alpha_baseline_fallback"
            dirty = False
            exact = False

    if source == "alpha_baseline_fallback":
        display = f"baseline {short}"
    elif dirty:
        display = f"{short}+dirty"
    else:
        display = short

    return {
        "commit_full": full,
        "commit_short": short,
        "source": source,
        "working_tree_dirty": dirty,
        "exact_source_snapshot": exact,
        "display": display,
        "baseline_commit_full": ALPHA_BASELINE_COMMIT_FULL,
        "baseline_commit_short": ALPHA_BASELINE_COMMIT_SHORT,
        "baseline_label": ALPHA_BASELINE_LABEL,
    }


def build_program_metadata(project_root: str | Path | None = None) -> dict[str, Any]:
    build = resolve_build_identity(project_root)
    return {
        "name": PRODUCT_NAME,
        "version": APPLICATION_VERSION,
        "release_channel": RELEASE_CHANNEL,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "build": dict(build),
        "display_version": f"{APPLICATION_VERSION} · {build['display']}",
    }


__all__ = [
    "ALPHA_BASELINE_COMMIT_FULL",
    "ALPHA_BASELINE_COMMIT_SHORT",
    "ALPHA_BASELINE_LABEL",
    "APPLICATION_VERSION",
    "METADATA_SCHEMA_VERSION",
    "PRODUCT_NAME",
    "RELEASE_CHANNEL",
    "build_program_metadata",
    "resolve_build_identity",
]
