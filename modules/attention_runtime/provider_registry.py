from __future__ import annotations

import hashlib
import importlib.util
import re
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from modules.attention_runtime.release_reproducibility import (
    resolve_requirement_tree,
    verify_release_stack,
)


_TRACKED_PACKAGES = (
    "torch",
    "diffusers",
    "triton-windows",
    "mslk",
    "xformers",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _installed_version(name: str) -> str | None:
    candidates = [name]
    if name == "triton-windows":
        candidates.append("triton")
    for candidate in candidates:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    module_name = "triton" if name == "triton-windows" else name.replace("-", "_")
    try:
        if importlib.util.find_spec(module_name) is not None:
            return "installed-version-unavailable"
    except Exception:
        pass
    return None


def _expected_version(name: str, requirement: str) -> str | None:
    exact = re.match(rf"^{re.escape(name)}==([^;\s]+)$", requirement, flags=re.IGNORECASE)
    if exact:
        return exact.group(1)
    direct = re.match(rf"^{re.escape(name)}\s*@\s*(\S+)", requirement, flags=re.IGNORECASE)
    if not direct:
        return None
    parsed = urlparse(direct.group(1))
    filename = unquote(Path(parsed.path).name)
    wheel_match = re.match(rf"^{re.escape(name.replace('-', '_'))}-(.+?)-(?:py\d|cp\d)", filename, flags=re.IGNORECASE)
    if wheel_match:
        return wheel_match.group(1)
    wheel_match = re.match(rf"^{re.escape(name)}-(.+?)-(?:py\d|cp\d)", filename, flags=re.IGNORECASE)
    return wheel_match.group(1) if wheel_match else None


def load_environment_contract(lock_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(lock_path) if lock_path is not None else _project_root() / "requirements" / "requirements-lock.txt"
    raw = path.read_bytes()
    tree = resolve_requirement_tree(path)
    requirements: dict[str, str] = {}
    for line in tree["lines"]:
        name = re.split(r"\s*@\s*|==|>=|<=|~=|!=|>|<", line, maxsplit=1)[0].strip().lower()
        requirements[name] = line

    packages: dict[str, Any] = {}
    for name in _TRACKED_PACKAGES:
        requirement = requirements.get(name)
        expected = _expected_version(name, requirement) if requirement else None
        installed = _installed_version(name)
        packages[name] = {
            "requirement": requirement,
            "expected_version": expected,
            "installed_version": installed,
            "matches_expected": (
                installed == expected
                if installed is not None and expected is not None
                else None
            ),
        }
    try:
        relative_path = str(path.resolve().relative_to(_project_root().resolve()))
    except Exception:
        relative_path = str(path)
    return {
        "schema_version": 1,
        "authoritative": True,
        "path": relative_path.replace("\\", "/"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "resolved_files": [
            str(item.resolve().relative_to(_project_root().resolve())).replace("\\", "/")
            if item.resolve().is_relative_to(_project_root().resolve())
            else str(item.resolve())
            for item in tree["files"]
        ],
        "missing_includes": [str(item) for item in tree["missing"]],
        "include_cycles": [
            [str(item) for item in cycle]
            for cycle in tree["cycles"]
        ],
        "packages": packages,
    }


def build_provider_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "providers": [
            {
                "backend": "xformers",
                "processor": "diffusers.models.attention_processor.XFormersAttnProcessor",
                "kernel_provider": None,
                "execution_required_for_provider_claim": True,
            },
            {
                "backend": "sdpa",
                "processor": "diffusers.models.attention_processor.AttnProcessor2_0",
                "kernel_provider": "torch_sdpa",
                "execution_required_for_provider_claim": False,
            },
            {
                "backend": "eager",
                "processor": "diffusers.models.attention_processor.AttnProcessor",
                "kernel_provider": "torch_eager",
                "execution_required_for_provider_claim": False,
            },
        ],
        "environment_contract": load_environment_contract(),
        "release_reproducibility": verify_release_stack(),
    }
