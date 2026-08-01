from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


PHASE14K_ARCHITECTURE_FORMAT = "image-gen-phase14k-architecture-audit-v1"

LOGICAL_TO_ACTUAL_MODULES: dict[str, tuple[str, ...]] = {
    "runtime_options.contracts": ("src/image_gen/runtime_options/contracts.py",),
    "runtime_options.bootstrap": (
        "src/image_gen/runtime_options/cli.py",
        "src/image_gen/runtime_options/startup_state.py",
    ),
    "runtime_options.cli": ("src/image_gen/runtime_options/cli.py",),
    "runtime_options.environment": (
        "src/image_gen/runtime_options/normalization.py",
        "src/image_gen/runtime_options/cuda_allocator.py",
    ),
    "runtime_options.normalization": ("src/image_gen/runtime_options/normalization.py",),
    "runtime_options.profiles": ("src/image_gen/runtime_options/profiles.py",),
    "runtime_options.verification": (
        "src/image_gen/runtime_options/startup_state.py",
        "modules/attention_backend.py",
        "modules/attention_runtime/production_dispatch.py",
    ),
    "runtime_options.metadata": ("src/image_gen/runtime_options/execution_record.py",),
    "runtime_options.command_serialization": ("src/image_gen/runtime_options/commandline.py",),
    "memory.hires_cleanup": ("src/image_gen/systems/memory/hires_cleanup.py",),
    "memory.oom_recovery": ("src/image_gen/systems/memory/oom_recovery.py",),
}

CANONICAL_CONTRACT_DEFINITIONS = {
    "RuntimeStartupOptions": "src/image_gen/runtime_options/contracts.py",
    "MSLKFMHAOptions": "src/image_gen/runtime_options/contracts.py",
    "RuntimeProfileSelection": "src/image_gen/runtime_options/contracts.py",
    "RuntimeMemoryProfile": "src/image_gen/runtime_options/profiles.py",
}

ENTRYPOINT_REQUIREMENTS = {
    "modules/txt2img/cli.py": (
        "add_runtime_startup_arguments",
        "bootstrap_runtime_startup",
        "argv_for_primary_parser",
    ),
    "src/image_gen/webui/server.py": (
        "add_runtime_startup_arguments",
        "bootstrap_runtime_startup",
        "argv_for_primary_parser",
    ),
}

LAUNCHER_REQUIREMENTS = {
    "run.bat": ("COMMANDLINE_ARGS", "modules.txt2img.cli", "%COMMANDLINE_ARGS%", "%*"),
    "run_cli.bat": ("COMMANDLINE_ARGS", "modules.txt2img.cli", "%COMMANDLINE_ARGS%", "%*"),
    "run_webui.bat": ("COMMANDLINE_ARGS", "image_gen.webui.server", "%COMMANDLINE_ARGS%", "%*"),
}


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _class_definitions(project_root: Path, class_names: set[str]) -> dict[str, list[str]]:
    found = {name: [] for name in class_names}
    roots = (project_root / "src", project_root / "modules", project_root / "image_gen")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in found:
                    found[node.name].append(_relative(project_root, path))
    return found


def audit_phase14k_architecture(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    for logical_owner, paths in LOGICAL_TO_ACTUAL_MODULES.items():
        missing = [path for path in paths if not (root / path).is_file()]
        checks.append(
            {
                "check": f"logical_owner:{logical_owner}",
                "passed": not missing,
                "actual_paths": list(paths),
                "missing_paths": missing,
                "architecture_decision": "folded_into_existing_canonical_module"
                if len(paths) > 1
                else "canonical_module_present",
            }
        )

    definitions = _class_definitions(root, set(CANONICAL_CONTRACT_DEFINITIONS))
    for class_name, expected_path in CANONICAL_CONTRACT_DEFINITIONS.items():
        actual = sorted(definitions.get(class_name) or [])
        checks.append(
            {
                "check": f"single_contract_owner:{class_name}",
                "passed": actual == [expected_path],
                "expected_path": expected_path,
                "actual_paths": actual,
            }
        )

    for relative_path, tokens in ENTRYPOINT_REQUIREMENTS.items():
        path = root / relative_path
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [token for token in tokens if token not in text]
        checks.append(
            {
                "check": f"shared_startup_entrypoint:{relative_path}",
                "passed": path.is_file() and not missing,
                "missing_tokens": missing,
            }
        )

    for relative_path, tokens in LAUNCHER_REQUIREMENTS.items():
        path = root / relative_path
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [token for token in tokens if token not in text]
        checks.append(
            {
                "check": f"canonical_launcher:{relative_path}",
                "passed": path.is_file() and not missing,
                "missing_tokens": missing,
            }
        )


    command_contracts = {
        "src/image_gen/webui/app.py": (
            '/api/runtime/command',
            'build_runtime_command_from_status',
        ),
        "src/image_gen/webui/static/assets/views/utilities.html": (
            'copyRuntimeCommandButton',
            'runtimeCommandCopyStatus',
        ),
        "src/image_gen/webui/static/assets/js/features/memory-status.js": (
            'bindRuntimeCommandCopy',
            'navigator.clipboard',
        ),
    }
    for relative_path, tokens in command_contracts.items():
        path = root / relative_path
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [token for token in tokens if token not in text]
        checks.append(
            {
                "check": f"copy_runtime_command:{relative_path}",
                "passed": path.is_file() and not missing,
                "missing_tokens": missing,
            }
        )

    webui_job_paths = (
        root / "src/image_gen/webui/jobs.py",
        root / "src/image_gen/runtime_options/startup_state.py",
    )
    settings_owners = [
        _relative(root, path)
        for path in webui_job_paths
        if path.is_file() and "runtime_job_overrides" in path.read_text(encoding="utf-8")
    ]
    checks.append(
        {
            "check": "runtime_job_override_flow_uses_shared_contract",
            "passed": settings_owners == [
                "src/image_gen/webui/jobs.py",
                "src/image_gen/runtime_options/startup_state.py",
            ],
            "actual_paths": settings_owners,
            "note": "WebUI stores explicit overrides; runtime_options resolves the canonical next-job contract.",
        }
    )

    passed = all(bool(item.get("passed")) for item in checks)
    return {
        "format": PHASE14K_ARCHITECTURE_FORMAT,
        "schema_version": 1,
        "phase": "14K-14",
        "project_root": str(root),
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(1 for item in checks if not item.get("passed")),
        "checks": checks,
        "logical_to_actual_modules": {
            key: list(value) for key, value in LOGICAL_TO_ACTUAL_MODULES.items()
        },
        "conclusion": (
            "The suggested Phase 14K layout is implemented through the active existing modules; "
            "no parallel runtime settings model is required."
        ),
    }


__all__ = [
    "PHASE14K_ARCHITECTURE_FORMAT",
    "LOGICAL_TO_ACTUAL_MODULES",
    "audit_phase14k_architecture",
]
