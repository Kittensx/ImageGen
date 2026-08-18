from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .profiles import runtime_profile_json_schema


PHASE14K_DELIVERY_FORMAT = "image-gen-phase14k-delivery-v1"
PHASE14K_DELIVERY_SCHEMA_VERSION = 1
PHASE14K_TARGET_VRAM_FORMAT = "image-gen-phase14k-target-vram-v1"
PHASE14K_TARGET_VRAM_SCHEMA_VERSION = 1

_REQUIRED_EXAMPLES = {
    "balanced_generation": "docs/examples/runtime_memory/phase14k13_balanced_webui.bat",
    "safe_hires_generation": "docs/examples/runtime_memory/phase14k13_safe_hires.bat",
    "mslk_policy_testing": "docs/examples/runtime_memory/phase14k13_mslk_policy_test.bat",
}
_REQUIRED_DOCUMENTATION = {
    "command_line_help": "docs/reference/RUNTIME_MEMORY_COMMAND_LINE.md",
    "runtime_profile_schema": "docs/reference/RUNTIME_MEMORY_PROFILE_SCHEMA.md",
    "runtime_profile_json_schema": "docs/reference/runtime_memory_profile.schema.json",
    "target_vram_diagnostics": "docs/testing/phase14k17_target_vram_baseline.json",
    "target_vram_diagnostics_reference": "docs/testing/PHASE_14K-17_TARGET_VRAM_DIAGNOSTICS.md",
}
_REQUIRED_VALIDATION_FILES = (
    "testing/tests/runtime/test_phase14k17_delivery_requirements.py",
    "testing/test_validations/historical/runtime/phase14k17_delivery.py",
    "testing/test_validations/historical/runtime/phase14k17_static_delivery_validation.bat",
    "testing/test_validations/historical/runtime/phase14k17_delivery_from_reports.bat",
    "testing/test_validations/historical/runtime/phase14k17_all_validation.bat",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mib(value: Any) -> float:
    try:
        return round(int(value or 0) / (1024 * 1024), 3)
    except (TypeError, ValueError):
        return 0.0


def _recovery_snapshot(case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = _mapping(case.get("oom_recovery"))
    for attempt in _sequence(recovery.get("attempts")):
        attempt_map = _mapping(attempt)
        for action in _sequence(attempt_map.get("actions")):
            action_map = _mapping(action)
            if action_map.get("action") != "recovery_memory_snapshot":
                continue
            before = _mapping(_mapping(action_map.get("before")).get("cuda"))
            after = _mapping(_mapping(action_map.get("after")).get("cuda"))
            return before, after
    return {}, {}


def summarize_phase14k_target_vram(
    *,
    real_environment_report: Mapping[str, Any],
    oom_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact, durable target-environment baseline from Phase 14K reports."""

    real = _mapping(real_environment_report)
    oom = _mapping(oom_report)
    real_cases = [_mapping(item) for item in _sequence(real.get("cases"))]
    oom_cases = [_mapping(item) for item in _sequence(oom.get("cases"))]

    generation_cases: list[dict[str, Any]] = []
    for case in real_cases:
        target = _mapping(case.get("target"))
        attention = _mapping(case.get("attention"))
        generation_cases.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "configuration": str(case.get("configuration") or ""),
                "target_id": str(target.get("id") or ""),
                "hires": bool(target.get("hires")),
                "passed": bool(case.get("passed")),
                "return_code": case.get("return_code"),
                "generation_time_sec": case.get("generation_time_sec"),
                "requested_backend": attention.get("requested_backend"),
                "effective_backend": attention.get("effective_backend"),
                "verified_kernel_provider": attention.get("verified_kernel_provider"),
                "peak_allocated_vram_bytes": int(case.get("peak_allocated_vram_bytes") or 0),
                "peak_allocated_vram_mib": _mib(case.get("peak_allocated_vram_bytes")),
                "peak_reserved_vram_bytes": int(case.get("peak_reserved_vram_bytes") or 0),
                "peak_reserved_vram_mib": _mib(case.get("peak_reserved_vram_bytes")),
                "image_error": case.get("image_error"),
            }
        )

    recovery_cases: list[dict[str, Any]] = []
    for case in oom_cases:
        before, after = _recovery_snapshot(case)
        before_allocated = int(before.get("allocated_vram_bytes") or 0)
        after_allocated = int(after.get("allocated_vram_bytes") or 0)
        before_reserved = int(before.get("reserved_vram_bytes") or 0)
        after_reserved = int(after.get("reserved_vram_bytes") or 0)
        recovery_cases.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "failure_stage": str(case.get("failure_stage") or ""),
                "recovery_profile": str(case.get("recovery_profile") or ""),
                "passed": bool(case.get("passed")),
                "retry_duration_ms": case.get("retry_duration_ms"),
                "peak_allocated_bytes": int(case.get("peak_allocated_bytes") or 0),
                "peak_reserved_bytes": int(case.get("peak_reserved_bytes") or 0),
                "before": {
                    "allocated_vram_bytes": before_allocated,
                    "allocated_vram_mib": _mib(before_allocated),
                    "reserved_vram_bytes": before_reserved,
                    "reserved_vram_mib": _mib(before_reserved),
                },
                "after": {
                    "allocated_vram_bytes": after_allocated,
                    "allocated_vram_mib": _mib(after_allocated),
                    "reserved_vram_bytes": after_reserved,
                    "reserved_vram_mib": _mib(after_reserved),
                },
                "released": {
                    "allocated_vram_bytes": max(before_allocated - after_allocated, 0),
                    "reserved_vram_bytes": max(before_reserved - after_reserved, 0),
                },
            }
        )

    base_cases = [item for item in generation_cases if not item["hires"]]
    hires_cases = [item for item in generation_cases if item["hires"]]
    device = _mapping(real.get("device")) or _mapping(oom.get("device"))
    return {
        "format": PHASE14K_TARGET_VRAM_FORMAT,
        "schema_version": PHASE14K_TARGET_VRAM_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_created_at_utc": {
            "real_environment": real.get("created_at_utc"),
            "oom_qualification": oom.get("created_at_utc"),
        },
        "device": device,
        "generation_summary": {
            "case_count": len(generation_cases),
            "passed_case_count": sum(1 for item in generation_cases if item["passed"]),
            "failed_case_count": sum(1 for item in generation_cases if not item["passed"]),
            "base_case_count": len(base_cases),
            "base_passed_case_count": sum(1 for item in base_cases if item["passed"]),
            "hires_case_count": len(hires_cases),
            "hires_passed_case_count": sum(1 for item in hires_cases if item["passed"]),
        },
        "generation_cases": generation_cases,
        "oom_recovery_summary": {
            "case_count": len(recovery_cases),
            "passed_case_count": sum(1 for item in recovery_cases if item["passed"]),
            "all_cases_include_before_after": bool(recovery_cases)
            and all(item["before"] and item["after"] for item in recovery_cases),
        },
        "oom_recovery_cases": recovery_cases,
        "qualification_interpretation": {
            "architecture_and_bounded_recovery_validated": bool(recovery_cases)
            and all(item["passed"] for item in recovery_cases),
            "hires_final_acceptance": False,
            "hires_status": "pre-14M/14N baseline",
            "note": (
                "Failed hires cases are retained as target capacity evidence and are not "
                "reclassified as passes. Repeat this matrix after the Phase 14M and 14N "
                "hires pipeline changes."
            ),
        },
    }


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, **evidence: Any) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "evidence": evidence,
        }
    )


def build_phase14k_delivery_report(
    *,
    project_root: str | Path,
    target_vram_report: Mapping[str, Any] | None,
    architecture_report: Mapping[str, Any] | None = None,
    unit_report: Mapping[str, Any] | None = None,
    acceptance_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the Phase 14K-17 delivery package without rewriting acceptance history."""

    root = Path(project_root).expanduser().resolve()
    target = _mapping(target_vram_report)
    architecture = _mapping(architecture_report)
    unit = _mapping(unit_report)
    acceptance = _mapping(acceptance_report)
    checks: list[dict[str, Any]] = []

    delta_manifest_paths = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.name == "_delta_manifests"
        or (path.is_file() and "delta" in path.name.lower() and "manifest" in path.name.lower())
    ]
    _check(
        checks,
        "changed-files-only-no-delta-manifests",
        not delta_manifest_paths,
        discovered_paths=delta_manifest_paths,
    )

    cli_path = root / "modules/txt2img/cli.py"
    runtime_cli_path = root / "src/image_gen/runtime_options/cli.py"
    cli_text = cli_path.read_text(encoding="utf-8") if cli_path.is_file() else ""
    runtime_cli_text = (
        runtime_cli_path.read_text(encoding="utf-8") if runtime_cli_path.is_file() else ""
    )
    help_tokens = (
        "COMMANDLINE_ARGS",
        "--runtime-profile",
        "--xformers",
        "--sdpa",
        "--medvram",
        "--lowvram",
        "--hires-memory-profile",
        "--oom-retry-profile",
        "--mslk-fmha-policy",
    )
    missing_help_tokens = [
        token for token in help_tokens if token not in cli_text and token not in runtime_cli_text
    ]
    _check(
        checks,
        "updated-command-line-help",
        not missing_help_tokens
        and "RUNTIME_MEMORY_COMMAND_LINE.md" in (cli_text + runtime_cli_text),
        missing_tokens=missing_help_tokens,
    )

    for name, relative in _REQUIRED_DOCUMENTATION.items():
        path = root / relative
        _check(checks, f"documentation:{name}", path.is_file(), path=relative)

    schema_path = root / _REQUIRED_DOCUMENTATION["runtime_profile_json_schema"]
    schema_matches = False
    schema_error: str | None = None
    if schema_path.is_file():
        try:
            schema_matches = json.loads(schema_path.read_text(encoding="utf-8")) == runtime_profile_json_schema()
        except (OSError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
    _check(
        checks,
        "runtime-profile-schema-matches-runtime",
        schema_matches,
        path=str(schema_path.relative_to(root)),
        error=schema_error,
    )

    for name, relative in _REQUIRED_EXAMPLES.items():
        path = root / relative
        source = path.read_text(encoding="utf-8") if path.is_file() else ""
        canonical_launcher = "run_webui.bat" if name == "balanced_generation" else "run.bat"
        _check(
            checks,
            f"example:{name}",
            path.is_file() and canonical_launcher in source and "COMMANDLINE_ARGS" in source,
            path=relative,
            canonical_launcher=canonical_launcher,
        )

    missing_validation = [relative for relative in _REQUIRED_VALIDATION_FILES if not (root / relative).is_file()]
    _check(
        checks,
        "unit-and-integration-tests-included",
        not missing_validation,
        missing_files=missing_validation,
        supplied_unit_report=bool(unit),
        supplied_unit_passed=unit.get("passed") if unit else None,
        supplied_unit_counts=_mapping(unit.get("counts")) if unit else {},
    )

    target_valid = (
        target.get("format") == PHASE14K_TARGET_VRAM_FORMAT
        and int(target.get("schema_version") or 0) == PHASE14K_TARGET_VRAM_SCHEMA_VERSION
        and int(_mapping(target.get("generation_summary")).get("case_count") or 0) > 0
        and int(_mapping(target.get("oom_recovery_summary")).get("case_count") or 0) == 12
        and bool(_mapping(target.get("oom_recovery_summary")).get("all_cases_include_before_after"))
    )
    _check(
        checks,
        "target-before-after-vram-diagnostics",
        target_valid,
        generation_summary=_mapping(target.get("generation_summary")),
        oom_recovery_summary=_mapping(target.get("oom_recovery_summary")),
    )

    all_delivery_checks_passed = all(item["passed"] for item in checks)
    acceptance_result = str(acceptance.get("result") or "not_supplied")
    unit_passed = unit.get("passed") if unit else None
    if all_delivery_checks_passed and acceptance_result == "pass" and unit_passed is True:
        validation_state = "accepted"
    elif all_delivery_checks_passed:
        validation_state = "delivery_complete_with_deferred_validation"
    else:
        validation_state = "delivery_incomplete"

    return {
        "format": PHASE14K_DELIVERY_FORMAT,
        "schema_version": PHASE14K_DELIVERY_SCHEMA_VERSION,
        "phase": "14K-17",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "delivery_complete": all_delivery_checks_passed,
        "validation_state": validation_state,
        "check_count": len(checks),
        "passed_check_count": sum(1 for item in checks if item["passed"]),
        "failed_check_count": sum(1 for item in checks if not item["passed"]),
        "checks": checks,
        "supplied_validation": {
            "architecture_present": bool(architecture),
            "architecture_passed": architecture.get("passed") if architecture else None,
            "unit_report_present": bool(unit),
            "unit_report_passed": unit_passed,
            "unit_counts": _mapping(unit.get("counts")) if unit else {},
            "phase14k16_acceptance_present": bool(acceptance),
            "phase14k16_acceptance_result": acceptance_result,
            "phase14k16_passed_criteria": acceptance.get("passed_criterion_count") if acceptance else None,
            "phase14k16_failed_criteria": acceptance.get("failed_criterion_count") if acceptance else None,
        },
        "known_deferred_work": {
            "hires_pipeline": "Repeat target hires qualification after Phase 14M and Phase 14N.",
            "phase14l": (
                "GPU Runtime Autotuner and Memory Profile Calibrator remains the next Phase 14K "
                "handoff, but target calibration should use the post-14M/14N hires pipeline."
            ),
        },
    }
