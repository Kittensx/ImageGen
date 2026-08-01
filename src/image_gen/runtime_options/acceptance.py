from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .architecture import PHASE14K_ARCHITECTURE_FORMAT
from .profiles import (
    RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION,
    RuntimeMemoryProfile,
    builtin_runtime_profiles,
)

PHASE14K_ACCEPTANCE_FORMAT = "image-gen-phase14k-acceptance-v1"
PHASE14K_ACCEPTANCE_SCHEMA_VERSION = 1


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _truthy_all(values: Iterable[Any]) -> bool:
    items = list(values)
    return bool(items) and all(bool(value) for value in items)


def _criterion(
    criterion_id: str,
    description: str,
    *,
    status: str,
    evidence: Iterable[Any] = (),
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    normalized = str(status).strip().lower()
    if normalized not in {"pass", "fail", "pending"}:
        raise ValueError(f"Unsupported acceptance status: {status!r}")
    return {
        "criterion_id": criterion_id,
        "description": description,
        "status": normalized,
        "passed": normalized == "pass",
        "evidence": list(evidence),
        "notes": [str(item) for item in notes],
    }


def _combined_status(*conditions: bool | None) -> str:
    if any(value is False for value in conditions):
        return "fail"
    if any(value is None for value in conditions):
        return "pending"
    return "pass"


def _report_passed(report: Mapping[str, Any] | None) -> bool | None:
    if report is None:
        return None
    source = _mapping(report)
    if "passed" in source:
        return bool(source.get("passed"))
    summary = _mapping(source.get("summary"))
    if "passed" in summary:
        return bool(summary.get("passed"))
    return False


def _unit_target_packages_ready(unit_report: Mapping[str, Any] | None) -> bool | None:
    if unit_report is None:
        return None
    source = _mapping(unit_report)
    missing = {str(item) for item in _sequence(source.get("missing_target_packages"))}
    excluded = {str(item) for item in _sequence(source.get("excluded_tests"))}
    if not missing and not excluded:
        return True
    if bool(source.get("target_packages_required")):
        return False
    return None


def _real_cases(real_report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if real_report is None:
        return []
    return [_mapping(item) for item in _sequence(_mapping(real_report).get("cases"))]


def _cases_for(
    cases: Iterable[Mapping[str, Any]],
    *,
    hires: bool | None = None,
    configurations: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for raw in cases:
        case = _mapping(raw)
        target = _mapping(case.get("target"))
        if hires is not None and bool(target.get("hires")) is not hires:
            continue
        if configurations is not None and str(case.get("configuration")) not in configurations:
            continue
        selected.append(case)
    return selected


def _case_acceptance(case: Mapping[str, Any], key: str) -> bool:
    return bool(_mapping(_mapping(case).get("acceptance")).get(key))


def _runtime_profile_schema_evidence() -> tuple[bool, dict[str, Any]]:
    profiles = builtin_runtime_profiles()
    serialized = [profile.to_dict() for profile in profiles]
    round_tripped = [
        RuntimeMemoryProfile.from_mapping(item, require_complete=True).to_dict()
        for item in serialized
    ]
    valid = (
        RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION == 1
        and len(serialized) >= 4
        and serialized == round_tripped
        and all(int(item.get("schema_version", 0)) == 1 for item in serialized)
        and all(str(item.get("profile_id") or "") for item in serialized)
    )
    return valid, {
        "schema_version": RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION,
        "profile_ids": [item.get("profile_id") for item in serialized],
        "round_trip_stable": serialized == round_tripped,
    }


def build_phase14k_acceptance_report(
    *,
    project_root: str | Path,
    architecture_report: Mapping[str, Any] | None,
    unit_report: Mapping[str, Any] | None,
    oom_report: Mapping[str, Any] | None,
    real_environment_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    architecture = _mapping(architecture_report) if architecture_report is not None else None
    unit = _mapping(unit_report) if unit_report is not None else None
    oom = _mapping(oom_report) if oom_report is not None else None
    real = _mapping(real_environment_report) if real_environment_report is not None else None

    architecture_ok = (
        None
        if architecture is None
        else bool(architecture.get("passed"))
        and architecture.get("format") == PHASE14K_ARCHITECTURE_FORMAT
        and int(architecture.get("failed_check_count", 0) or 0) == 0
    )
    unit_ok = _report_passed(unit)
    target_packages_ready = _unit_target_packages_ready(unit)
    real_ok = _report_passed(real)
    oom_ok = _report_passed(oom)
    cases = _real_cases(real)

    reported_attention = (
        None
        if real is None
        else _truthy_all(
            bool(_mapping(case.get("attention")).get("requested_backend"))
            and bool(_mapping(case.get("attention")).get("effective_backend"))
            for case in cases
        )
    )
    backend_modes_work = (
        None
        if real is None
        else _truthy_all(_case_acceptance(case, "backend_matches_configuration") for case in cases)
    )

    hires_profile_cases = _cases_for(
        cases,
        hires=True,
        configurations={"xformers_low_vram", "xformers_maximum_fallback_armed"},
    )
    independent_hires = (
        None
        if real is None
        else _truthy_all(_case_acceptance(case, "independent_hires_profile") for case in hires_profile_cases)
    )

    low_vram_hires_cases = _cases_for(
        cases,
        hires=True,
        configurations={"xformers_low_vram", "xformers_maximum_fallback_armed"},
    )
    sequential_residency = (
        None
        if real is None
        else _truthy_all(
            _case_acceptance(case, "low_vram_hires_residency")
            for case in low_vram_hires_cases
        )
    )
    measurable_cleanup = (
        None
        if real is None
        else _truthy_all(
            _case_acceptance(case, "pre_hires_cleanup_measurable")
            for case in low_vram_hires_cases
        )
    )
    hires_preview = (
        None
        if real is None
        else _truthy_all(
            _case_acceptance(case, "hires_preview_suspension_recorded")
            for case in low_vram_hires_cases
        )
    )

    maximum_cases = _cases_for(
        cases,
        configurations={"xformers_maximum_fallback_armed"},
    )
    vae_truthful = (
        None
        if real is None
        else _truthy_all(
            _case_acceptance(case, "vae_controls_truthful") for case in maximum_cases
        )
    )

    runtime_records_complete = (
        None
        if real is None
        else _truthy_all(
            _case_acceptance(case, "runtime_metadata_and_replay_ready") for case in cases
        )
    )

    oom_cases = [_mapping(item) for item in _sequence((oom or {}).get("cases"))]
    oom_matrix_complete = None
    if oom is not None:
        matrix = _mapping(oom.get("matrix"))
        expected = int(matrix.get("expected_case_count", 0) or 0)
        oom_matrix_complete = (
            bool(oom_ok)
            and expected == 12
            and len(oom_cases) == expected
            and all(bool(case.get("passed")) for case in oom_cases)
            and all(
                bool(_mapping(case.get("checks")).get("exactly_one_retry"))
                and bool(_mapping(case.get("checks")).get("retry_succeeded"))
                and bool(_mapping(case.get("checks")).get("retry_duration_recorded"))
                and bool(_mapping(case.get("checks")).get("peak_vram_telemetry"))
                for case in oom_cases
            )
        )

    parity_included = None
    if unit is not None:
        command = [str(item) for item in _sequence(unit.get("command"))]
        parity_included = any("test_run_webui_parity_smoke.py" in item for item in command)

    schema_ok, schema_evidence = _runtime_profile_schema_evidence()

    criteria = [
        _criterion(
            "14K16-01-commandline-launchers",
            "COMMANDLINE_ARGS works consistently for CLI and WebUI launchers.",
            status=_combined_status(architecture_ok, unit_ok),
            evidence=[
                {"architecture_passed": architecture_ok},
                {"unit_integration_passed": unit_ok},
            ],
        ),
        _criterion(
            "14K16-02-convenience-flags",
            "--xformers, --sdpa, --medvram, and --lowvram work as documented.",
            status=_combined_status(unit_ok, target_packages_ready, real_ok, backend_modes_work),
            evidence=[
                {"unit_integration_passed": unit_ok},
                {"target_packages_ready": target_packages_ready},
                {"real_environment_passed": real_ok},
                {"backend_modes_work": backend_modes_work},
            ],
        ),
        _criterion(
            "14K16-03-explicit-backend-memory-forms",
            "Explicit attention-backend and memory-policy forms work.",
            status=_combined_status(unit_ok, real_ok, backend_modes_work),
            evidence=[
                {"unit_integration_passed": unit_ok},
                {"real_environment_passed": real_ok},
                {"backend_modes_work": backend_modes_work},
            ],
        ),
        _criterion(
            "14K16-04-mslk-preinitialization",
            "MSLK FMHA CLI values are applied before attention initialization.",
            status=_combined_status(unit_ok, target_packages_ready),
            evidence=[
                {"unit_integration_passed": unit_ok},
                {"target_package_tests_included": target_packages_ready},
            ],
        ),
        _criterion(
            "14K16-05-requested-effective-attention",
            "Requested and effective attention backends are both reported.",
            status=_combined_status(real_ok, reported_attention),
            evidence=[
                {"real_environment_passed": real_ok},
                {"all_cases_report_requested_and_effective": reported_attention},
            ],
        ),
        _criterion(
            "14K16-06-independent-hires-profile",
            "Hires can use an independently stronger memory profile.",
            status=_combined_status(real_ok, independent_hires),
            evidence=[
                {"qualifying_case_count": len(hires_profile_cases)},
                {"independent_hires_profile_verified": independent_hires},
            ],
        ),
        _criterion(
            "14K16-07-low-vram-hires-residency",
            "Low-VRAM hires sampling keeps text encoder and VAE off GPU during the UNet stage.",
            status=_combined_status(real_ok, sequential_residency),
            evidence=[
                {"qualifying_case_count": len(low_vram_hires_cases)},
                {"sequential_residency_verified": sequential_residency},
            ],
        ),
        _criterion(
            "14K16-08-pre-hires-cleanup-telemetry",
            "Pre-hires cleanup is measurable and recorded.",
            status=_combined_status(real_ok, measurable_cleanup),
            evidence=[
                {"qualifying_case_count": len(low_vram_hires_cases)},
                {"measurable_cleanup_verified": measurable_cleanup},
            ],
        ),
        _criterion(
            "14K16-09-hires-preview-suspension",
            "Preview can be suspended specifically for hires.",
            status=_combined_status(real_ok, hires_preview),
            evidence=[
                {"qualifying_case_count": len(low_vram_hires_cases)},
                {"hires_preview_suspension_verified": hires_preview},
            ],
        ),
        _criterion(
            "14K16-10-vae-controls",
            "VAE tiling and slicing controls are truthful and tested.",
            status=_combined_status(real_ok, vae_truthful),
            evidence=[
                {"qualifying_case_count": len(maximum_cases)},
                {"vae_controls_truthful": vae_truthful},
            ],
        ),
        _criterion(
            "14K16-11-bounded-oom-recovery",
            "OOM recovery is bounded and recorded.",
            status=_combined_status(oom_ok, oom_matrix_complete),
            evidence=[
                {"oom_qualification_passed": oom_ok},
                {"complete_12_case_matrix": oom_matrix_complete},
            ],
        ),
        _criterion(
            "14K16-12-runtime-replay-diagnostics",
            "Runtime settings appear in replay metadata and diagnostics.",
            status=_combined_status(unit_ok, real_ok, runtime_records_complete),
            evidence=[
                {"unit_integration_passed": unit_ok},
                {"runtime_records_complete": runtime_records_complete},
            ],
        ),
        _criterion(
            "14K16-13-no-argument-startup",
            "The normal no-argument startup behavior remains functional.",
            status=_combined_status(unit_ok),
            evidence=[{"unit_integration_passed": unit_ok}],
        ),
        _criterion(
            "14K16-14-base-generation-parity",
            "Existing base-generation parity tests continue to pass.",
            status=_combined_status(unit_ok, parity_included),
            evidence=[
                {"unit_integration_passed": unit_ok},
                {"parity_test_included": parity_included},
            ],
        ),
        _criterion(
            "14K16-15-runtime-profile-schema",
            "The runtime profile schema is stable enough for Phase 14L to consume.",
            status=_combined_status(unit_ok, schema_ok),
            evidence=[schema_evidence],
        ),
    ]

    failed = [item for item in criteria if item["status"] == "fail"]
    pending = [item for item in criteria if item["status"] == "pending"]
    passed = [item for item in criteria if item["status"] == "pass"]
    result = "fail" if failed else "pending" if pending else "pass"

    return {
        "format": PHASE14K_ACCEPTANCE_FORMAT,
        "schema_version": PHASE14K_ACCEPTANCE_SCHEMA_VERSION,
        "phase": "14K-16",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "result": result,
        "passed": result == "pass",
        "criterion_count": len(criteria),
        "passed_criterion_count": len(passed),
        "failed_criterion_count": len(failed),
        "pending_criterion_count": len(pending),
        "criteria": criteria,
        "source_reports": {
            "architecture": {
                "present": architecture is not None,
                "passed": architecture_ok,
                "format": (architecture or {}).get("format"),
            },
            "unit_integration": {
                "present": unit is not None,
                "passed": unit_ok,
                "missing_target_packages": list((unit or {}).get("missing_target_packages") or []),
                "excluded_tests": list((unit or {}).get("excluded_tests") or []),
            },
            "oom_qualification": {
                "present": oom is not None,
                "passed": oom_ok,
                "case_count": len(oom_cases),
            },
            "real_environment": {
                "present": real is not None,
                "passed": real_ok,
                "case_count": len(cases),
            },
        },
        "phase14l_handoff": {
            "runtime_profile_schema_version": RUNTIME_MEMORY_PROFILE_SCHEMA_VERSION,
            "runtime_profile_schema_verified": schema_ok,
            "acceptance_required_before_calibration": True,
        },
    }


__all__ = [
    "PHASE14K_ACCEPTANCE_FORMAT",
    "PHASE14K_ACCEPTANCE_SCHEMA_VERSION",
    "build_phase14k_acceptance_report",
]
