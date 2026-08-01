from __future__ import annotations

import importlib.util
from contextlib import contextmanager
import os
from importlib import metadata
from typing import Any, Callable

from modules.attention_runtime import (
    EXPLICIT_BACKENDS,
    attention_layout_capture_report,
    backend_candidates,
    build_model_attention_signature,
    build_provider_registry,
    build_verified_xformers_processor,
    get_execution_evidence,
    module_device_dtype,
    require_verified_production_dispatch,
    run_xformers_layout_matrix,
    verified_layout_executor,
)


_ALLOWED_BACKENDS = {"auto", "default", "eager", "sdpa", "xformers"}
_EXPLICIT_BACKENDS = set(EXPLICIT_BACKENDS)


def resolve_attention_backend(value: str | None = None) -> str:
    """Return the canonical requested IMAGE_GEN attention backend."""

    selected = str(
        value
        if value is not None
        else os.environ.get("IMAGE_GEN_ATTENTION_BACKEND", "auto")
    ).strip().lower()
    aliases = {
        "vanilla": "eager",
        "classic": "eager",
        "math": "eager",
        "torch": "sdpa",
        "torch_sdpa": "sdpa",
        "memory_efficient": "xformers",
        "unchanged": "default",
    }
    selected = aliases.get(selected, selected)
    if selected not in _ALLOWED_BACKENDS:
        raise ValueError(
            "IMAGE_GEN_ATTENTION_BACKEND must be one of: "
            "auto, default, eager, sdpa, xformers."
        )
    return selected


def resolve_attention_slicing(value: str | None = None) -> str:
    selected = str(
        value
        if value is not None
        else os.environ.get("IMAGE_GEN_ATTENTION_SLICING", "off")
    ).strip().lower()
    if selected not in {"off", "auto", "max"}:
        raise ValueError("IMAGE_GEN_ATTENTION_SLICING must be one of: off, auto, max.")
    return selected


def _sliced_processor_verified(unet: Any) -> tuple[bool, list[str], list[str]]:
    names, modules = _processor_details(unet)
    lowered = [name.lower() for name in names]
    return bool(lowered and all("slicedattnprocessor" in name for name in lowered)), names, modules


def configure_unet_attention_slicing(
    unet: Any,
    *,
    mode: str | None = None,
    backend_report: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Configure and verify framework attention slicing without misreporting it.

    Diffusers implements slicing by replacing the active attention processors.
    Therefore a successful slicing request no longer executes xformers or SDPA;
    the report is updated to the verified sliced-eager processor reality.
    """

    selected = resolve_attention_slicing(mode)
    report = dict(backend_report or attention_backend_report(unet))
    before_names, before_modules = _processor_details(unet)
    slicing_report: dict[str, Any] = {
        "schema_version": 1,
        "requested": selected,
        "applied": False,
        "verified": selected == "off",
        "effective": "off",
        "processor_types_before": before_names,
        "processor_modules_before": before_modules,
        "processor_types_after": before_names,
        "processor_modules_after": before_modules,
        "replaced_backend": None,
        "reason": "Attention slicing disabled." if selected == "off" else None,
    }
    if selected == "off":
        report["attention_slicing"] = slicing_report
        setattr(unet, "_image_gen_attention_backend_report", dict(report))
        return report

    setter = getattr(unet, "set_attention_slice", None)
    if not callable(setter):
        message = (
            "Attention slicing was requested, but the active UNet does not expose "
            "set_attention_slice()."
        )
        slicing_report["reason"] = message
        report["attention_slicing"] = slicing_report
        if strict:
            raise RuntimeError(message)
        return report

    try:
        setter(selected)
        verified, after_names, after_modules = _sliced_processor_verified(unet)
        if not verified:
            raise RuntimeError(
                "set_attention_slice returned without attaching verified "
                f"SlicedAttnProcessor instances; active processors={after_names}."
            )
    except Exception as exc:
        slicing_report.update(
            {
                "reason": f"{type(exc).__name__}: {exc}",
                "processor_types_after": _processor_details(unet)[0],
                "processor_modules_after": _processor_details(unet)[1],
            }
        )
        report["attention_slicing"] = slicing_report
        if strict:
            raise RuntimeError(
                f"Unable to activate verified attention slicing mode {selected!r}: {exc}"
            ) from exc
        return report

    replaced = report.get("effective_backend")
    slicing_report.update(
        {
            "applied": True,
            "verified": True,
            "effective": selected,
            "processor_types_after": after_names,
            "processor_modules_after": after_modules,
            "replaced_backend": replaced,
            "reason": (
                "Diffusers attention slicing replaced the previously active attention "
                "processor family."
            ),
        }
    )
    report.update(
        {
            "effective_backend": "eager_sliced",
            "effective_processor": after_names,
            "processor_types_after": after_names,
            "processor_modules_after": after_modules,
            "effective_provider": "torch_eager_sliced",
            "effective_operator": None,
            "kernel_provider": "torch_eager_sliced",
            "operator_executed": False,
            "custom_provider_executed": False,
            "xformers_enablement_completed": False,
            "fallback_reason": (
                f"Attention slicing mode {selected} replaced the previously verified "
                f"{replaced} processor."
            ),
            "attention_slicing": slicing_report,
        }
    )
    setattr(unet, "_image_gen_attention_backend_report", dict(report))
    return report


@contextmanager
def temporary_attention_slicing(
    unet: Any,
    mode: str,
    *,
    strict: bool = False,
):
    """Temporarily replace processors for a memory-critical stage and restore them."""

    selected = resolve_attention_slicing(mode)
    if selected == "off":
        yield {
            "schema_version": 1,
            "requested": "off",
            "applied": False,
            "verified": True,
            "restored": True,
        }
        return

    processors = getattr(unet, "attn_processors", None)
    original_processors = dict(processors) if isinstance(processors, dict) else None
    original_report = dict(getattr(unet, "_image_gen_attention_backend_report", {}) or {})
    stage_report: dict[str, Any]
    try:
        configured = configure_unet_attention_slicing(
            unet,
            mode=selected,
            backend_report=original_report,
            strict=strict,
        )
        stage_report = dict(configured.get("attention_slicing") or {})
        stage_report["temporary"] = True
        yield stage_report
    finally:
        restored = False
        restore_error = None
        if original_processors is not None:
            setter = getattr(unet, "set_attn_processor", None)
            if callable(setter):
                try:
                    setter(original_processors)
                    restored = True
                except Exception as exc:  # pragma: no cover - model-specific
                    restore_error = f"{type(exc).__name__}: {exc}"
        setattr(unet, "_image_gen_attention_backend_report", original_report)
        if 'stage_report' in locals():
            stage_report["restored"] = restored
            stage_report["restore_error"] = restore_error



def _processor_details(unet: Any) -> tuple[list[str], list[str]]:
    processors = getattr(unet, "attn_processors", None)
    if not isinstance(processors, dict):
        return [], []
    names = sorted({type(processor).__name__ for processor in processors.values()})
    modules = sorted({type(processor).__module__ for processor in processors.values()})
    return names, modules


def _installed_version(*distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    for module_name in distribution_names:
        normalized = "triton" if module_name == "triton-windows" else module_name
        try:
            if importlib.util.find_spec(normalized) is not None:
                return "installed-version-unavailable"
        except Exception:
            continue
    return None


def _runtime_versions() -> dict[str, str | None]:
    return {
        "torch": _installed_version("torch"),
        "diffusers": _installed_version("diffusers"),
        "xformers": _installed_version("xformers"),
        "mslk": _installed_version("mslk"),
        "triton": _installed_version("triton-windows", "triton"),
    }


def _kernel_provider_evidence(
    backend: str | None,
    processor_names: list[str],
    processor_modules: list[str],
) -> tuple[str | None, list[str]]:
    if backend == "sdpa":
        return "torch_sdpa", ["The verified effective processor is PyTorch SDPA-backed."]
    if backend == "eager":
        return "torch_eager", ["The verified effective processor is Diffusers eager attention."]
    # An XFormersAttnProcessor only proves the public API/processor path. It does
    # not prove which MSLK/xFormers operator executed.
    if backend == "xformers":
        return None, [
            "An XFormers attention processor is attached, but no kernel provider is claimed until a real operator call records execution evidence."
        ]
    lowered = " ".join([*processor_names, *processor_modules]).lower()
    if "slicedattnprocessor" in lowered:
        return "torch_eager_sliced", [
            "Processor naming indicates the verified Diffusers sliced-attention family."
        ]
    if "attnprocessor2_0" in lowered or "sdpa" in lowered:
        return "torch_sdpa", ["Processor naming indicates the SDPA processor family."]
    return None, []


def _verify_processor(backend: str, processor_names: list[str]) -> bool:
    lowered = [name.lower() for name in processor_names]
    if not lowered:
        return False
    if backend == "xformers":
        return any("xformers" in name for name in lowered)
    if backend == "sdpa":
        return any("attnprocessor2_0" in name or "sdpa" in name for name in lowered)
    if backend == "eager":
        return any(
            "attnprocessor" in name
            and "xformers" not in name
            and "2_0" not in name
            and "sdpa" not in name
            for name in lowered
        )
    return False


def _set_standard_processor(
    unet: Any,
    backend: str,
    *,
    processor_factory: Callable[[str], Any] | None,
) -> None:
    setter = getattr(unet, "set_attn_processor", None)
    if not callable(setter):
        raise RuntimeError(
            f"UNet does not expose set_attn_processor for {backend} attention."
        )
    if processor_factory is not None:
        processor = processor_factory(backend)
    elif backend == "eager":
        from diffusers.models.attention_processor import AttnProcessor

        processor = AttnProcessor()
    elif backend == "sdpa":
        from diffusers.models.attention_processor import AttnProcessor2_0

        processor = AttnProcessor2_0()
    else:
        raise ValueError(f"Unsupported standard processor backend: {backend!r}.")
    setter(processor)


def _normalize_injected_xformers_test(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    passed = bool(normalized.get("passed", True))
    return {
        "schema_version": 1,
        "test_kind": "injected_layout_validation",
        "passed": passed,
        "layout_count": int(normalized.get("layout_count", 1)),
        "failure_count": 0 if passed else 1,
        "results": [normalized],
    }


def _matrix_failure_summary(matrix: dict[str, Any]) -> str:
    failures = [item for item in matrix.get("results", []) if not item.get("passed")]
    if not failures:
        return "xFormers model-layout validation did not pass."
    pieces: list[str] = []
    for failure in failures[:8]:
        pieces.append(
            "kind={kind} heads={heads} q/k/v={q}/{k}/{v}: {error}".format(
                kind=failure.get("attention_kind"),
                heads=failure.get("heads"),
                q=failure.get("q_head_dim"),
                k=failure.get("k_head_dim"),
                v=failure.get("v_head_dim"),
                error=failure.get("error") or "unsupported",
            )
        )
    return "; ".join(pieces)


def _activate_xformers_compat(
    unet: Any,
    *,
    model_signature: dict[str, Any],
    xformers_processor_factory: Callable[[], Any] | None,
    xformers_smoke_test: Callable[[Any], dict[str, Any]] | None,
    xformers_layout_executor: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Validate projection-derived layouts and then attach xFormers.

    Diffusers' generic enable helper probes a synthetic float32/K=40 shape before
    IMAGE_GEN has historically placed the UNet. Phase 14K-2.2 intentionally does
    not use that helper for a real model. Validation is performed against the
    loaded UNet's projection-derived layouts on its final device and dtype.
    """

    details: dict[str, Any] = {
        "compatibility_path": None,
        "diffusers_enable_skipped": True,
        "diffusers_enable_skip_reason": (
            "IMAGE_GEN uses projection-derived model-layout validation instead of Diffusers' generic synthetic probe."
        ),
        "manual_processor_attached": False,
        "compatibility_matrix": None,
        "smoke_test": None,
        "production_dispatch": None,
    }

    placement = module_device_dtype(unet)
    if (
        int(model_signature.get("attention_module_count") or 0) == 0
        or xformers_smoke_test is not None
    ):
        production_dispatch = {
            "schema_version": 1,
            "verified": True,
            "test_or_nonstandard_unet_bypass": True,
            "rejection_reasons": [],
        }
    else:
        production_dispatch = require_verified_production_dispatch(
            model_signature, validation_dtype=str(placement["dtype"])
        )
    details["production_dispatch"] = production_dispatch

    injected_smoke: dict[str, Any] | None = None
    if xformers_smoke_test is not None:
        injected_smoke = dict(xformers_smoke_test(unet))
        matrix = _normalize_injected_xformers_test(injected_smoke)
    elif int(model_signature.get("attention_module_count") or 0) > 0:
        matrix = run_xformers_layout_matrix(
            unet,
            model_signature,
            executor=xformers_layout_executor or verified_layout_executor,
        )
    else:
        # Test doubles and unusual non-Diffusers UNets may not expose modules.
        # In that narrow case, permit the object's own helper as a compatibility
        # fallback. Real Diffusers UNets always take the model-layout path above.
        enabler = getattr(unet, "enable_xformers_memory_efficient_attention", None)
        if not callable(enabler):
            raise RuntimeError(
                "No projection-derived attention layouts were found and the UNet exposes no xFormers helper."
            )
        enabler()
        names, _modules = _processor_details(unet)
        if not _verify_processor("xformers", names):
            raise RuntimeError(
                "The fallback xFormers helper returned without attaching an XFormers processor."
            )
        details["compatibility_path"] = "unet_helper_without_discoverable_layouts"
        details["diffusers_enable_skipped"] = False
        details["diffusers_enable_skip_reason"] = None
        details["compatibility_matrix"] = {
            "schema_version": 1,
            "test_kind": "unet_helper_without_discoverable_layouts",
            "passed": True,
            "layout_count": 0,
            "failure_count": 0,
            "results": [],
        }
        return details

    details["compatibility_matrix"] = matrix
    details["smoke_test"] = injected_smoke if injected_smoke is not None else matrix
    if not bool(matrix.get("passed")):
        raise RuntimeError(
            "xFormers projection-derived layout validation failed: "
            + _matrix_failure_summary(matrix)
        )

    setter = getattr(unet, "set_attn_processor", None)
    if not callable(setter):
        raise RuntimeError("UNet does not expose set_attn_processor for xFormers activation.")
    if xformers_processor_factory is None:
        processor = build_verified_xformers_processor()
    else:
        processor = xformers_processor_factory()
    setter(processor)
    details["manual_processor_attached"] = True
    details["compatibility_path"] = "projection_layout_matrix_then_manual_processor"

    names, _modules = _processor_details(unet)
    if not _verify_processor("xformers", names):
        raise RuntimeError(
            "Model-layout validation passed, but IMAGE_GEN did not attach an XFormers attention processor. "
            f"Active processors: {names or ['<unreported>']}"
        )
    return details


def _explicit_failure_message(backend: str, exc: BaseException) -> str:
    alternatives = {
        "xformers": (
            "Review the model attention signature and compatibility matrix, verify the active environment contains the locked custom xformers/MSLK build, or use --sdpa / --eager-attention."
        ),
        "sdpa": (
            "Verify the installed PyTorch and Diffusers versions support SDPA, or use --xformers / --eager-attention."
        ),
        "eager": (
            "Verify the UNet exposes Diffusers attention processors, or use --attention-backend auto."
        ),
    }
    guidance = alternatives.get(
        backend,
        "Use --attention-backend auto to permit a verified fallback.",
    )
    return (
        f"Requested {backend} attention could not be activated and verified. "
        f"{guidance} Original error: {type(exc).__name__}: {exc}"
    )


def configure_unet_attention(
    unet: Any,
    *,
    backend: str | None = None,
    processor_factory: Callable[[str], Any] | None = None,
    xformers_processor_factory: Callable[[], Any] | None = None,
    xformers_smoke_test: Callable[[Any], dict[str, Any]] | None = None,
    xformers_layout_executor: Callable[..., Any] | None = None,
    model_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure and verify one Diffusers UNet attention implementation."""

    requested = resolve_attention_backend(backend)
    signature = model_signature or build_model_attention_signature(unet)
    setattr(unet, "_image_gen_model_attention_signature", dict(signature))
    placement = module_device_dtype(unet)
    before_names, before_modules = _processor_details(unet)
    candidates = backend_candidates(requested)
    attempts: list[dict[str, Any]] = []
    effective_backend: str | None = None
    after_names = list(before_names)
    after_modules = list(before_modules)
    xformers_enablement_completed = False
    xformers_compatibility: dict[str, Any] | None = None

    for candidate in candidates:
        try:
            compatibility_details = None
            if candidate in {"eager", "sdpa"}:
                _set_standard_processor(
                    unet,
                    candidate,
                    processor_factory=processor_factory,
                )
            elif candidate == "xformers":
                compatibility_details = _activate_xformers_compat(
                    unet,
                    model_signature=signature,
                    xformers_processor_factory=xformers_processor_factory,
                    xformers_smoke_test=xformers_smoke_test,
                    xformers_layout_executor=xformers_layout_executor,
                )
            else:
                raise ValueError(
                    f"Unsupported attention backend activation request: {candidate!r}."
                )

            after_names, after_modules = _processor_details(unet)
            verified = _verify_processor(candidate, after_names)
            if not verified:
                raise RuntimeError(
                    "Activation returned without attaching the expected attention processor. "
                    f"Active processors: {after_names or ['<unreported>']}"
                )
            effective_backend = candidate
            xformers_enablement_completed = candidate == "xformers"
            if candidate == "xformers":
                xformers_compatibility = compatibility_details
            attempts.append(
                {
                    "backend": candidate,
                    "activated": True,
                    "verified": True,
                    "error": None,
                    "compatibility_path": (
                        compatibility_details.get("compatibility_path")
                        if isinstance(compatibility_details, dict)
                        else None
                    ),
                }
            )
            break
        except Exception as exc:
            attempts.append(
                {
                    "backend": candidate,
                    "activated": False,
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "compatibility_path": None,
                }
            )
            if requested in _EXPLICIT_BACKENDS:
                raise RuntimeError(_explicit_failure_message(candidate, exc)) from exc

    if effective_backend is None:
        attempt_summary = "; ".join(
            f"{item['backend']}: {item['error']}" for item in attempts
        )
        raise RuntimeError(
            "IMAGE_GEN could not activate any supported attention backend. "
            f"Attempts: {attempt_summary}"
        )

    provider, provider_evidence = _kernel_provider_evidence(
        effective_backend,
        after_names,
        after_modules,
    )
    failed_attempts = [item for item in attempts if not item["verified"]]
    fallback_reason = None
    if failed_attempts:
        failed_names = ", ".join(item["backend"] for item in failed_attempts)
        fallback_reason = (
            f"Automatic selection used {effective_backend} after {failed_names} failed verification."
        )

    provider_registry = build_provider_registry()
    execution_evidence = get_execution_evidence()
    mslk_runtime_configuration = dict(
        execution_evidence.get("mslk_runtime_configuration") or {}
    )
    compatibility_matrix = (
        dict(xformers_compatibility.get("compatibility_matrix") or {})
        if isinstance(xformers_compatibility, dict)
        else {}
    )
    rejected_layouts = [
        dict(item)
        for item in compatibility_matrix.get("results", [])
        if not item.get("passed")
    ]
    release_reproducibility = provider_registry.get("release_reproducibility")
    capability_profile = (
        dict((release_reproducibility or {}).get("mslk_profile") or {})
        if isinstance(release_reproducibility, dict)
        else {}
    )
    report = {
        "schema_version": 2,
        "requested_backend": requested,
        "effective_backend": effective_backend,
        "effective_processor": after_names,
        "explicit_request": requested in _EXPLICIT_BACKENDS,
        "automatic_selection": requested in {"auto", "default"},
        "applied": True,
        "verified": True,
        "expected_processor_attached": True,
        "processor_types_before": before_names,
        "processor_modules_before": before_modules,
        "processor_types_after": after_names,
        "processor_modules_after": after_modules,
        "activation_attempts": attempts,
        "fallback_reason": fallback_reason,
        "xformers_enablement_completed": xformers_enablement_completed,
        "xformers_compatibility": xformers_compatibility,
        "xformers": xformers_compatibility,
        "production_dispatch": (
            xformers_compatibility.get("production_dispatch")
            if isinstance(xformers_compatibility, dict)
            else None
        ),
        "kernel_provider": (None if effective_backend == "xformers" else provider),
        "effective_provider": (
            "mslk_triton_splitk" if effective_backend == "xformers" else provider
        ),
        "effective_operator": (
            "triton_splitKF" if effective_backend == "xformers" else None
        ),
        "operator_executed": False,
        "validated_layout_count": int(compatibility_matrix.get("layout_count") or 0),
        "rejected_layouts": rejected_layouts,
        "capability_profile": {
            "profile_id": capability_profile.get("profile_id"),
            "profile_revision": capability_profile.get("profile_revision"),
            "profile_sha256": capability_profile.get("profile_sha256"),
            "profile_path": capability_profile.get("profile_path"),
            "valid": capability_profile.get("valid"),
        },
        "kernel_provider_evidence": provider_evidence,
        "custom_provider_execution": execution_evidence,
        "custom_provider_executed": False,
        "mslk_startup_configuration": mslk_runtime_configuration.get("startup"),
        "mslk_kernel_first_use": mslk_runtime_configuration.get("first_use"),
        "versions": _runtime_versions(),
        "environment_contract": provider_registry["environment_contract"],
        "release_reproducibility": release_reproducibility,
        "published_wheel_provenance": (
            dict((release_reproducibility or {}).get("packages") or {})
            if isinstance(release_reproducibility, dict)
            else {}
        ),
        "provider_registry": provider_registry,
        "model_attention_signature": signature,
        "runtime_attention_capture": attention_layout_capture_report(unet),
        "validation_device": placement["device"],
        "validation_dtype": placement["dtype"],
        "environment_variable": "IMAGE_GEN_ATTENTION_BACKEND",
        "xformers_installation_changed": False,
    }
    setattr(unet, "_image_gen_attention_backend_report", dict(report))
    return report


def attention_backend_report(unet: Any) -> dict[str, Any]:
    existing = getattr(unet, "_image_gen_attention_backend_report", None)
    if isinstance(existing, dict):
        report = dict(existing)
        signature = getattr(unet, "_image_gen_model_attention_signature", None)
        if isinstance(signature, dict):
            report["model_attention_signature"] = dict(signature)
        report["runtime_attention_capture"] = attention_layout_capture_report(unet)
        execution = get_execution_evidence()
        report["custom_provider_execution"] = execution
        mslk_runtime_configuration = dict(
            execution.get("mslk_runtime_configuration") or {}
        )
        report["mslk_startup_configuration"] = mslk_runtime_configuration.get(
            "startup"
        )
        report["mslk_kernel_first_use"] = mslk_runtime_configuration.get(
            "first_use"
        )
        if report.get("effective_backend") == "xformers":
            report["effective_provider"] = "mslk_triton_splitk"
            report["effective_operator"] = "triton_splitKF"
            report["custom_provider_executed"] = bool(execution.get("executed"))
            report["operator_executed"] = bool(execution.get("executed"))
            report["kernel_provider"] = (
                "mslk_triton_splitk" if execution.get("executed") else None
            )
        return report
    processor_names, processor_modules = _processor_details(unet)
    provider, provider_evidence = _kernel_provider_evidence(
        None,
        processor_names,
        processor_modules,
    )
    provider_registry = build_provider_registry()
    return {
        "schema_version": 2,
        "requested_backend": "unreported",
        "effective_backend": "unverified",
        "effective_processor": processor_names,
        "explicit_request": False,
        "automatic_selection": False,
        "applied": False,
        "verified": False,
        "expected_processor_attached": False,
        "processor_types_before": [],
        "processor_modules_before": [],
        "processor_types_after": processor_names,
        "processor_modules_after": processor_modules,
        "activation_attempts": [],
        "fallback_reason": "No verified attention backend report was attached to this UNet.",
        "xformers_enablement_completed": False,
        "xformers_compatibility": None,
        "xformers": None,
        "kernel_provider": provider,
        "effective_provider": provider,
        "effective_operator": None,
        "operator_executed": False,
        "validated_layout_count": 0,
        "rejected_layouts": [],
        "capability_profile": {},
        "kernel_provider_evidence": provider_evidence,
        "custom_provider_execution": get_execution_evidence(),
        "custom_provider_executed": False,
        "versions": _runtime_versions(),
        "environment_contract": provider_registry["environment_contract"],
        "release_reproducibility": provider_registry.get("release_reproducibility"),
        "published_wheel_provenance": dict(
            (provider_registry.get("release_reproducibility") or {}).get("packages") or {}
        ),
        "provider_registry": provider_registry,
        "model_attention_signature": getattr(unet, "_image_gen_model_attention_signature", None),
        "runtime_attention_capture": attention_layout_capture_report(unet),
        "validation_device": module_device_dtype(unet)["device"],
        "validation_dtype": module_device_dtype(unet)["dtype"],
        "environment_variable": "IMAGE_GEN_ATTENTION_BACKEND",
        "xformers_installation_changed": False,
    }
