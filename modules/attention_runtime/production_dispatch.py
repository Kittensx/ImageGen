from __future__ import annotations

import os
from threading import RLock
from typing import Any

from modules.attention_runtime.release_reproducibility import verify_release_stack

_REQUIRED_PROVIDER = "mslk_triton_splitk"
_REQUIRED_OPERATOR = "triton_splitKF"
_EXECUTION_LOCK = RLock()
_EXECUTION_STATE: dict[str, Any] = {
    "schema_version": 1,
    "executed": False,
    "successful_call_count": 0,
    "failed_call_count": 0,
    "logical_head_dimensions": [],
    "last_success": None,
    "last_failure": None,
}


def hardware_qualification_mode() -> str:
    value = str(os.environ.get("IMAGE_GEN_HARDWARE_QUALIFICATION") or "").strip().lower()
    if value in {"validated", "validated_reference", "official"}:
        return "validated"
    if value in {"community", "community_unverified", "unverified"}:
        return "community_unverified"
    return "validated"


def prepare_mslk_process_environment() -> dict[str, Any]:
    """Apply the launch policy appropriate to the hardware qualification level.

    SM120 keeps the published blackwell_safe policy. Community hardware defaults
    to MSLK's capability-oriented auto policy and is allowed to prove support by
    executing IMAGE_GEN's real projection-derived layout matrix.
    """

    qualification = hardware_qualification_mode()
    previous = os.environ.get("MSLK_FMHA_POLICY")
    default_policy = "blackwell_safe" if qualification == "validated" else "auto"
    os.environ.setdefault("MSLK_FMHA_POLICY", default_policy)
    return {
        "MSLK_FMHA_POLICY": os.environ.get("MSLK_FMHA_POLICY"),
        "IMAGE_GEN_HARDWARE_QUALIFICATION": qualification,
        "policy_was_preexisting": previous is not None,
        "policy_changed": previous is None,
    }


def _mslk_runtime_configuration() -> dict[str, Any]:
    try:
        from mslk.attention.fmha.fmha_tuning_policy import (
            get_fmha_environment_snapshot,
            get_fmha_first_use_evidence,
        )

        return {
            "startup": get_fmha_environment_snapshot(),
            "first_use": get_fmha_first_use_evidence(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _model_head_dimensions(signature: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(value)
            for value in signature.get("unique_head_dimensions", [])
            if isinstance(value, int) and not isinstance(value, bool)
        }
    )


def verified_production_dispatch_decision(
    model_signature: dict[str, Any],
    *,
    validation_dtype: str,
) -> dict[str, Any]:
    prepare_mslk_process_environment()
    from mslk.attention.fmha import triton_splitk

    profile = triton_splitk.get_production_validation_diagnostics()
    release_identity = verify_release_stack()
    required_dims = _model_head_dimensions(model_signature)
    validated_dims = sorted(int(v) for v in profile.get("validated_head_dimensions", []))
    missing_dims = sorted(set(required_dims) - set(validated_dims))
    reasons: list[str] = []
    if not release_identity.get("runtime_compatible"):
        details = "; ".join(
            f"{item.get('field')}: {item.get('detail') or item.get('actual')}"
            for item in release_identity.get("errors", [])
        )
        reasons.append(
            "The published attention stack does not satisfy the active capability contract"
            + (f": {details}" if details else ".")
        )
    if not profile.get("valid"):
        reasons.append("The packaged MSLK production validation profile does not match the active runtime.")
    if validation_dtype != "float16":
        reasons.append(
            f"The validated production profile is FP16-only; the loaded UNet dtype is {validation_dtype}."
        )
    if missing_dims:
        reasons.append(
            "The model requires head dimensions absent from the validated profile: "
            + ", ".join(str(v) for v in missing_dims)
        )
    if profile.get("provider") not in {None, _REQUIRED_PROVIDER}:
        reasons.append(
            f"Validation profile provider {profile.get('provider')!r} is not {_REQUIRED_PROVIDER!r}."
        )
    if profile.get("operator") not in {None, _REQUIRED_OPERATOR}:
        reasons.append(
            f"Validation profile operator {profile.get('operator')!r} is not {_REQUIRED_OPERATOR!r}."
        )

    return {
        "schema_version": 1,
        "verified": not reasons,
        "provider": _REQUIRED_PROVIDER,
        "operator": _REQUIRED_OPERATOR,
        "processor": "ImageGenMSLKXFormersAttnProcessor",
        "required_head_dimensions": required_dims,
        "validated_head_dimensions": validated_dims,
        "missing_head_dimensions": missing_dims,
        "validation_dtype": validation_dtype,
        "profile": profile,
        "release_identity": release_identity,
        "rejection_reasons": reasons,
    }


def capability_production_dispatch_decision(
    model_signature: dict[str, Any],
    *,
    validation_dtype: str,
) -> dict[str, Any]:
    """Admit community hardware to the real layout execution probe.

    This intentionally does not claim that the SM120 validation profile matches.
    The subsequent xFormers layout matrix executes the requested MSLK operator on
    the active GPU and is the compatibility gate for this model/runtime pair.
    """

    environment = prepare_mslk_process_environment()
    reasons: list[str] = []
    cuda_available = False
    compute_capability = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            compute_capability = list(torch.cuda.get_device_capability(torch.cuda.current_device()))
        else:
            reasons.append("CUDA is unavailable for the community MSLK capability probe.")
    except Exception as exc:
        reasons.append(f"PyTorch CUDA capability probe failed: {type(exc).__name__}: {exc}")

    package_imports: dict[str, Any] = {}
    try:
        import xformers
        from mslk.attention.fmha import triton_splitk

        package_imports = {
            "xformers": getattr(xformers, "__version__", "unknown"),
            "mslk_triton_splitk": str(getattr(triton_splitk, "__file__", "available")),
        }
    except Exception as exc:
        reasons.append(f"MSLK/xFormers import failed: {type(exc).__name__}: {exc}")

    return {
        "schema_version": 1,
        "verified": not reasons,
        "qualification": "community_unverified",
        "verification_scope": "capability_probe_pending_layout_execution",
        "provider": _REQUIRED_PROVIDER,
        "operator": _REQUIRED_OPERATOR,
        "processor": "ImageGenMSLKXFormersAttnProcessor",
        "required_head_dimensions": _model_head_dimensions(model_signature),
        "validated_head_dimensions": [],
        "missing_head_dimensions": [],
        "validation_dtype": validation_dtype,
        "cuda_available": cuda_available,
        "compute_capability": compute_capability,
        "package_imports": package_imports,
        "environment": environment,
        "rejection_reasons": reasons,
    }


def production_dispatch_decision(
    model_signature: dict[str, Any],
    *,
    validation_dtype: str,
) -> dict[str, Any]:
    if hardware_qualification_mode() == "validated":
        decision = verified_production_dispatch_decision(
            model_signature, validation_dtype=validation_dtype
        )
        decision["qualification"] = "validated"
        decision["verification_scope"] = "published_reference_profile"
        return decision
    return capability_production_dispatch_decision(
        model_signature, validation_dtype=validation_dtype
    )


def require_production_dispatch(
    model_signature: dict[str, Any],
    *,
    validation_dtype: str,
) -> dict[str, Any]:
    decision = production_dispatch_decision(
        model_signature, validation_dtype=validation_dtype
    )
    if not decision["verified"]:
        detail = "; ".join(decision.get("rejection_reasons") or []) or "unknown mismatch"
        raise RuntimeError("MSLK production dispatch is unavailable: " + detail)
    return decision


def require_verified_production_dispatch(
    model_signature: dict[str, Any],
    *,
    validation_dtype: str,
) -> dict[str, Any]:
    decision = verified_production_dispatch_decision(
        model_signature, validation_dtype=validation_dtype
    )
    if not decision["verified"]:
        detail = "; ".join(decision["rejection_reasons"]) or "unknown mismatch"
        mismatches = decision.get("profile", {}).get("mismatches") or []
        if mismatches:
            detail += "; fingerprint mismatches=" + repr(mismatches)
        raise RuntimeError("Verified MSLK production dispatch is unavailable: " + detail)
    return decision


def verified_layout_executor(query: Any, key: Any, value: Any, **kwargs: Any) -> Any:
    import xformers.ops as xops
    from mslk.attention.fmha import triton_splitk

    kwargs["op"] = (triton_splitk.FwOp, None)
    return xops.memory_efficient_attention(query, key, value, **kwargs)


def _record_execution(*, success: bool, event: dict[str, Any]) -> None:
    with _EXECUTION_LOCK:
        if success:
            _EXECUTION_STATE["executed"] = True
            _EXECUTION_STATE["successful_call_count"] += 1
            dims = set(_EXECUTION_STATE["logical_head_dimensions"])
            logical = event.get("logical_head_dimension")
            if isinstance(logical, int):
                dims.add(logical)
            _EXECUTION_STATE["logical_head_dimensions"] = sorted(dims)
            _EXECUTION_STATE["last_success"] = dict(event)
        else:
            _EXECUTION_STATE["failed_call_count"] += 1
            _EXECUTION_STATE["last_failure"] = dict(event)


def get_execution_evidence() -> dict[str, Any]:
    with _EXECUTION_LOCK:
        result = dict(_EXECUTION_STATE)
        result["logical_head_dimensions"] = list(
            _EXECUTION_STATE["logical_head_dimensions"]
        )
        if isinstance(_EXECUTION_STATE.get("last_success"), dict):
            result["last_success"] = dict(_EXECUTION_STATE["last_success"])
        if isinstance(_EXECUTION_STATE.get("last_failure"), dict):
            result["last_failure"] = dict(_EXECUTION_STATE["last_failure"])
        result["mslk_runtime_configuration"] = _mslk_runtime_configuration()
        return result


def reset_execution_evidence_for_testing() -> None:
    with _EXECUTION_LOCK:
        _EXECUTION_STATE.update(
            {
                "executed": False,
                "successful_call_count": 0,
                "failed_call_count": 0,
                "logical_head_dimensions": [],
                "last_success": None,
                "last_failure": None,
            }
        )


def build_verified_xformers_processor() -> Any:
    prepare_mslk_process_environment()
    from diffusers.models.attention_processor import XFormersAttnProcessor
    from mslk.attention.fmha import triton_splitk

    class ImageGenMSLKXFormersAttnProcessor(XFormersAttnProcessor):
        image_gen_provider = _REQUIRED_PROVIDER
        image_gen_operator = _REQUIRED_OPERATOR

        def __init__(self) -> None:
            super().__init__(attention_op=(triton_splitk.FwOp, None))

        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            encoder_hidden_states: Any = None,
            attention_mask: Any = None,
            temb: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            heads = int(getattr(attn, "heads", 0) or 0)
            out_features = int(getattr(getattr(attn, "to_q", None), "out_features", 0) or 0)
            logical_dim = out_features // heads if heads > 0 and out_features % heads == 0 else None
            tile_dim = None
            if logical_dim is not None:
                try:
                    tile_dim = triton_splitk.get_logical_tile_head_dim(logical_dim)[1]
                except Exception:
                    tile_dim = logical_dim
            event = {
                "provider": _REQUIRED_PROVIDER,
                "operator": _REQUIRED_OPERATOR,
                "processor": type(self).__name__,
                "attention_kind": "cross" if encoder_hidden_states is not None else "self",
                "logical_head_dimension": logical_dim,
                "tile_head_dimension": tile_dim,
                "heads": heads,
                "input_shape": list(getattr(hidden_states, "shape", ())),
                "dtype": str(getattr(hidden_states, "dtype", "unknown")).replace("torch.", ""),
                "device": str(getattr(hidden_states, "device", "unknown")),
            }
            try:
                output = super().__call__(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    *args,
                    **kwargs,
                )
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                _record_execution(success=False, event=event)
                raise
            event["output_shape"] = list(getattr(output, "shape", ()))
            _record_execution(success=True, event=event)
            return output

    ImageGenMSLKXFormersAttnProcessor.__name__ = "ImageGenMSLKXFormersAttnProcessor"
    ImageGenMSLKXFormersAttnProcessor.__qualname__ = "ImageGenMSLKXFormersAttnProcessor"
    ImageGenMSLKXFormersAttnProcessor.__module__ = __name__
    return ImageGenMSLKXFormersAttnProcessor()


__all__ = [
    "build_verified_xformers_processor",
    "capability_production_dispatch_decision",
    "get_execution_evidence",
    "hardware_qualification_mode",
    "prepare_mslk_process_environment",
    "production_dispatch_decision",
    "require_production_dispatch",
    "require_verified_production_dispatch",
    "reset_execution_evidence_for_testing",
    "verified_layout_executor",
    "verified_production_dispatch_decision",
]
