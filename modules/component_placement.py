from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class ComponentPlacementReport:
    component: str
    owner: str
    requested_device: str
    requested_dtype: str | None
    before_device: str | None
    before_dtypes: list[str]
    after_device: str | None
    after_dtypes: list[str]
    keep_in_fp32_patterns: list[str]
    kept_in_fp32_names: list[str]
    device_move_applied: bool
    dtype_cast_applied: bool
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _named_floating_tensors(module: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, parameter in module.named_parameters():
        if parameter.is_floating_point():
            yield name, parameter
    for name, buffer in module.named_buffers():
        if buffer.is_floating_point():
            yield name, buffer


def _first_device(module: torch.nn.Module) -> torch.device | None:
    for _name, tensor in _named_floating_tensors(module):
        return tensor.device
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _floating_dtypes(module: torch.nn.Module) -> list[torch.dtype]:
    return sorted(
        {tensor.dtype for _name, tensor in _named_floating_tensors(module)},
        key=str,
    )


def _keep_patterns(module: torch.nn.Module) -> list[str]:
    raw = getattr(module, "_keep_in_fp32_modules", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw if str(value)]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(pattern in name for pattern in patterns)


def _resolved_device_index(device: torch.device) -> int | None:
    """Resolve an implicit accelerator index when the runtime can provide it.

    PyTorch accepts ``torch.device("cuda")`` as the current CUDA device, while
    tensors report their concrete placement as ``cuda:0``, ``cuda:1``, and so
    on. Direct device-object equality therefore rejects a correct placement.
    """

    if device.index is not None:
        return int(device.index)
    if device.type == "cuda":
        try:
            return int(torch.cuda.current_device())
        except Exception:
            # During CPU-only validation there is no CUDA runtime from which to
            # resolve the implicit index. The caller can still compare by type.
            return None
    return None


def devices_equivalent(
    actual: str | torch.device,
    expected: str | torch.device,
) -> bool:
    """Return whether two device specifications identify the same placement.

    In particular, ``cuda`` is equivalent to the current concrete CUDA device,
    which is normally reported by tensors as ``cuda:0``.
    """

    actual_device = torch.device(actual)
    expected_device = torch.device(expected)
    if actual_device.type != expected_device.type:
        return False
    if actual_device == expected_device:
        return True

    actual_index = _resolved_device_index(actual_device)
    expected_index = _resolved_device_index(expected_device)
    if actual_index is not None and expected_index is not None:
        return actual_index == expected_index

    # An unresolved unindexed accelerator means "the current device". This
    # fallback is needed by CPU-only tests and remains stricter than accepting a
    # different device type.
    return actual_device.index is None or expected_device.index is None


def _restore_required_fp32_tensors(
    module: torch.nn.Module,
    patterns: list[str],
    *,
    device: torch.device,
) -> list[str]:
    if not patterns:
        return []

    restored: list[str] = []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.is_floating_point() and _matches_any(name, patterns):
                parameter.data = parameter.data.to(device=device, dtype=torch.float32)
                restored.append(name)
        for name, buffer in module.named_buffers():
            if buffer.is_floating_point() and _matches_any(name, patterns):
                buffer.data = buffer.data.to(device=device, dtype=torch.float32)
                restored.append(name)
    return sorted(set(restored))


def _verify_placement(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    keep_patterns: list[str],
) -> None:
    errors: list[str] = []
    for name, tensor in _named_floating_tensors(module):
        if not devices_equivalent(tensor.device, device):
            errors.append(f"{name}: device={tensor.device}, expected={device}")
        if dtype is None:
            continue
        expected_dtype = torch.float32 if _matches_any(name, keep_patterns) else dtype
        if tensor.dtype != expected_dtype:
            errors.append(
                f"{name}: dtype={tensor.dtype}, expected={expected_dtype}"
            )
        if len(errors) >= 12:
            break
    if errors:
        raise RuntimeError(
            f"Component placement verification failed for {type(module).__name__}: "
            + "; ".join(errors)
        )


def component_matches_placement(
    module: torch.nn.Module,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> bool:
    target_device = torch.device(device)
    try:
        _verify_placement(
            module,
            device=target_device,
            dtype=dtype,
            keep_patterns=_keep_patterns(module),
        )
    except RuntimeError:
        return False
    return True


def place_component(
    module: torch.nn.Module,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    owner: str,
    component_name: str | None = None,
) -> ComponentPlacementReport:
    """Move and cast one model without using Diffusers' warning-producing dtype path.

    Diffusers overrides ``ModelMixin.to`` to warn whenever a dtype is supplied,
    even when ``_keep_in_fp32_modules`` is empty. IMAGE_GEN manually constructs
    models from config/state dictionaries, so it cannot use ``from_pretrained``'s
    ``torch_dtype`` loading path. Calling the base ``torch.nn.Module.to`` method
    performs the equivalent tensor move while this function explicitly restores
    any named precision-sensitive tensors to float32.
    """

    target_device = torch.device(device)
    target_dtype = dtype
    before_device = _first_device(module)
    before_dtypes = _floating_dtypes(module)
    keep_patterns = _keep_patterns(module)

    device_move_applied = before_device is not None and before_device != target_device
    dtype_cast_applied = target_dtype is not None and any(
        tensor_dtype != target_dtype for tensor_dtype in before_dtypes
    )

    if target_dtype is None:
        torch.nn.Module.to(module, device=target_device)
        kept_names: list[str] = []
    else:
        # Intentionally call the PyTorch base implementation. This avoids the
        # Diffusers ModelMixin.to warning while retaining normal Module.to rules.
        torch.nn.Module.to(module, device=target_device, dtype=target_dtype)
        kept_names = _restore_required_fp32_tensors(
            module,
            keep_patterns,
            device=target_device,
        )

    _verify_placement(
        module,
        device=target_device,
        dtype=target_dtype,
        keep_patterns=keep_patterns,
    )

    after_device = _first_device(module)
    after_dtypes = _floating_dtypes(module)
    report = ComponentPlacementReport(
        component=component_name or type(module).__name__,
        owner=str(owner),
        requested_device=str(target_device),
        requested_dtype=str(target_dtype) if target_dtype is not None else None,
        before_device=str(before_device) if before_device is not None else None,
        before_dtypes=[str(value) for value in before_dtypes],
        after_device=str(after_device) if after_device is not None else None,
        after_dtypes=[str(value) for value in after_dtypes],
        keep_in_fp32_patterns=keep_patterns,
        kept_in_fp32_names=kept_names,
        device_move_applied=bool(device_move_applied),
        dtype_cast_applied=bool(dtype_cast_applied),
        verified=True,
    )
    setattr(module, "_image_gen_placement_report", report.to_dict())
    return report


def component_placement_report(module: torch.nn.Module) -> dict[str, Any]:
    existing = getattr(module, "_image_gen_placement_report", None)
    if isinstance(existing, dict):
        return dict(existing)
    device = _first_device(module)
    return {
        "component": type(module).__name__,
        "owner": "external_or_unreported",
        "requested_device": None,
        "requested_dtype": None,
        "before_device": None,
        "before_dtypes": [],
        "after_device": str(device) if device is not None else None,
        "after_dtypes": [str(value) for value in _floating_dtypes(module)],
        "keep_in_fp32_patterns": _keep_patterns(module),
        "kept_in_fp32_names": [],
        "device_move_applied": False,
        "dtype_cast_applied": False,
        "verified": False,
    }
