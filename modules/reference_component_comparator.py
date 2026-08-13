from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from modules.state_dict_converter import StateDictConverter


@dataclass(frozen=True)
class ComponentTensorComparison:
    source_key: str
    converted_key: str
    matched: bool
    reason: str
    source_shape: tuple[int, ...] | None
    reference_shape: tuple[int, ...] | None
    source_dtype: str
    reference_dtype: str
    max_abs_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "converted_key": self.converted_key,
            "matched": self.matched,
            "reason": self.reason,
            "source_shape": list(self.source_shape) if self.source_shape is not None else None,
            "reference_shape": list(self.reference_shape) if self.reference_shape is not None else None,
            "source_dtype": self.source_dtype,
            "reference_dtype": self.reference_dtype,
            "max_abs_diff": self.max_abs_diff,
        }


class ReferenceComponentComparator:
    """Compare monolithic checkpoint components against Diffusers reference weights.

    Tensor loading is lazy: only the source/reference tensor currently being
    compared is materialized. This avoids loading the full checkpoint and the
    full component reference into RAM simultaneously.
    """

    def __init__(self, converter: StateDictConverter | None = None) -> None:
        self.converter = converter or StateDictConverter()

    def _mapped_key(self, component: str, source_key: str) -> str:
        marker = object()
        if component == "unet":
            converted = self.converter.convert_unet_state_dict({source_key: marker})
            if len(converted) != 1:
                raise RuntimeError(
                    f"Expected one converted key for {component}:{source_key}, got {len(converted)}"
                )
            return next(iter(converted.keys()))
        if component == "vae":
            return self.converter.convert_vae_key(source_key)
        raise ValueError(f"Unsupported reference component: {component!r}")

    def compare(
        self,
        *,
        checkpoint_path: str | Path,
        reference_path: str | Path,
        component: str,
    ) -> dict[str, Any]:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        reference = Path(reference_path).expanduser().resolve()
        prefix = {
            "unet": "model.diffusion_model.",
            "vae": "first_stage_model.",
        }.get(component)
        if prefix is None:
            raise ValueError(f"Unsupported reference component: {component!r}")

        results: list[ComponentTensorComparison] = []
        collisions: dict[str, list[str]] = {}

        with safe_open(str(checkpoint), framework="pt", device="cpu") as source_handle, safe_open(
            str(reference), framework="pt", device="cpu"
        ) as reference_handle:
            source_full_keys = sorted(key for key in source_handle.keys() if key.startswith(prefix))
            reference_keys = set(reference_handle.keys())
            converted_to_source: dict[str, str] = {}

            for source_full_key in source_full_keys:
                source_key = source_full_key[len(prefix):]
                converted_key = self._mapped_key(component, source_key)
                prior = converted_to_source.get(converted_key)
                if prior is not None:
                    collisions.setdefault(converted_key, [prior]).append(source_full_key)
                else:
                    converted_to_source[converted_key] = source_full_key

                if converted_key not in reference_keys:
                    source_slice = source_handle.get_slice(source_full_key)
                    results.append(
                        ComponentTensorComparison(
                            source_key=source_full_key,
                            converted_key=converted_key,
                            matched=False,
                            reason="missing_in_reference",
                            source_shape=tuple(int(v) for v in source_slice.get_shape()),
                            reference_shape=None,
                            source_dtype=str(source_slice.get_dtype()),
                            reference_dtype="",
                        )
                    )
                    continue

                source_slice = source_handle.get_slice(source_full_key)
                reference_slice = reference_handle.get_slice(converted_key)
                source_dtype = str(source_slice.get_dtype())
                reference_dtype = str(reference_slice.get_dtype())
                source_tensor = source_handle.get_tensor(source_full_key)
                if component == "vae":
                    source_tensor = self.converter.convert_vae_tensor(converted_key, source_tensor)
                source_shape = tuple(int(v) for v in source_tensor.shape)
                reference_shape = tuple(int(v) for v in reference_slice.get_shape())

                if source_shape != reference_shape:
                    results.append(
                        ComponentTensorComparison(
                            source_key=source_full_key,
                            converted_key=converted_key,
                            matched=False,
                            reason="shape_mismatch",
                            source_shape=source_shape,
                            reference_shape=reference_shape,
                            source_dtype=source_dtype,
                            reference_dtype=reference_dtype,
                        )
                    )
                    continue

                reference_tensor = reference_handle.get_tensor(converted_key)
                same_dtype = source_tensor.dtype == reference_tensor.dtype
                same_values = bool(torch.equal(source_tensor, reference_tensor))
                if same_dtype and same_values:
                    results.append(
                        ComponentTensorComparison(
                            source_key=source_full_key,
                            converted_key=converted_key,
                            matched=True,
                            reason="",
                            source_shape=source_shape,
                            reference_shape=reference_shape,
                            source_dtype=source_dtype,
                            reference_dtype=reference_dtype,
                            max_abs_diff=0.0,
                        )
                    )
                    continue

                max_abs_diff: float | None = None
                if source_tensor.is_floating_point() and reference_tensor.is_floating_point():
                    diff = torch.max(
                        torch.abs(
                            source_tensor.to(dtype=torch.float32)
                            - reference_tensor.to(dtype=torch.float32)
                        )
                    )
                    max_abs_diff = float(diff.item()) if diff.numel() else 0.0
                reason = "dtype_mismatch" if not same_dtype and same_values else "value_mismatch"
                results.append(
                    ComponentTensorComparison(
                        source_key=source_full_key,
                        converted_key=converted_key,
                        matched=False,
                        reason=reason,
                        source_shape=source_shape,
                        reference_shape=reference_shape,
                        source_dtype=source_dtype,
                        reference_dtype=reference_dtype,
                        max_abs_diff=max_abs_diff,
                    )
                )

            converted_keys = set(converted_to_source.keys())
            missing_in_converted = sorted(reference_keys - converted_keys)

        failures = [item for item in results if not item.matched]
        return {
            "component": component,
            "checkpoint_path": str(checkpoint),
            "reference_path": str(reference),
            "source_tensor_count": len(results),
            "reference_tensor_count": len(reference_keys),
            "matched_tensor_count": len(results) - len(failures),
            "failed_tensor_count": len(failures),
            "missing_in_converted": missing_in_converted,
            "collision_count": len(collisions),
            "collisions": collisions,
            "matched": not failures and not missing_in_converted and not collisions,
            "comparisons": [item.to_dict() for item in results],
        }
