from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .model_signature import build_model_attention_signature, iter_attention_modules
from .reports import shape_list, stable_hash, tensor_metadata


_CAPTURE_ENV = "IMAGE_GEN_ATTENTION_CAPTURE"
_CAPTURE_DIR_ENV = "IMAGE_GEN_ATTENTION_CAPTURE_DIR"
_ALLOWED_MODES = {"off", "metadata", "full"}


def resolve_capture_mode(value: str | None = None) -> str:
    selected = str(value if value is not None else os.environ.get(_CAPTURE_ENV, "metadata")).strip().lower()
    aliases = {"0": "off", "false": "off", "none": "off", "1": "metadata", "true": "metadata"}
    selected = aliases.get(selected, selected)
    if selected not in _ALLOWED_MODES:
        raise ValueError(f"{_CAPTURE_ENV} must be one of: off, metadata, full.")
    return selected


def _sequence_metadata(value: Any) -> dict[str, int | None]:
    shape = shape_list(value)
    if not shape:
        return {"batch": None, "sequence_length": None, "feature_dim": None}
    if len(shape) == 3:
        return {"batch": shape[0], "sequence_length": shape[1], "feature_dim": shape[2]}
    if len(shape) == 4:
        return {"batch": shape[0], "sequence_length": shape[2] * shape[3], "feature_dim": shape[1]}
    return {
        "batch": shape[0] if shape else None,
        "sequence_length": None,
        "feature_dim": shape[-1] if shape else None,
    }


def _projection_out_features(module: Any, name: str) -> int | None:
    projection = getattr(module, name, None)
    value = getattr(projection, "out_features", None)
    return int(value) if isinstance(value, int) else None


def _head_dim(out_features: int | None, heads: int | None) -> int | None:
    if out_features is None or heads is None or heads <= 0:
        return None
    quotient, remainder = divmod(out_features, heads)
    return quotient if remainder == 0 else None


def _projected_shape(batch: int | None, sequence: int | None, out_features: int | None) -> list[int] | None:
    if None in {batch, sequence, out_features}:
        return None
    return [int(batch), int(sequence), int(out_features)]


def _operator_shape(batch: int | None, sequence: int | None, heads: int | None, head_dim: int | None) -> list[int] | None:
    if None in {batch, sequence, heads, head_dim}:
        return None
    return [int(batch), int(sequence), int(heads), int(head_dim)]


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


class AttentionLayoutCapture:
    def __init__(
        self,
        unet: Any,
        *,
        mode: str,
        output_dir: str | Path | None,
        model_signature: dict[str, Any],
    ) -> None:
        self.unet = unet
        self.mode = mode
        self.output_dir = Path(output_dir) if output_dir else None
        self.model_signature = dict(model_signature)
        self.records: list[dict[str, Any]] = []
        self._seen_modules: set[str] = set()
        self._handles: dict[str, Any] = {}
        self._module_count = int(model_signature.get("attention_module_count") or 0)

    def install(self) -> None:
        if self.mode == "off":
            return
        for path, module in iter_attention_modules(self.unet):
            self._install_hook(path, module)
        self._publish()

    def _install_hook(self, path: str, module: Any) -> None:
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            return

        def hook_with_kwargs(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            self._capture(path, _module, args, kwargs, output)

        try:
            handle = register(hook_with_kwargs, with_kwargs=True)
        except TypeError:
            def hook_legacy(_module: Any, args: tuple[Any, ...], output: Any) -> None:
                self._capture(path, _module, args, {}, output)
            handle = register(hook_legacy)
        self._handles[path] = handle

    def _capture(
        self,
        path: str,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if path in self._seen_modules:
            return
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        encoder_hidden_states = kwargs.get(
            "encoder_hidden_states",
            args[1] if len(args) > 1 else None,
        )
        attention_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        hidden = _sequence_metadata(hidden_states)
        source = _sequence_metadata(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        heads_value = getattr(module, "heads", None)
        heads = int(heads_value) if isinstance(heads_value, int) else None
        q_out = _projection_out_features(module, "to_q")
        k_out = _projection_out_features(module, "to_k")
        v_out = _projection_out_features(module, "to_v")
        q_dim = _head_dim(q_out, heads)
        k_dim = _head_dim(k_out, heads)
        v_dim = _head_dim(v_out, heads)
        record: dict[str, Any] = {
            "module_path": path,
            "attention_kind": "cross" if encoder_hidden_states is not None else "self",
            "hidden_states": tensor_metadata(hidden_states),
            "encoder_hidden_states": tensor_metadata(encoder_hidden_states),
            "query_projection_shape": _projected_shape(hidden["batch"], hidden["sequence_length"], q_out),
            "key_projection_shape": _projected_shape(source["batch"], source["sequence_length"], k_out),
            "value_projection_shape": _projected_shape(source["batch"], source["sequence_length"], v_out),
            "query_operator_shape": _operator_shape(hidden["batch"], hidden["sequence_length"], heads, q_dim),
            "key_operator_shape": _operator_shape(source["batch"], source["sequence_length"], heads, k_dim),
            "value_operator_shape": _operator_shape(source["batch"], source["sequence_length"], heads, v_dim),
            "batch": hidden["batch"],
            "query_sequence_length": hidden["sequence_length"],
            "key_value_sequence_length": source["sequence_length"],
            "heads": heads,
            "q_head_dim": q_dim,
            "k_head_dim": k_dim,
            "v_head_dim": v_dim,
            "dtype": str(getattr(hidden_states, "dtype", "unknown")).replace("torch.", ""),
            "device": str(getattr(hidden_states, "device", "unknown")),
            "attention_mask": tensor_metadata(attention_mask),
            "output_shape": shape_list(_first_tensor(output)),
            "full_tensor_capture": None,
        }
        if self.mode == "full":
            record["full_tensor_capture"] = self._save_full_tensors(
                path,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
            )
        self.records.append(record)
        self._seen_modules.add(path)
        handle = self._handles.pop(path, None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
        self._publish()

    def _save_full_tensors(
        self,
        path: str,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any,
    ) -> str:
        import torch

        root = self.output_dir or Path("artifacts") / "attention_validation" / "runtime"
        full_dir = root / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path).strip("_") or "attention"
        target = full_dir / f"{safe_name}.pt"
        payload = {
            "hidden_states": hidden_states.detach().cpu() if hasattr(hidden_states, "detach") else hidden_states,
            "encoder_hidden_states": (
                encoder_hidden_states.detach().cpu()
                if hasattr(encoder_hidden_states, "detach")
                else encoder_hidden_states
            ),
            "attention_mask": attention_mask.detach().cpu() if hasattr(attention_mask, "detach") else attention_mask,
        }
        torch.save(payload, target)
        return str(target)

    def report(self) -> dict[str, Any]:
        unique_layouts: dict[tuple[Any, ...], dict[str, Any]] = {}
        head_dims: set[int] = set()
        dtypes: set[str] = set()
        mask_requirements: set[str] = set()
        for record in self.records:
            for name in ("q_head_dim", "k_head_dim", "v_head_dim"):
                value = record.get(name)
                if isinstance(value, int):
                    head_dims.add(value)
            dtype = record.get("dtype")
            if dtype:
                dtypes.add(str(dtype))
            mask = record.get("attention_mask")
            mask_requirements.add("none" if mask is None else str(mask.get("python_type") or "tensor"))
            key = (
                record.get("attention_kind"),
                record.get("query_sequence_length"),
                record.get("key_value_sequence_length"),
                record.get("heads"),
                record.get("q_head_dim"),
                record.get("k_head_dim"),
                record.get("v_head_dim"),
                record.get("dtype"),
                record.get("device"),
                "none" if mask is None else str(mask.get("python_type") or "tensor"),
            )
            unique_layouts.setdefault(
                key,
                {
                    "attention_kind": record.get("attention_kind"),
                    "query_sequence_length": record.get("query_sequence_length"),
                    "key_value_sequence_length": record.get("key_value_sequence_length"),
                    "heads": record.get("heads"),
                    "q_head_dim": record.get("q_head_dim"),
                    "k_head_dim": record.get("k_head_dim"),
                    "v_head_dim": record.get("v_head_dim"),
                    "dtype": record.get("dtype"),
                    "device": record.get("device"),
                    "mask_requirement": "none" if mask is None else str(mask.get("python_type") or "tensor"),
                    "module_paths": [],
                },
            )["module_paths"].append(record["module_path"])
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": self.mode,
            "enabled": self.mode != "off",
            "metadata_only": self.mode == "metadata",
            "captured_module_count": len(self._seen_modules),
            "expected_module_count": self._module_count,
            "complete": self._module_count > 0 and len(self._seen_modules) >= self._module_count,
            "records": list(self.records),
            "unique_layouts": list(unique_layouts.values()),
            "unique_head_dimensions": sorted(head_dims),
            "dtypes": sorted(dtypes),
            "mask_requirements": sorted(mask_requirements),
            "static_signature_hash": self.model_signature.get("signature_hash"),
        }
        result["runtime_signature_hash"] = stable_hash(
            {key: value for key, value in result.items() if key != "runtime_signature_hash"}
        )
        return result

    def _publish(self) -> None:
        report = self.report()
        setattr(self.unet, "_image_gen_attention_runtime_capture_report", report)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            target = self.output_dir / "model_attention_runtime_signature.json"
            target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def close(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()


def install_attention_layout_capture(
    unet: Any,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
    model_signature: dict[str, Any] | None = None,
) -> AttentionLayoutCapture:
    selected_mode = resolve_capture_mode(mode)
    selected_output = output_dir or os.environ.get(_CAPTURE_DIR_ENV)
    signature = model_signature or build_model_attention_signature(unet)
    capture = AttentionLayoutCapture(
        unet,
        mode=selected_mode,
        output_dir=selected_output,
        model_signature=signature,
    )
    setattr(unet, "_image_gen_attention_layout_capture", capture)
    capture.install()
    return capture


def attention_layout_capture_report(unet: Any) -> dict[str, Any] | None:
    capture = getattr(unet, "_image_gen_attention_layout_capture", None)
    if isinstance(capture, AttentionLayoutCapture):
        return capture.report()
    report = getattr(unet, "_image_gen_attention_runtime_capture_report", None)
    return dict(report) if isinstance(report, dict) else None
