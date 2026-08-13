from __future__ import annotations

import inspect
from typing import Any, Iterable, Mapping

from image_gen.runtime.adapters.registry import STANDARD_DIFFUSERS_LOADER_ID
from image_gen.runtime.adapters.standard_mapping import normalize_standard_state_dict

_COMPONENTS = ("unet", "text_encoder", "text_encoder_2")


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class StandardDiffusersAdapterLoader:
    """IMAGE_GEN standard LoRA loader backed by the pinned Diffusers/PEFT stack.

    The orchestration layer selects this implementation by stable loader ID. All
    standard-format conversion, component mapping, application, verification,
    activation, and deactivation behavior is owned here rather than in
    ``LoRARuntimeManager``.
    """

    loader_id = STANDARD_DIFFUSERS_LOADER_ID

    @staticmethod
    def _call_with_supported_kwargs(target: Any, kwargs: dict[str, Any]) -> Any:
        filtered = {key: value for key, value in kwargs.items() if value is not None}
        try:
            signature = inspect.signature(target)
            accepts_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_var_kwargs:
                filtered = {key: value for key, value in filtered.items() if key in signature.parameters}
        except (TypeError, ValueError):
            pass
        return target(**filtered)

    @staticmethod
    def _stable_diffusion_lora_loader_mixin() -> Any:
        try:
            from diffusers.loaders import StableDiffusionLoraLoaderMixin

            return StableDiffusionLoraLoaderMixin
        except Exception:
            try:
                from diffusers.loaders import LoraLoaderMixin

                return LoraLoaderMixin
            except Exception as exc:
                raise ValueError(
                    "Unable to import diffusers LoRA loader support. Expected the pinned diffusers build "
                    "to expose StableDiffusionLoraLoaderMixin or LoraLoaderMixin."
                ) from exc

    @staticmethod
    def _count_prefixed_keys(state_dict: Mapping[str, Any], prefix: str) -> int:
        return sum(1 for key in state_dict if str(key or "").startswith(prefix))

    @staticmethod
    def _verify_adapter_presence(module: Any, adapter_name: str) -> bool:
        if module is None or not adapter_name:
            return False
        peft_config = getattr(module, "peft_config", None)
        if isinstance(peft_config, Mapping) and adapter_name in peft_config:
            return True
        active = getattr(module, "active_adapters", None)
        if callable(active):
            try:
                current = active()
            except Exception:
                current = []
        else:
            current = active
        if isinstance(current, str):
            return current == adapter_name
        if isinstance(current, (list, tuple, set)):
            return adapter_name in current
        return False

    @staticmethod
    def _entry_context(entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = dict(entry.get("metadata") or {})
        inspection_record = dict(entry.get("adapter_inspection") or metadata.get("_adapter_inspection") or {})
        runtime_plan = dict(entry.get("adapter_runtime_plan") or metadata.get("_adapter_runtime_plan") or {})
        return inspection_record, runtime_plan

    def _error(
        self,
        *,
        entry: Mapping[str, Any],
        failed_target: str,
        detail: str,
    ) -> ValueError:
        inspection_record, runtime_plan = self._entry_context(entry)
        label = str(entry.get("requested_name") or entry.get("name") or entry.get("adapter_name") or "adapter")
        adapter_format = str(inspection_record.get("adapter_format") or entry.get("adapter_format") or "unknown_adapter")
        family = str(
            runtime_plan.get("active_checkpoint_family")
            or inspection_record.get("model_family")
            or entry.get("model_family")
            or "unknown"
        )
        targets = list(runtime_plan.get("expected_component_targets") or inspection_record.get("target_scopes") or [])
        target_text = ",".join(str(item) for item in targets) or "unknown"
        return ValueError(
            f"LoRA '{label}' failed standard loader execution: loader={self.loader_id}; "
            f"format={adapter_format}; family={family}; targets={target_text}; "
            f"failed_target={failed_target}; {detail}"
        )

    def _load_lora_state(
        self,
        *,
        path: str,
        components: Any,
        entry: Mapping[str, Any],
        mixin: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        inspection_record, runtime_plan = self._entry_context(entry)
        adapter_format = str(inspection_record.get("adapter_format") or entry.get("adapter_format") or "")
        model_family = str(runtime_plan.get("active_checkpoint_family") or inspection_record.get("model_family") or "")
        expected_targets = list(runtime_plan.get("expected_component_targets") or inspection_record.get("target_scopes") or [])
        kwargs = {
            "pretrained_model_name_or_path_or_dict": path,
            "unet_config": getattr(getattr(components, "unet", None), "config", None),
            "return_lora_metadata": True,
        }
        try:
            loaded = self._call_with_supported_kwargs(mixin.lora_state_dict, kwargs)
        except TypeError:
            kwargs.pop("return_lora_metadata", None)
            loaded = self._call_with_supported_kwargs(mixin.lora_state_dict, kwargs)
        except Exception as exc:
            raise self._error(entry=entry, failed_target="conversion", detail=str(exc)) from exc
        if not isinstance(loaded, tuple):
            raise self._error(entry=entry, failed_target="conversion", detail="diffusers returned a non-tuple LoRA state payload.")
        if len(loaded) == 3:
            state_dict, network_alphas, metadata = loaded
        elif len(loaded) == 2:
            state_dict, network_alphas = loaded
            metadata = {}
        else:
            raise self._error(entry=entry, failed_target="conversion", detail="diffusers returned an unexpected LoRA tuple shape.")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise self._error(entry=entry, failed_target="conversion", detail="diffusers conversion did not produce a usable state dict.")
        network_alphas = dict(network_alphas or {}) if isinstance(network_alphas, Mapping) else {}
        metadata = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
        normalized_state, normalized_alphas, mapping_report = normalize_standard_state_dict(
            dict(state_dict),
            network_alphas,
            adapter_format=adapter_format,
            model_family=model_family,
            expected_targets=expected_targets,
        )
        if int(mapping_report.get("unmapped_parameter_count") or 0) > 0:
            examples = ", ".join(mapping_report.get("unmapped_parameter_examples") or []) or "none"
            raise self._error(
                entry=entry,
                failed_target="mapping",
                detail=f"standard LoRA conversion left unmapped adapter parameter keys: {examples}.",
            )
        return normalized_state, normalized_alphas, metadata, mapping_report

    def _build_runtime_load_report(
        self,
        *,
        entry: Mapping[str, Any],
        state_dict: Mapping[str, Any],
        network_alphas: Mapping[str, Any],
        metadata: Mapping[str, Any],
        mapping_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        entry_metadata = dict(entry.get("metadata") or {})
        scan_cache = dict(entry_metadata.get("_lora_scan_cache") or {})
        inspection_record, runtime_plan = self._entry_context(entry)
        expected_targets = list(runtime_plan.get("expected_component_targets") or inspection_record.get("target_scopes") or [])
        expected_components = [target for target in _COMPONENTS if target in expected_targets]
        target_counts = dict(inspection_record.get("target_counts") or {})
        requested_weight = _coerce_float(entry.get("weight"), 1.0)
        source_tensor_format = str(scan_cache.get("tensor_key_format") or entry_metadata.get("tensor_key_format") or "")
        source_network_type = str(
            inspection_record.get("network_type")
            or scan_cache.get("network_type")
            or entry_metadata.get("network_type")
            or ""
        )
        candidate_counts = {
            "unet": self._count_prefixed_keys(state_dict, "unet."),
            "text_encoder": self._count_prefixed_keys(state_dict, "text_encoder."),
            "text_encoder_2": self._count_prefixed_keys(state_dict, "text_encoder_2."),
        }
        component_target_count = sum(
            int(target_counts.get(key) or 0)
            for key in (
                "unet_target_groups",
                "text_encoder_target_groups",
                "text_encoder_2_target_groups",
                "other_target_groups",
            )
        )
        source_rank = inspection_record.get("source_rank")
        source_alpha = inspection_record.get("source_alpha")
        return {
            "adapter_name": str(entry.get("adapter_name") or ""),
            "resolved_path": str(entry.get("resolved_path") or entry.get("path") or ""),
            "loader_id": self.loader_id,
            "loader_path": self.loader_id,
            "inspection_contract_version": str(inspection_record.get("contract_version") or ""),
            "runtime_plan_contract_version": str(runtime_plan.get("contract_version") or ""),
            "adapter_format": str(inspection_record.get("adapter_format") or entry.get("adapter_format") or ""),
            "model_family": str(inspection_record.get("model_family") or entry.get("detected_model_family") or entry.get("model_family") or ""),
            "active_checkpoint_family": str(runtime_plan.get("active_checkpoint_family") or ""),
            "expected_component_targets": expected_targets,
            "expected_components": expected_components,
            "detected_target_group_counts": target_counts,
            "module_target_count": component_target_count,
            "unsupported_target_count": 0,
            "unsupported_targets": [],
            "source_rank": source_rank,
            "source_alpha": source_alpha,
            "algorithm_native_scale": None,
            "algorithm_native_scale_source": "diffusers_loader_internal_rank_alpha_normalization",
            "requested_weight": requested_weight,
            "effective_user_multiplier": requested_weight,
            "final_effective_scale": requested_weight,
            "final_effective_scale_semantics": "user multiplier applied after loader-native rank/alpha normalization",
            "key_prefix_mode": str(mapping_report.get("prefix_mode") or ""),
            "mapping_contract_version": str(mapping_report.get("contract_version") or ""),
            "mapping_count": int(mapping_report.get("mapping_count") or 0),
            "recognized_mapping_count": int(mapping_report.get("recognized_mapping_count") or 0),
            "unmapped_parameter_count": int(mapping_report.get("unmapped_parameter_count") or 0),
            "unmapped_parameter_examples": list(mapping_report.get("unmapped_parameter_examples") or [])[:8],
            "mapping_examples": list(mapping_report.get("mapping_examples") or [])[:8],
            "source_tensor_format": source_tensor_format,
            "source_network_type": source_network_type,
            "converted_key_count": len(state_dict),
            "network_alpha_count": len(network_alphas),
            "metadata_key_count": len(metadata),
            "unet_candidate_keys": candidate_counts["unet"],
            "text_encoder_candidate_keys": candidate_counts["text_encoder"],
            "text_encoder_2_candidate_keys": candidate_counts["text_encoder_2"],
            "converted_key_examples": [str(key) for key in list(state_dict.keys())[:8]],
            "unet_expected": "unet" in expected_components,
            "text_encoder_expected": "text_encoder" in expected_components,
            "text_encoder_2_expected": "text_encoder_2" in expected_components,
            "unet_loaded": "unet" not in expected_components,
            "text_encoder_loaded": "text_encoder" not in expected_components,
            "text_encoder_2_loaded": "text_encoder_2" not in expected_components,
            "verification_failures": [],
            "activation_state": "prepared",
            "verified": False,
        }

    def _load_component(
        self,
        *,
        mixin: Any,
        component_name: str,
        components: Any,
        state_dict: Mapping[str, Any],
        network_alphas: Mapping[str, Any],
        metadata_for_load: Mapping[str, Any] | None,
        adapter_name: str,
        entry: Mapping[str, Any],
    ) -> bool:
        module = getattr(components, component_name, None)
        if module is None:
            raise self._error(
                entry=entry,
                failed_target=component_name,
                detail=f"active runtime components do not expose {component_name}.",
            )
        try:
            if component_name == "unet":
                self._call_with_supported_kwargs(
                    mixin.load_lora_into_unet,
                    {
                        "state_dict": state_dict,
                        "network_alphas": network_alphas,
                        "unet": module,
                        "adapter_name": adapter_name,
                        "metadata": metadata_for_load,
                    },
                )
            else:
                self._call_with_supported_kwargs(
                    mixin.load_lora_into_text_encoder,
                    {
                        "state_dict": state_dict,
                        "network_alphas": network_alphas,
                        "text_encoder": module,
                        "prefix": component_name,
                        "adapter_name": adapter_name,
                        "metadata": metadata_for_load,
                    },
                )
        except Exception as exc:
            raise self._error(entry=entry, failed_target=component_name, detail=str(exc)) from exc
        return self._verify_adapter_presence(module, adapter_name)

    def load(self, *, components: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
        path = str(entry.get("resolved_path") or entry.get("path") or "").strip()
        adapter_name = str(entry.get("adapter_name") or "").strip()
        if not path:
            raise self._error(entry=entry, failed_target="resolution", detail="runtime entry is missing a resolved adapter path.")
        if not adapter_name:
            raise self._error(entry=entry, failed_target="identity", detail="runtime entry is missing an adapter name.")

        mixin = self._stable_diffusion_lora_loader_mixin()
        state_dict, network_alphas, metadata, mapping_report = self._load_lora_state(
            path=path,
            components=components,
            entry=entry,
            mixin=mixin,
        )
        report = self._build_runtime_load_report(
            entry=entry,
            state_dict=state_dict,
            network_alphas=network_alphas,
            metadata=metadata,
            mapping_report=mapping_report,
        )
        candidate_counts = {
            "unet": int(report["unet_candidate_keys"]),
            "text_encoder": int(report["text_encoder_candidate_keys"]),
            "text_encoder_2": int(report["text_encoder_2_candidate_keys"]),
        }
        if not any(candidate_counts.values()):
            examples = ", ".join(report.get("converted_key_examples") or []) or "none"
            raise self._error(
                entry=entry,
                failed_target="mapping",
                detail=f"conversion produced no supported component weights; converted examples={examples}.",
            )

        metadata_for_load = metadata if metadata and not network_alphas else None
        if report.get("key_prefix_mode") == "deterministic_component_normalization":
            metadata_for_load = None

        expected_components = set(report.get("expected_components") or [])
        verification_failures: list[str] = []
        for component_name in _COMPONENTS:
            candidate_count = candidate_counts[component_name]
            expected = component_name in expected_components
            if expected and candidate_count <= 0:
                verification_failures.append(f"{component_name}:expected_target_missing_after_conversion")
                report[f"{component_name}_loaded"] = False
                continue
            if candidate_count <= 0:
                report[f"{component_name}_loaded"] = not expected
                continue
            loaded = self._load_component(
                mixin=mixin,
                component_name=component_name,
                components=components,
                state_dict=state_dict,
                network_alphas=network_alphas,
                metadata_for_load=metadata_for_load,
                adapter_name=adapter_name,
                entry=entry,
            )
            report[f"{component_name}_loaded"] = bool(loaded)
            if not loaded:
                verification_failures.append(f"{component_name}:adapter_not_registered")

        report["verification_failures"] = verification_failures
        report["verified"] = not verification_failures and all(
            bool(report.get(f"{component_name}_loaded"))
            for component_name in expected_components
        )
        report["activation_state"] = "loaded_verified" if report["verified"] else "verification_failed"
        if not report["verified"]:
            detail = ", ".join(verification_failures) or "component verification failed"
            raise self._error(entry=entry, failed_target="verification", detail=detail)
        return {
            **dict(entry),
            "runtime_applied": True,
            "runtime_load_report": report,
        }

    @staticmethod
    def _deactivate_module(module: Any) -> bool:
        if module is None:
            return False
        setter = getattr(module, "set_adapters", None)
        if callable(setter):
            try:
                setter([], adapter_weights=[])
                return True
            except TypeError:
                try:
                    setter([])
                    return True
                except Exception:
                    pass
            except Exception:
                pass
        disable = getattr(module, "disable_adapters", None)
        if callable(disable):
            try:
                disable()
                return True
            except Exception:
                pass
        return False

    def deactivate(self, *, components: Any) -> None:
        for component_name in _COMPONENTS:
            self._deactivate_module(getattr(components, component_name, None))

    @staticmethod
    def _component_stack(stack: Iterable[Mapping[str, Any]], component_name: str) -> tuple[list[str], list[float]]:
        names: list[str] = []
        weights: list[float] = []
        candidate_field = f"{component_name}_candidate_keys"
        loaded_field = f"{component_name}_loaded"
        for item in stack:
            report = dict(item.get("runtime_load_report") or {})
            name = str(item.get("adapter_name") or "").strip()
            if not name or not bool(report.get(loaded_field)):
                continue
            if int(report.get(candidate_field) or 0) <= 0:
                continue
            names.append(name)
            weights.append(_coerce_float(item.get("weight"), 1.0))
        return names, weights

    def activate(self, *, components: Any, stack: Iterable[Mapping[str, Any]]) -> None:
        activated_any = False
        for component_name in _COMPONENTS:
            module = getattr(components, component_name, None)
            if module is None:
                continue
            names, weights = self._component_stack(stack, component_name)
            if not names:
                self._deactivate_module(module)
                continue
            setter = getattr(module, "set_adapters", None)
            if callable(setter):
                try:
                    setter(names, adapter_weights=weights)
                    activated_any = True
                    continue
                except TypeError:
                    try:
                        setter(names, weights)
                        activated_any = True
                        continue
                    except Exception as exc:
                        raise ValueError(
                            f"LoRA stack activation failed: loader={self.loader_id}; component={component_name}; {exc}"
                        ) from exc
                except Exception as exc:
                    raise ValueError(
                        f"LoRA stack activation failed: loader={self.loader_id}; component={component_name}; {exc}"
                    ) from exc
            enable = getattr(module, "enable_adapters", None)
            if callable(enable) and len(names) == 1 and abs(weights[0] - 1.0) < 1e-9:
                try:
                    enable()
                    activated_any = True
                    continue
                except Exception as exc:
                    raise ValueError(
                        f"LoRA stack activation failed: loader={self.loader_id}; component={component_name}; {exc}"
                    ) from exc
            raise ValueError(
                f"LoRA stack activation failed: loader={self.loader_id}; component={component_name}; "
                "runtime module cannot set adapter names and weights deterministically."
            )
        if not activated_any:
            raise ValueError(
                f"LoRA adapters were loaded but loader={self.loader_id} could not activate any requested component stack."
            )
