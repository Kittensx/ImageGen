from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping

from image_gen.runtime.spatial_requirements import (
    RuntimeSpatialRequirements,
    resolve_runtime_spatial_requirements,
)
from image_gen.systems.validation.capabilities import capability_for
from modules.checkpoint_inspector import build_architecture_contract
from modules.registry.component_selection import canonical_model_family
from modules.sd3_shared_text_encoders import available_shared_text_encoder_sources


GENERATION_CAPABILITY_CONTRACT_VERSION = 1
GENERATION_CAPABILITY_SCHEMA = "image-gen-generation-capabilities-v1"

_REASON_FLOW_MATCH = "flow_match_contract"
_REASON_VP_SIGMA = "vp_sigma_contract"
_REASON_NO_ACTIVE_MODEL = "no_active_model"
_REASON_MODEL_UNRESOLVED = "model_capability_unresolved"
_REASON_ADVANCED_UNRESOLVED = "advanced_model_composition_unresolved"
_REASON_FIXED_CFG_RESCALE = "sd3_flow_match_not_qualified"
_REASON_SUPPORTED_CFG_RESCALE = "cfg_rescale_supported"
_REASON_SAMPLER_DOMAIN_CONSTRAINT = "sampler_prediction_domain_constraint"
_REASON_SAMPLER_DOMAIN_MISMATCH = "sampler_prediction_domain_mismatch"
_REASON_SCHEDULER_DOMAIN_CONSTRAINT = "scheduler_prediction_domain_constraint"
_REASON_SCHEDULER_DOMAIN_MISMATCH = "scheduler_prediction_domain_mismatch"
_REASON_SAMPLER_PAIR_CONSTRAINT = "sampler_scheduler_pair_constraint"
_REASON_SAMPLER_PAIR_MISMATCH = "sampler_scheduler_pair_mismatch"
_REASON_SCHEDULER_PAIR_CONSTRAINT = "scheduler_sampler_pair_constraint"
_REASON_SCHEDULER_PAIR_MISMATCH = "scheduler_sampler_pair_mismatch"
_REASON_PIXEL_ALIGNMENT = "pixel_alignment_required"
_REASON_PIXEL_ALIGNMENT_UNRESOLVED = "pixel_alignment_unresolved"
_REASON_REQUIRED_ASSET_MISSING = "required_asset_missing"
_REASON_NEURAL_HIRES_SUPPORTED = "pixel_neural_hires_supported"
_REASON_VAE_HOOK = "vae_compatibility_not_evaluated"
_REASON_ADAPTER_HOOK = "adapter_qualification_not_evaluated"
_REASON_RUNTIME_ASSET_HOOK = "runtime_asset_prerequisites_declared"
_REASON_EXPERIMENTAL_DISABLED = "experimental_mode_disabled"
_REASON_EXPERIMENTAL_ENABLED = "experimental_mode_enabled"
_REASON_T5_AVAILABLE = "t5_conditioning_available"
_REASON_T5_UNAVAILABLE = "t5_conditioning_unavailable"
_REASON_T5_ENABLED = "t5_conditioning_enabled"
_REASON_T5_DISABLED = "t5_conditioning_disabled"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class GenerationCapabilityContract:
    """Immutable canonical GFP-02 capability representation.

    API callers receive ``to_dict()`` copies, while the authoritative in-process
    representation remains recursively immutable. ``serialize()`` is the stable
    deterministic representation used by acceptance tests and future provenance.
    """

    schema: str
    contract_version: int
    payload: Mapping[str, Any]

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> "GenerationCapabilityContract":
        return cls(
            schema=GENERATION_CAPABILITY_SCHEMA,
            contract_version=GENERATION_CAPABILITY_CONTRACT_VERSION,
            payload=_freeze(dict(payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": int(self.contract_version),
            **_thaw(self.payload),
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


@dataclass(frozen=True)
class _CatalogItem:
    name: str
    label: str
    plugin_id: str
    schedule_domain: str
    preferred_scheduler: str = ""


class GenerationCapabilityService:
    def __init__(self, *, model_selection, catalog, upscaler_catalog, component_selection=None, context=None) -> None:
        self._model_selection = model_selection
        self._catalog = catalog
        self._upscaler_catalog = upscaler_catalog
        self._component_selection = component_selection
        self._context = context

    def resolve_active(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.resolve_contract_active(request=request).to_dict()

    def resolve_contract_active(self, request: dict[str, Any] | None = None) -> GenerationCapabilityContract:
        return self.resolve_contract_for_model(self._model_selection.current_payload(), request=request)

    def resolve_request(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.resolve_contract_for_request(request=request).to_dict()

    def resolve_contract_for_request(self, request: dict[str, Any] | None = None) -> GenerationCapabilityContract:
        normalized = dict(request or {})
        model = self.model_context_for_request(normalized)
        return self.resolve_contract_for_model(model, request=normalized)

    def resolve_for_model(self, active_model: dict[str, Any] | None, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.resolve_contract_for_model(active_model, request=request).to_dict()

    def resolve_contract_for_model(
        self,
        active_model: dict[str, Any] | None,
        request: dict[str, Any] | None = None,
    ) -> GenerationCapabilityContract:
        request = dict(request or {})
        model = dict(active_model or {})
        prediction_domain = self._prediction_domain(model)
        flow_match = prediction_domain == "flow_match"
        unknown_domain = prediction_domain == "unknown"
        spatial = self._spatial_requirements(model)

        samplers = self._catalog_items("samplers")
        schedulers = self._catalog_items("schedulers")
        domain_samplers = self._filter_items_for_domain(samplers, prediction_domain)
        domain_schedulers = self._filter_items_for_domain(schedulers, prediction_domain)
        selected_sampler = str(request.get("sampler_name") or "").strip()
        selected_scheduler = str(request.get("scheduler_name") or "").strip()
        neural_upscalers = self._supported_neural_upscalers()

        architecture = str(model.get("architecture") or "").strip()
        binding_status = "bound" if model.get("resolved_path") and architecture else "unbound"
        model_reason = self._prediction_reason(prediction_domain, binding_status=binding_status)
        qualification = self._qualification_payload(architecture)
        runtime_profile = dict(model.get("runtime_profile") or {})
        allowed_samplers, allowed_schedulers, pair_contract = self._resolve_pair_constraints(
            domain_samplers,
            domain_schedulers,
            selected_sampler=selected_sampler,
            selected_scheduler=selected_scheduler,
            runtime_profile=runtime_profile,
            changed_field=str(request.get("_capability_changed_field") or ""),
        )
        preferred_sampler = self._preferred_sampler_name(allowed_samplers, runtime_profile)
        scheduler_sampler_basis = (
            selected_sampler
            if selected_sampler in {item.name for item in allowed_samplers}
            else preferred_sampler
        )
        preferred_scheduler = self._preferred_scheduler_name(
            allowed_schedulers,
            allowed_samplers,
            scheduler_sampler_basis,
            runtime_profile,
        )
        architecture_contract = dict(model.get("architecture_contract") or {})
        advanced_resolution = dict(model.get("advanced_model_resolution") or {})
        runtime_requirements = list(qualification.get("requirements") or [])
        t5_control = self._text_encoder_3_control(model, request, runtime_profile)

        payload = {
            "binding": {
                "status": binding_status,
                "resolved_path": str(model.get("resolved_path") or model.get("requested_path") or ""),
                "selection_id": str(model.get("selection_id") or request.get("_webui_model_selection_id") or ""),
                "source": str(model.get("source") or ""),
                "reason_code": (
                    advanced_resolution.get("reason_code")
                    or (model_reason if binding_status != "bound" else "model_contract_bound")
                ),
            },
            "model": {
                "architecture": architecture,
                "architecture_variant": str(model.get("architecture_variant") or ""),
                "profile": str(runtime_profile.get("profile_id") or runtime_profile.get("family") or ""),
                "architecture_summary": str(model.get("architecture_summary") or architecture_contract.get("summary") or ""),
                "identity_source": str(
                    model.get("identity_source")
                    or model.get("architecture_source")
                    or architecture_contract.get("source")
                    or ""
                ),
                "fingerprint_verified": bool(model.get("fingerprint_verified", False)),
                "prediction_domain": prediction_domain,
                "denoiser_type": str(architecture_contract.get("denoiser_type") or ""),
                "pixel_alignment_multiple": spatial.pixel_alignment_multiple,
                "qualification_state": qualification["state"],
                "qualification_status": qualification["status"],
            },
            "reason_codes": {
                "prediction_domain": model_reason,
                "qualification": qualification["reason_code"],
            },
            "spatial": spatial.to_dict(),
            "catalog": {
                "allowed_samplers": [item.name for item in allowed_samplers],
                "allowed_schedulers": [item.name for item in allowed_schedulers],
                "allowed_neural_hires_upscalers": [item["upscaler_id"] for item in neural_upscalers],
                "sampler_scheduler_pair": pair_contract,
            },
            "controls": {
                "cfg_rescale": self._cfg_rescale_control(flow_match=flow_match, unknown_domain=unknown_domain),
                "pixel_alignment": self._pixel_alignment_control(spatial),
                "sampler_name": self._selection_control(
                    selected_sampler,
                    allowed_samplers,
                    prediction_domain=prediction_domain,
                    kind="sampler",
                    preferred_name=preferred_sampler,
                    domain_allowed_count=len(domain_samplers),
                    paired_with=str(pair_contract.get("sampler_paired_with") or ""),
                ),
                "scheduler_name": self._selection_control(
                    selected_scheduler,
                    allowed_schedulers,
                    prediction_domain=prediction_domain,
                    kind="scheduler",
                    preferred_name=preferred_scheduler,
                    domain_allowed_count=len(domain_schedulers),
                    paired_with=str(pair_contract.get("scheduler_paired_with") or ""),
                ),
                "hires_strategy": self._hires_strategy_control(request, neural_upscalers),
                "text_encoder_3": t5_control,
            },
            "qualification_hooks": {
                "vae": {
                    "state": "conditional",
                    "reason_code": _REASON_VAE_HOOK,
                    "selected_path": str(request.get("vae_path") or ""),
                    "dependencies": ["architecture", "runtime_profile", "vae_path"],
                    "explanation": "VAE compatibility is represented in the contract but remains a later qualification layer.",
                },
                "adapters": {
                    "state": "recognized_unqualified",
                    "reason_code": _REASON_ADAPTER_HOOK,
                    "types": ["lora"],
                    "dependencies": ["architecture", "adapter_set"],
                    "explanation": "Adapter and LoRA compatibility hooks are reserved for qualified capability expansion.",
                },
                "runtime_assets": {
                    "state": "conditional" if architecture else "unsupported",
                    "reason_code": _REASON_RUNTIME_ASSET_HOOK if architecture else _REASON_MODEL_UNRESOLVED,
                    "prerequisites": runtime_requirements,
                    "dependencies": ["architecture", "runtime_profile"],
                },
                "experimental": {
                    "state": "experimental" if bool(request.get("experimental_mode")) else "hidden",
                    "reason_code": _REASON_EXPERIMENTAL_ENABLED if bool(request.get("experimental_mode")) else _REASON_EXPERIMENTAL_DISABLED,
                    "enabled": bool(request.get("experimental_mode")),
                },
            },
            "expansion_hooks": {
                "sd2": "runtime_profile",
                "sdxl": "runtime_profile",
                "vae": "qualification_hooks.vae",
                "lora": "qualification_hooks.adapters",
            },
        }
        if advanced_resolution:
            payload["advanced_model_resolution"] = advanced_resolution
        return GenerationCapabilityContract.build(payload)

    def model_context_for_request(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        normalized = dict(request or {})
        if not bool(normalized.get("advanced_models_enabled")):
            current = dict(self._model_selection.current_payload() or {})
            requested_path = str(normalized.get("model_path") or "").strip()
            if not requested_path:
                return current

            requested_token = requested_path.replace("\\", "/").casefold()
            current_paths = {
                str(current.get("resolved_path") or "").strip().replace("\\", "/").casefold(),
                str(current.get("requested_path") or "").strip().replace("\\", "/").casefold(),
            }
            if requested_token and requested_token in current_paths:
                return current

            # A remembered checkpoint can be selected before the resident runtime has
            # reactivated it after a backend restart. Capability resolution must not
            # depend on GPU residency: authorize/inspect the requested checkpoint
            # without mutating the active-model selection or loading the model.
            try:
                authorized = self._model_selection.authorize(
                    requested_path,
                    source="generation_capability_request",
                )
                return dict(authorized.to_dict())
            except (OSError, ValueError, RuntimeError):
                return current
        if self._component_selection is None:
            return self._unresolved_advanced_context(
                "Advanced Models capability resolution is unavailable because no component-selection service is bound."
            )
        try:
            resolved = self._component_selection.resolve_selection(
                str(normalized.get("advanced_model_family") or ""),
                normalized.get("advanced_model_components") or {},
                t5_device=normalized.get("advanced_model_t5_device") or "cpu",
                allow_digital_components=bool(normalized.get("advanced_model_allow_digital_components", True)),
            )
            return self.model_context_for_advanced_composition(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return self._unresolved_advanced_context(str(exc))

    def model_context_for_advanced_composition(self, resolved_composition: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(resolved_composition or {})
        family = canonical_model_family(resolved.get("family"))
        base_path = str(resolved.get("base_source_path") or "").strip()
        if not family or not base_path:
            raise ValueError("Advanced Models resolved composition is missing authoritative family or base source evidence.")

        authorized = self._model_selection.authorize(base_path, source="advanced_models_capability")
        payload = authorized.to_dict()
        inspected_family = canonical_model_family(payload.get("architecture"))
        if inspected_family and inspected_family != family:
            raise ValueError(
                "Advanced Models component-registry family conflicts with checkpoint-header architecture: "
                f"registry={family!r}, checkpoint={inspected_family!r}."
            )

        architecture_contract = dict(payload.get("architecture_contract") or {})
        if not architecture_contract:
            architecture_contract = build_architecture_contract(
                family,
                prediction_type=payload.get("prediction_type"),
                conditioning_dimension=payload.get("conditioning_dimension"),
                source="advanced_model_component_registry",
            ).to_dict()

        composition_hash = str(resolved.get("composition_sha256") or "").strip().lower()
        selection_id = f"advanced-{composition_hash[:12]}" if composition_hash else f"advanced-{authorized.selection_id}"
        payload.update(
            {
                "selection_id": selection_id,
                "requested_path": "",
                "resolved_path": base_path,
                "model_name": f"Advanced {resolved.get('family_label') or family} composition",
                "model_name_source": "advanced_component_composition",
                "source": "advanced_models",
                "status": "ready",
                "architecture": family,
                "architecture_summary": f"Advanced component composition / {resolved.get('family_label') or family}",
                "architecture_source": "component_registry+checkpoint_header",
                "checkpoint_kind": "component_composition",
                "architecture_contract": architecture_contract,
                "advanced_model_composition": resolved,
                "advanced_model_resolution": {
                    "state": "supported",
                    "reason_code": "advanced_model_composition_resolved",
                    "composition_sha256": composition_hash,
                    "provider_version": str(resolved.get("provider_version") or ""),
                },
            }
        )
        return payload

    @staticmethod
    def _unresolved_advanced_context(explanation: str) -> dict[str, Any]:
        return {
            "selection_id": "",
            "requested_path": "",
            "resolved_path": "",
            "source": "advanced_models",
            "status": "unresolved",
            "architecture": "",
            "architecture_contract": {},
            "runtime_profile": {},
            "advanced_model_resolution": {
                "state": "unsupported",
                "reason_code": _REASON_ADVANCED_UNRESOLVED,
                "explanation": str(explanation or "Advanced Models composition is unresolved."),
            },
        }

    def enforce_request(self, payload: dict[str, Any], *, active_model: dict[str, Any] | None = None) -> dict[str, Any]:
        request = dict(payload or {})
        if not self._request_bool(request.get("advanced_models_enabled")):
            # Keep disabled Advanced Models preferences available to the browser, but
            # never let an internal resolved composition from an earlier job leak into
            # whole-checkpoint execution. The composition hash is load identity, so it
            # must also be empty while Advanced Models is off.
            request.pop("_advanced_model_resolved", None)
            request["advanced_model_composition_sha256"] = ""
        contract = self.resolve_for_model(active_model or {}, request=request)
        spatial = dict(contract.get("spatial") or {})
        request["_generation_spatial_requirements"] = {
            "latent_scale_factor": int(spatial.get("latent_scale_factor") or 0),
            "latent_patch_multiple": int(spatial.get("latent_patch_multiple") or 0),
            "pixel_alignment_multiple": int(spatial.get("pixel_alignment_multiple") or 0),
            "source": str(spatial.get("source") or ""),
            "contract_schema": GENERATION_CAPABILITY_SCHEMA,
            "contract_version": GENERATION_CAPABILITY_CONTRACT_VERSION,
        }

        cfg_control = contract["controls"]["cfg_rescale"]
        if cfg_control["state"] == "fixed":
            request["cfg_rescale"] = float(cfg_control["effective_value"])
            if request.get("hires_enabled") or "hires_cfg_rescale" in request:
                request["hires_cfg_rescale"] = float(cfg_control["effective_value"])

        sampler_control = contract["controls"]["sampler_name"]
        scheduler_control = contract["controls"]["scheduler_name"]
        if request.get("sampler_name") and sampler_control["selection_state"] == "unsupported":
            raise ValueError(
                f"Sampler is incompatible with the active model contract ({sampler_control['reason_code']})."
            )
        if request.get("scheduler_name") and scheduler_control["selection_state"] == "unsupported":
            raise ValueError(
                f"Scheduler is incompatible with the active model contract ({scheduler_control['reason_code']})."
            )

        if not bool(request.get("advanced_models_enabled")):
            t5_control = contract["controls"].get("text_encoder_3") or {}
            t5_requested = self._request_bool(request.get("sd3_t5_enabled"), default=bool(t5_control.get("default_enabled")))
            if t5_requested and not bool(t5_control.get("available")):
                raise ValueError(
                    f"T5/T5XXL conditioning was requested but no qualified source is available ({t5_control.get('reason_code') or _REASON_T5_UNAVAILABLE})."
                )
            if bool(t5_control.get("available")):
                request["sd3_t5_enabled"] = bool(t5_requested)
                if t5_requested:
                    allowed_sources = {
                        str(item.get("value") or "")
                        for item in (t5_control.get("source_options") or [])
                        if str(item.get("value") or "")
                    }
                    requested_source = str(
                        request.get("sd3_t5_source")
                        or t5_control.get("effective_source")
                        or t5_control.get("default_source")
                        or "auto"
                    ).strip().lower()
                    if requested_source == "auto":
                        requested_source = str(t5_control.get("default_source") or t5_control.get("effective_source") or "").strip().lower()
                    if requested_source not in allowed_sources:
                        raise ValueError(
                            "The selected T5/T5XXL source is not available for the active model capability contract "
                            f"({_REASON_T5_UNAVAILABLE})."
                        )
                    request["sd3_t5_source"] = requested_source
                    requested_device = str(request.get("text_encoder_3_device") or "auto").strip().lower()
                    request["text_encoder_3_device"] = requested_device if requested_device in {"auto", "cpu", "cuda"} else "auto"
                else:
                    request["sd3_t5_source"] = "auto"
                    request["text_encoder_3_device"] = "off"
            else:
                request["sd3_t5_enabled"] = False
                request["sd3_t5_source"] = "auto"
                request["text_encoder_3_device"] = "off"
        return request

    @staticmethod
    def _prediction_domain(model: Mapping[str, Any]) -> str:
        contract = dict(model.get("architecture_contract") or {})
        domain = str(contract.get("denoising_domain") or contract.get("prediction_domain") or "").strip().casefold()
        if domain in {"flow_match", "vp_sigma"}:
            return domain
        runtime_profile = dict(model.get("runtime_profile") or {})
        runtime_domain = str(runtime_profile.get("denoising_domain") or runtime_profile.get("prediction_domain") or "").strip().casefold()
        if runtime_domain in {"flow_match", "vp_sigma"}:
            return runtime_domain
        return "unknown"

    @staticmethod
    def _spatial_requirements(model: Mapping[str, Any]) -> RuntimeSpatialRequirements:
        contract = dict(model.get("architecture_contract") or {})
        runtime_profile = dict(model.get("runtime_profile") or {})
        return resolve_runtime_spatial_requirements(
            denoiser_kind=str(contract.get("denoiser_type") or ""),
            runtime_profile=runtime_profile,
            latent_scale_factor=(
                contract.get("latent_scale_factor")
                or runtime_profile.get("latent_scale_factor")
            ),
            latent_patch_multiple=contract.get("latent_patch_multiple"),
            fail_closed=True,
        )

    @staticmethod
    def _prediction_reason(prediction_domain: str, *, binding_status: str) -> str:
        if prediction_domain == "flow_match":
            return _REASON_FLOW_MATCH
        if prediction_domain == "vp_sigma":
            return _REASON_VP_SIGMA
        return _REASON_NO_ACTIVE_MODEL if binding_status == "unbound" else _REASON_MODEL_UNRESOLVED

    @staticmethod
    def _qualification_payload(architecture: str) -> dict[str, Any]:
        normalized = str(architecture or "").strip().casefold()
        capability = capability_for(normalized)
        if not normalized:
            state = "unsupported"
            reason_code = _REASON_NO_ACTIVE_MODEL
        elif str(capability.architecture) == "unknown":
            state = "recognized_unqualified"
            reason_code = _REASON_MODEL_UNRESOLVED
        elif capability.generation_supported:
            state = "supported"
            reason_code = "architecture_generation_qualified"
        else:
            state = "recognized_unqualified"
            reason_code = "architecture_generation_unqualified"
        return {
            "state": state,
            "status": str(capability.status),
            "reason_code": reason_code,
            "reason": str(capability.reason),
            "requirements": list(capability.requirements),
        }

    @staticmethod
    def _request_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if token in {"0", "false", "no", "off", "disabled", "disable", ""}:
            return False
        return bool(default)

    def _text_encoder_3_control(
        self,
        model: Mapping[str, Any],
        request: Mapping[str, Any],
        runtime_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        roles = {str(item or "").strip().casefold() for item in (runtime_profile.get("conditioning_roles") or [])}
        supports_t5 = bool({"t5", "t5xxl", "text_encoder_3"}.intersection(roles))
        if bool(request.get("advanced_models_enabled")):
            return {
                "state": "hidden",
                "reason_code": "advanced_model_component_control_owned",
                "available": False,
                "enabled": False,
                "default_enabled": False,
                "dependencies": ["advanced_models_enabled", "advanced_model_components"],
                "explanation": "Advanced Models owns T5 component selection for this composition.",
            }
        if not supports_t5:
            return {
                "state": "hidden",
                "reason_code": _REASON_T5_UNAVAILABLE,
                "available": False,
                "enabled": False,
                "default_enabled": False,
                "dependencies": ["runtime_profile.conditioning_roles"],
                "explanation": "The active runtime profile does not declare an optional T5/T5XXL conditioning role.",
            }

        embedded = bool(model.get("has_t5") or model.get("has_text_encoder_3"))
        source_options: list[dict[str, Any]] = []
        if embedded:
            source_options.append({
                "value": "embedded",
                "label": "Embedded T5 / T5XXL (checkpoint)",
                "source_kind": "embedded",
                "source_path": str(model.get("resolved_path") or model.get("model_path") or ""),
                "source_layout": "checkpoint",
                "component_sha256": "",
                "asset_sha256": "",
            })

        external_options: list[dict[str, Any]] = []
        if self._context is not None:
            try:
                external_options = [
                    dict(item.to_dict())
                    for item in available_shared_text_encoder_sources(self._context, "t5xxl")
                ]
            except (OSError, ValueError, RuntimeError):
                external_options = []
        source_options.extend(external_options)

        shared_available = bool(external_options)
        available = bool(source_options)
        default_enabled = bool(embedded)
        default_source = "embedded" if embedded else str((source_options[0].get("value") if source_options else "") or "")
        enabled = self._request_bool(request.get("sd3_t5_enabled"), default=default_enabled) if available else False
        requested_source = str(request.get("sd3_t5_source") or "auto").strip().lower()
        allowed_sources = {str(item.get("value") or "") for item in source_options}
        if requested_source == "auto" or requested_source not in allowed_sources:
            effective_source = default_source
        else:
            effective_source = requested_source

        if not available:
            return {
                "state": "hidden",
                "reason_code": _REASON_T5_UNAVAILABLE,
                "available": False,
                "enabled": False,
                "default_enabled": False,
                "default_source": "",
                "effective_source": "",
                "source_options": [],
                "embedded_available": False,
                "shared_available": False,
                "dependencies": ["runtime_profile.conditioning_roles", "text_encoder_assets"],
                "explanation": "T5/T5XXL conditioning is supported by this runtime profile, but no embedded or qualified shared T5 asset is currently resolvable.",
            }

        selected_option = next(
            (item for item in source_options if str(item.get("value") or "") == effective_source),
            source_options[0],
        )
        return {
            "state": "supported",
            "reason_code": _REASON_T5_ENABLED if enabled else _REASON_T5_AVAILABLE,
            "available": True,
            "enabled": bool(enabled),
            "default_enabled": default_enabled,
            "default_source": default_source,
            "effective_source": effective_source,
            "source_options": source_options,
            "embedded_available": embedded,
            "shared_available": shared_available,
            "shared_path": str(selected_option.get("source_path") or "") if str(selected_option.get("source_kind") or "") == "external" else "",
            "shared_layout": str(selected_option.get("source_layout") or "") if str(selected_option.get("source_kind") or "") == "external" else "",
            "allowed_devices": ["auto", "cpu", "cuda"],
            "dependencies": ["runtime_profile.conditioning_roles", "checkpoint_text_encoder_packaging", "text_encoder_assets"],
            "explanation": "Optional T5/T5XXL conditioning is available. Enable it and choose one of the qualified T5 sources exposed by this capability contract.",
        }

    def _catalog_items(self, kind: str) -> list[_CatalogItem]:
        payload = self._catalog.plugins() if callable(getattr(self._catalog, "plugins", None)) else {}
        items: list[_CatalogItem] = []
        for record in payload.get(kind, []) or []:
            metadata = dict(record.get("metadata") or {})
            capabilities = dict(record.get("capabilities") or {})
            domain = str(
                capabilities.get("schedule_domain")
                or metadata.get("schedule_domain")
                or capabilities.get("prediction_domain")
                or metadata.get("prediction_domain")
                or "unknown"
            ).strip().casefold()
            items.append(
                _CatalogItem(
                    name=str(record.get("name") or ""),
                    label=str(record.get("label") or record.get("name") or ""),
                    plugin_id=str(record.get("plugin_id") or ""),
                    schedule_domain=domain,
                    preferred_scheduler=str(capabilities.get("preferred_scheduler") or ""),
                )
            )
        return sorted(items, key=lambda item: (item.name.casefold(), item.plugin_id.casefold()))

    @staticmethod
    def _filter_items_for_domain(items: list[_CatalogItem], prediction_domain: str) -> list[_CatalogItem]:
        domain = str(prediction_domain or "unknown").strip().casefold() or "unknown"
        if domain not in {"vp_sigma", "flow_match"}:
            return []
        return [item for item in items if item.schedule_domain == domain]

    def _pair_compatible(self, sampler: _CatalogItem, scheduler: _CatalogItem) -> bool:
        validate = getattr(self._catalog, "validate_pair", None)
        if not callable(validate):
            # Lightweight test/fallback catalogs predate pair-level capability
            # negotiation.  Domain filtering remains fail-closed for model
            # compatibility, while production catalogs always expose the
            # canonical runtime-registry validator.
            return True
        try:
            result = validate(sampler.name, scheduler.name)
        except Exception:
            return False
        if isinstance(result, Mapping):
            return bool(result.get("is_compatible", False))
        return bool(getattr(result, "is_compatible", False))

    def _samplers_for_scheduler(
        self,
        samplers: list[_CatalogItem],
        scheduler: _CatalogItem,
    ) -> list[_CatalogItem]:
        return [item for item in samplers if self._pair_compatible(item, scheduler)]

    def _schedulers_for_sampler(
        self,
        schedulers: list[_CatalogItem],
        sampler: _CatalogItem,
    ) -> list[_CatalogItem]:
        return [item for item in schedulers if self._pair_compatible(sampler, item)]

    def _resolve_pair_constraints(
        self,
        domain_samplers: list[_CatalogItem],
        domain_schedulers: list[_CatalogItem],
        *,
        selected_sampler: str,
        selected_scheduler: str,
        runtime_profile: Mapping[str, Any],
        changed_field: str = "",
    ) -> tuple[list[_CatalogItem], list[_CatalogItem], dict[str, Any]]:
        validate = getattr(self._catalog, "validate_pair", None)
        if not callable(validate):
            return domain_samplers, domain_schedulers, {
                "pair_validation_available": False,
                "selected_pair_compatible": None,
                "sampler_paired_with": "",
                "scheduler_paired_with": "",
                "selection_priority": "",
            }

        sampler_by_name = {item.name: item for item in domain_samplers}
        scheduler_by_name = {item.name: item for item in domain_schedulers}
        selected_sampler_item = sampler_by_name.get(selected_sampler)
        selected_scheduler_item = scheduler_by_name.get(selected_scheduler)
        profile_sampler = str(runtime_profile.get("sampler_name") or "").strip()
        profile_scheduler = str(runtime_profile.get("scheduler_name") or "").strip()
        priority = str(changed_field or "").strip()
        if priority not in {"sampler_name", "scheduler_name"}:
            if (
                selected_scheduler_item is not None
                and selected_scheduler == profile_scheduler
                and profile_sampler in sampler_by_name
                and (selected_sampler_item is None or not self._pair_compatible(selected_sampler_item, selected_scheduler_item))
            ):
                priority = "scheduler_name"
            else:
                priority = "sampler_name"

        selected_pair_compatible: bool | None = None
        if selected_sampler_item is not None and selected_scheduler_item is not None:
            selected_pair_compatible = self._pair_compatible(selected_sampler_item, selected_scheduler_item)

        allowed_samplers = list(domain_samplers)
        allowed_schedulers = list(domain_schedulers)
        sampler_paired_with = ""
        scheduler_paired_with = ""

        if selected_pair_compatible is True:
            allowed_samplers = self._samplers_for_scheduler(domain_samplers, selected_scheduler_item)
            allowed_schedulers = self._schedulers_for_sampler(domain_schedulers, selected_sampler_item)
            sampler_paired_with = selected_scheduler_item.label
            scheduler_paired_with = selected_sampler_item.label
        elif selected_sampler_item is not None and selected_scheduler_item is not None:
            if priority == "scheduler_name":
                allowed_samplers = self._samplers_for_scheduler(domain_samplers, selected_scheduler_item)
                sampler_basis_name = (
                    profile_sampler
                    if profile_sampler in {item.name for item in allowed_samplers}
                    else self._preferred_sampler_name(allowed_samplers, runtime_profile)
                )
                sampler_basis = next((item for item in allowed_samplers if item.name == sampler_basis_name), None)
                if sampler_basis is not None:
                    allowed_schedulers = self._schedulers_for_sampler(domain_schedulers, sampler_basis)
                    scheduler_paired_with = sampler_basis.label
                sampler_paired_with = selected_scheduler_item.label
            else:
                allowed_schedulers = self._schedulers_for_sampler(domain_schedulers, selected_sampler_item)
                scheduler_basis_name = self._preferred_scheduler_name(
                    allowed_schedulers,
                    [selected_sampler_item],
                    selected_sampler_item.name,
                    runtime_profile,
                )
                scheduler_basis = next((item for item in allowed_schedulers if item.name == scheduler_basis_name), None)
                if scheduler_basis is not None:
                    allowed_samplers = self._samplers_for_scheduler(domain_samplers, scheduler_basis)
                    sampler_paired_with = scheduler_basis.label
                scheduler_paired_with = selected_sampler_item.label
        elif selected_scheduler_item is not None:
            allowed_samplers = self._samplers_for_scheduler(domain_samplers, selected_scheduler_item)
            sampler_paired_with = selected_scheduler_item.label
        elif selected_sampler_item is not None:
            allowed_schedulers = self._schedulers_for_sampler(domain_schedulers, selected_sampler_item)
            scheduler_paired_with = selected_sampler_item.label

        return allowed_samplers, allowed_schedulers, {
            "pair_validation_available": True,
            "selected_pair_compatible": selected_pair_compatible,
            "sampler_paired_with": sampler_paired_with,
            "scheduler_paired_with": scheduler_paired_with,
            "selection_priority": priority,
        }

    def _supported_neural_upscalers(self) -> list[dict[str, Any]]:
        payload = self._upscaler_catalog.payload() if callable(getattr(self._upscaler_catalog, "payload", None)) else {}
        items = payload.get("supported_neural") or []
        selectable = [dict(item) for item in items if bool(item.get("selectable", True))]
        return sorted(selectable, key=lambda item: str(item.get("upscaler_id") or "").casefold())

    @staticmethod
    def _cfg_rescale_control(*, flow_match: bool, unknown_domain: bool) -> dict[str, Any]:
        if flow_match:
            return {
                "state": "fixed",
                "reason_code": _REASON_FIXED_CFG_RESCALE,
                "effective_value": 0.0,
                "allowed_range": [0.0, 0.0],
                "dependencies": ["prediction_domain"],
                "explanation": "The active flow-match model contract fixes CFG rescale at 0.0.",
            }
        if unknown_domain:
            return {
                "state": "unsupported",
                "reason_code": _REASON_MODEL_UNRESOLVED,
                "effective_value": None,
                "allowed_range": [0.0, 1.0],
                "dependencies": ["prediction_domain"],
                "explanation": "CFG rescale is unavailable until the model prediction domain is authoritative.",
            }
        return {
            "state": "supported",
            "reason_code": _REASON_SUPPORTED_CFG_RESCALE,
            "effective_value": None,
            "allowed_range": [0.0, 1.0],
            "dependencies": ["prediction_domain"],
            "explanation": "CFG rescale is available for the active VP-sigma model contract.",
        }

    @staticmethod
    def _pixel_alignment_control(spatial: RuntimeSpatialRequirements) -> dict[str, Any]:
        if spatial.pixel_alignment_multiple <= 0:
            return {
                "state": "unsupported",
                "reason_code": _REASON_PIXEL_ALIGNMENT_UNRESOLVED,
                "effective_multiple": 0,
                "dependencies": ["denoiser_type", "runtime_profile"],
                "explanation": "Pixel alignment is unresolved; spatial validation must fail closed.",
            }
        return {
            "state": "constrained",
            "reason_code": _REASON_PIXEL_ALIGNMENT,
            "effective_multiple": spatial.pixel_alignment_multiple,
            "latent_scale_factor": spatial.latent_scale_factor,
            "latent_patch_multiple": spatial.latent_patch_multiple,
            "dependencies": ["denoiser_type", "runtime_profile", "vae"],
            "explanation": f"Image dimensions must align to {spatial.pixel_alignment_multiple} pixels for this runtime contract.",
        }

    @staticmethod
    def _preferred_sampler_name(allowed_items: list[_CatalogItem], runtime_profile: Mapping[str, Any]) -> str:
        allowed_names = [item.name for item in allowed_items]
        profile_name = str(runtime_profile.get("sampler_name") or "").strip()
        if profile_name in allowed_names:
            return profile_name
        for conventional in ("kes", "simple_euler", "dpmpp_2m"):
            if conventional in allowed_names:
                return conventional
        return allowed_names[0] if allowed_names else ""

    @staticmethod
    def _preferred_scheduler_name(
        allowed_items: list[_CatalogItem],
        allowed_samplers: list[_CatalogItem],
        sampler_name: str,
        runtime_profile: Mapping[str, Any],
    ) -> str:
        allowed_names = [item.name for item in allowed_items]
        profile_name = str(runtime_profile.get("scheduler_name") or "").strip()
        if profile_name in allowed_names:
            return profile_name
        sampler = next((item for item in allowed_samplers if item.name == sampler_name), None)
        if sampler and sampler.preferred_scheduler in allowed_names:
            return sampler.preferred_scheduler
        for conventional in ("simple_kes", "standard_karras"):
            if conventional in allowed_names:
                return conventional
        return allowed_names[0] if allowed_names else ""

    @staticmethod
    def _hires_strategy_control(request: Mapping[str, Any], neural_upscalers: list[dict[str, Any]]) -> dict[str, Any]:
        allowed_ids = [str(item.get("upscaler_id") or "") for item in neural_upscalers if str(item.get("upscaler_id") or "").strip()]
        enabled = bool(request.get("hires_enabled"))
        strategy = str(request.get("hires_strategy") or "pixel_neural").strip().casefold() or "pixel_neural"
        selected = str(request.get("hires_upscaler_id") or request.get("hires_upscaler") or "").strip()
        neural_available = bool(allowed_ids)
        selection_state = "inactive"
        blocking = False
        reason_code = _REASON_NEURAL_HIRES_SUPPORTED if neural_available else _REASON_REQUIRED_ASSET_MISSING
        explanation = "Pixel-neural hires has at least one selectable neural upscaler." if neural_available else "Pixel-neural hires requires a selectable neural upscaler asset."
        if enabled and strategy == "pixel_neural":
            if selected and selected in allowed_ids:
                selection_state = "compatible"
            else:
                selection_state = "needs_action"
                blocking = True
                reason_code = _REASON_REQUIRED_ASSET_MISSING
                explanation = "Pixel-neural hires requires an installed qualified upscaler with a stable ID."
        return {
            "state": "conditional",
            "reason_code": reason_code,
            "dependencies": ["hires_enabled", "hires_strategy", "hires_upscaler_id"],
            "selection_state": selection_state,
            "selected_strategy": strategy,
            "selected_upscaler_id": selected,
            "blocking": blocking,
            "control_id": "hiresUpscaler",
            "explanation": explanation,
            "strategies": {
                "pixel_neural": {
                    "state": "supported" if neural_available else "unsupported",
                    "reason_code": _REASON_NEURAL_HIRES_SUPPORTED if neural_available else _REASON_REQUIRED_ASSET_MISSING,
                    "allowed_upscaler_ids": allowed_ids,
                    "dependencies": ["hires_upscaler_id"],
                    "explanation": (
                        "Pixel-neural hires has at least one selectable neural upscaler."
                        if neural_available
                        else "Pixel-neural hires requires a selectable neural upscaler asset."
                    ),
                },
            },
        }

    @staticmethod
    def _selection_control(
        selected_name: str,
        allowed_items: list[_CatalogItem],
        *,
        prediction_domain: str,
        kind: str,
        preferred_name: str = "",
        domain_allowed_count: int | None = None,
        paired_with: str = "",
    ) -> dict[str, Any]:
        allowed_names = [item.name for item in allowed_items]
        if not selected_name:
            selection_state = "compatible" if allowed_names else "unresolved"
        elif selected_name in allowed_names:
            selection_state = "compatible"
        else:
            selection_state = "unsupported"

        is_sampler = kind == "sampler"
        domain_constraint_reason = _REASON_SAMPLER_DOMAIN_CONSTRAINT if is_sampler else _REASON_SCHEDULER_DOMAIN_CONSTRAINT
        domain_mismatch_reason = _REASON_SAMPLER_DOMAIN_MISMATCH if is_sampler else _REASON_SCHEDULER_DOMAIN_MISMATCH
        pair_constraint_reason = _REASON_SAMPLER_PAIR_CONSTRAINT if is_sampler else _REASON_SCHEDULER_PAIR_CONSTRAINT
        pair_mismatch_reason = _REASON_SAMPLER_PAIR_MISMATCH if is_sampler else _REASON_SCHEDULER_PAIR_MISMATCH
        pair_constrained = bool(paired_with) and (domain_allowed_count is None or len(allowed_names) < int(domain_allowed_count))
        if prediction_domain not in {"vp_sigma", "flow_match"} or not allowed_names:
            state = "unsupported"
            reason_code = _REASON_MODEL_UNRESOLVED
        elif selection_state == "unsupported":
            state = "constrained"
            reason_code = pair_mismatch_reason if pair_constrained else domain_mismatch_reason
        else:
            state = "constrained"
            reason_code = pair_constraint_reason if pair_constrained else domain_constraint_reason
        explanation = (
            f"{kind.title()} choices are constrained by compatibility with {paired_with}."
            if pair_constrained
            else f"{kind.title()} choices are constrained by the active {prediction_domain} model contract."
        )
        return {
            "state": state,
            "reason_code": reason_code,
            "allowed_count": len(allowed_names),
            "allowed_names": allowed_names,
            "selection_state": selection_state,
            "selected_name": selected_name,
            "preferred_name": preferred_name if preferred_name in allowed_names else "",
            "replacement_name": (
                preferred_name if preferred_name in allowed_names and selection_state != "compatible" else ""
            ),
            "replacement_reason": (
                "runtime_profile_or_registry_default"
                if preferred_name in allowed_names and selection_state != "compatible"
                else ""
            ),
            "prediction_domain": prediction_domain,
            "paired_with": paired_with if pair_constrained else "",
            "explanation": explanation,
            "dependencies": [
                "prediction_domain",
                f"{kind}_registry",
                *(
                    ["scheduler_name" if is_sampler else "sampler_name"]
                    if pair_constrained
                    else []
                ),
            ],
        }


__all__ = [
    "GENERATION_CAPABILITY_CONTRACT_VERSION",
    "GENERATION_CAPABILITY_SCHEMA",
    "GenerationCapabilityContract",
    "GenerationCapabilityService",
]
