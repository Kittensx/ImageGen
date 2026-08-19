from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from image_gen.systems.image_conditioning import resolve_image_conditioned_step_plan

from .builtins import (
    HIRES_AUTO_PROFILE_VERSION,
    builtin_auto_profile_id,
    builtin_auto_profile_name,
    builtin_auto_profile_values,
)

AUTO_SELECT = "auto"
BUILTIN_PIXEL_RESIZE_ID = "builtin.pixel_resize.bicubic"
BUILTIN_PIXEL_RESIZE_SHA256 = hashlib.sha256(
    b"image-gen:builtin:pixel_resize:bicubic:v1"
).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def normalize_family(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "sdxl" in text:
        return "sdxl"
    if text.startswith("sd3") or "stable diffusion 3" in text:
        return "sd3.x"
    if text.startswith("sd2") or "stable diffusion 2" in text:
        return "sd2.x"
    if text.startswith("sd1") or "stable diffusion 1" in text:
        return "sd1.x"
    return text


@dataclass(frozen=True)
class HiresResolutionContext:
    model_family: str
    checkpoint_sha256: str = ""
    width: int = 512
    height: int = 512
    requested_scale: float = 2.0
    requested_width: int = 0
    requested_height: int = 0
    explicit_user_upscaler: str = ""
    preferred_user_upscaler: str = ""
    base_values: dict[str, Any] = field(default_factory=dict)
    runtime_profile: dict[str, Any] = field(default_factory=dict)
    vae_contract: dict[str, Any] = field(default_factory=dict)
    available_upscalers: tuple[dict[str, Any], ...] = ()
    available_samplers: tuple[str, ...] = ()
    available_schedulers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        available_upscalers: Sequence[Mapping[str, Any]] = (),
        available_samplers: Sequence[Any] = (),
        available_schedulers: Sequence[Any] = (),
    ) -> "HiresResolutionContext":
        raw = dict(payload or {})
        dims = dict(raw.get("current_dimensions") or {})
        target = dict(raw.get("requested_target") or {})
        def _names(values: Sequence[Any]) -> tuple[str, ...]:
            out: list[str] = []
            for item in values:
                if isinstance(item, Mapping):
                    value = item.get("value") or item.get("name") or item.get("plugin_id")
                else:
                    value = item
                if value not in (None, ""):
                    out.append(str(value))
            return tuple(dict.fromkeys(out))
        return cls(
            model_family=normalize_family(raw.get("model_family") or raw.get("family")),
            checkpoint_sha256=str(raw.get("checkpoint_sha256") or raw.get("checkpoint_identity") or "").strip().casefold(),
            width=max(1, int(dims.get("width") or raw.get("width") or 512)),
            height=max(1, int(dims.get("height") or raw.get("height") or 512)),
            requested_scale=float(raw.get("requested_scale") or 2.0),
            requested_width=max(0, int(target.get("width") or raw.get("requested_width") or 0)),
            requested_height=max(0, int(target.get("height") or raw.get("requested_height") or 0)),
            explicit_user_upscaler=str(raw.get("explicit_user_upscaler") or "").strip(),
            preferred_user_upscaler=str(raw.get("preferred_user_upscaler") or "").strip(),
            base_values=dict(raw.get("base_values") or {}),
            runtime_profile=dict(raw.get("runtime_profile") or {}),
            vae_contract=dict(raw.get("vae_contract") or {}),
            available_upscalers=tuple(dict(item) for item in available_upscalers),
            available_samplers=_names(available_samplers),
            available_schedulers=_names(available_schedulers),
        )


@dataclass(frozen=True)
class HiresResolutionResult:
    values: dict[str, Any]
    field_sources: dict[str, dict[str, Any]]
    applied_profiles: tuple[dict[str, Any], ...]
    applied_assignments: tuple[dict[str, Any], ...]
    selected_upscaler: dict[str, Any]
    valid: bool
    warnings: tuple[str, ...]
    unresolved_requirements: tuple[str, ...]
    resolution_fingerprint: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "auto",
            "values": _json_safe(self.values),
            "field_sources": _json_safe(self.field_sources),
            "applied_profiles": _json_safe(self.applied_profiles),
            "applied_assignments": _json_safe(self.applied_assignments),
            "selected_upscaler": _json_safe(self.selected_upscaler),
            "validation": {
                "valid": bool(self.valid),
                "unresolved_requirements": list(self.unresolved_requirements),
            },
            "warnings": list(self.warnings),
            "resolution_fingerprint": self.resolution_fingerprint,
            "diagnostics": _json_safe(self.diagnostics),
        }


def builtin_resize_descriptor() -> dict[str, Any]:
    return {
        "value": BUILTIN_PIXEL_RESIZE_ID,
        "upscaler_id": BUILTIN_PIXEL_RESIZE_ID,
        "label": "Built-in Bicubic Resize",
        "display_name": "Built-in Bicubic Resize",
        "sha256": BUILTIN_PIXEL_RESIZE_SHA256,
        "selectable": True,
        "available": True,
        "native_scale": 0,
        "architecture": "interpolation",
        "strategy": "pixel_resize",
        "builtin": True,
    }


def _assignment_matches(assignment: Any, context: HiresResolutionContext, upscaler_sha: str = "") -> bool:
    scope = str(assignment.scope).casefold()
    family = normalize_family(context.model_family)
    if scope == "global":
        return True
    if scope == "model_family":
        return normalize_family(assignment.model_family) == family
    if scope == "checkpoint":
        return bool(context.checkpoint_sha256) and assignment.checkpoint_sha256.casefold() == context.checkpoint_sha256
    if scope == "upscaler":
        return bool(upscaler_sha) and assignment.upscaler_sha256.casefold() == upscaler_sha
    if scope == "model_family_upscaler":
        return (
            bool(upscaler_sha)
            and normalize_family(assignment.model_family) == family
            and assignment.upscaler_sha256.casefold() == upscaler_sha
        )
    if scope == "checkpoint_upscaler":
        return (
            bool(upscaler_sha)
            and bool(context.checkpoint_sha256)
            and assignment.checkpoint_sha256.casefold() == context.checkpoint_sha256
            and assignment.upscaler_sha256.casefold() == upscaler_sha
        )
    return False


_SCOPE_PRECEDENCE = {
    "global": 10,
    "upscaler": 20,
    "model_family": 30,
    "checkpoint": 40,
    "model_family_upscaler": 50,
    "checkpoint_upscaler": 60,
}


class HiresAutoResolver:
    def __init__(self, service) -> None:
        self.service = service

    @staticmethod
    def _available_upscalers(context: HiresResolutionContext) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for raw in context.available_upscalers:
            item = dict(raw)
            item_id = str(item.get("value") or item.get("upscaler_id") or "").strip()
            if not item_id or item_id == AUTO_SELECT:
                continue
            if item.get("available", item.get("selectable", True)) is False:
                continue
            item["upscaler_id"] = item_id
            item.setdefault("value", item_id)
            item.setdefault("strategy", "pixel_neural")
            output.append(item)
        if not any(item.get("upscaler_id") == BUILTIN_PIXEL_RESIZE_ID for item in output):
            output.append(builtin_resize_descriptor())
        return output

    def _profile_layers(self, context: HiresResolutionContext, *, upscaler_sha: str = "") -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for assignment in self.service.list_default_assignments():
            if not _assignment_matches(assignment, context, upscaler_sha):
                continue
            profile = self.service.get_profile(assignment.profile_id)
            if profile is not None:
                rows.append((assignment, profile))
        return sorted(rows, key=lambda pair: (_SCOPE_PRECEDENCE.get(str(pair[0].scope).casefold(), 0), pair[0].assignment_key))

    @staticmethod
    def _builtin_identity(family: str) -> tuple[str, str]:
        normalized = normalize_family(family)
        return builtin_auto_profile_id(normalized), builtin_auto_profile_name(normalized)

    @staticmethod
    def _refinement_policy_summary(values: Mapping[str, Any], *, family: str) -> dict[str, Any]:
        requested_steps = max(1, int(values.get("hires_steps") or 1))
        strength = float(values.get("hires_denoising_strength") or 1.0)
        step_policy = str(values.get("hires_step_policy") or "a1111_fixed_steps_v1").strip().lower()
        plan = resolve_image_conditioned_step_plan(
            requested_refinement_steps=requested_steps,
            denoising_strength=strength,
            step_policy=step_policy,
        )
        explanation = (
            f"User-facing hires steps target {plan.effective_refinement_steps} active second-pass evaluations; "
            f"the runtime constructs an internal {plan.internal_schedule_steps}-step schedule to preserve "
            f"denoising strength {plan.normalized_denoising_strength:.3g}."
        )
        return {
            "family": normalize_family(family),
            "policy_version": HIRES_AUTO_PROFILE_VERSION,
            "requested_refinement_steps": requested_steps,
            "internal_schedule_steps": int(plan.internal_schedule_steps),
            "effective_refinement_steps": int(plan.effective_refinement_steps),
            "denoising_strength": float(plan.normalized_denoising_strength),
            "safe_denoising_strength": float(plan.safe_denoising_strength),
            "step_policy": step_policy,
            "explanation": explanation,
        }

    @staticmethod
    def _merge_layer(values: dict[str, Any], sources: dict[str, dict[str, Any]], profile, assignment=None) -> None:
        scope = str(getattr(assignment, "scope", "builtin_family") or "builtin_family")
        for key in profile.values:
            values[key] = profile.values[key]
            sources[key] = {
                "source": "profile",
                "profile_id": profile.profile_id,
                "profile_name": profile.name,
                "scope": scope,
                "assignment_key": getattr(assignment, "assignment_key", ""),
            }

    def _preferred_assignment_upscaler(
        self,
        context: HiresResolutionContext,
        available: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str]:
        """Resolve the most specific assignment that identifies a preferred upscaler.

        Combination assignments carry an exact upscaler fingerprint, so Auto can use
        them to choose an asset before upscaler-scoped profile layers are merged.
        Multiple matching assignments remain deterministic: exact-checkpoint scope
        wins over family scope, then native-scale proximity and assignment key break
        ties without relying on filenames.
        """
        by_sha = {
            str(item.get("sha256") or "").strip().casefold(): item
            for item in available
            if str(item.get("sha256") or "").strip()
        }
        family = normalize_family(context.model_family)
        candidates: list[tuple[int, float, str, Any, dict[str, Any]]] = []
        requested_scale = max(1.01, float(context.requested_scale or 2.0))
        for assignment in self.service.list_default_assignments():
            scope = str(assignment.scope or "").strip().casefold()
            priority = 0
            if scope == "checkpoint_upscaler":
                if not context.checkpoint_sha256 or assignment.checkpoint_sha256.casefold() != context.checkpoint_sha256:
                    continue
                priority = 2
            elif scope == "model_family_upscaler":
                if normalize_family(assignment.model_family) != family:
                    continue
                priority = 1
            else:
                continue
            item = by_sha.get(str(assignment.upscaler_sha256 or "").strip().casefold())
            if item is None:
                continue
            native_scale = float(item.get("native_scale") or requested_scale)
            candidates.append(
                (
                    -priority,
                    abs(native_scale - requested_scale),
                    assignment.assignment_key,
                    assignment,
                    item,
                )
            )
        if not candidates:
            return None, ""
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        _neg_priority, _distance, _key, assignment, item = candidates[0]
        return item, f"preferred {assignment.scope} default assignment"

    @staticmethod
    def _choose_upscaler(context: HiresResolutionContext, candidate_id: str, available: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, bool]:
        by_id = {str(item.get("upscaler_id") or ""): item for item in available}
        explicit = str(context.explicit_user_upscaler or "").strip()
        if explicit:
            return by_id.get(explicit), "explicit user selection", True
        requested = str(candidate_id or "").strip()
        if requested and requested != AUTO_SELECT:
            return by_id.get(requested), "resolved profile/default", True
        preferred = str(context.preferred_user_upscaler or "").strip()
        if preferred and preferred in by_id:
            return by_id[preferred], "saved preferred upscaler", False
        neural = [item for item in available if str(item.get("strategy") or "pixel_neural") == "pixel_neural"]
        if neural:
            scale = max(1.01, float(context.requested_scale or 2.0))
            neural.sort(key=lambda item: (abs(float(item.get("native_scale") or 4) - scale), str(item.get("upscaler_id") or "")))
            return neural[0], "best qualified installed upscaler for requested scale", False
        return by_id.get(BUILTIN_PIXEL_RESIZE_ID), "built-in safe resize fallback", False

    @staticmethod
    def _resolve_plugin(value: Any, *, runtime_recommendation: Any, base_value: Any, available: Sequence[str]) -> str:
        selected = str(value or "").strip()
        if selected and selected != AUTO_SELECT:
            return selected
        for candidate in (runtime_recommendation, base_value):
            text = str(candidate or "").strip()
            if text and (not available or text in available):
                return text
        return str(available[0]) if available else ""

    @staticmethod
    def _effective_requested_scale(context: HiresResolutionContext) -> float:
        """Return the requested hires scale, honoring explicit target dimensions."""

        if context.requested_width > 0 and context.requested_height > 0:
            scale_x = float(context.requested_width) / float(max(1, context.width))
            scale_y = float(context.requested_height) / float(max(1, context.height))
            if abs(scale_x - scale_y) <= 1e-6:
                return max(1.0, scale_x)
            # Non-uniform targets still need a conservative scale signal for
            # native-scale mismatch policy. The larger axis determines how much
            # useful native neural enlargement is required.
            return max(1.0, scale_x, scale_y)
        return max(1.0, float(context.requested_scale or 1.0))

    @classmethod
    def _apply_quality_correction_policy(
        cls,
        *,
        context: HiresResolutionContext,
        family: str,
        selected_upscaler: Mapping[str, Any],
        values: dict[str, Any],
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply bounded HA5 native-scale correction policy without overriding profiles."""

        requested_scale = cls._effective_requested_scale(context)
        native_scale = int(selected_upscaler.get("native_scale") or 0)
        field = "hires_final_size_correction_filter"
        existing_source = dict(sources.get(field) or {})
        result = {
            "applied": False,
            "requested_scale": requested_scale,
            "native_scale": native_scale,
            "selected_filter": values.get(field),
            "reason": "",
        }

        # Saved user/default profiles remain authoritative. HA5 built-in quality
        # policy only fills the Auto recommendation layer beneath them.
        if existing_source.get("source") == "profile":
            result["reason"] = "profile override preserved"
            return result

        if normalize_family(family) != "sd3.x":
            result["reason"] = "no SD3.x correction policy"
            return result

        # The known SD3.5 regression is native x4 -> requested x1.5. In that
        # substantial overshoot case, area reduction can soften the learned
        # edge/detail signal before VAE re-encode. Bicubic remains only
        # the exact-size correction step; the .pth model still performs the
        # actual super-resolution inference.
        if native_scale >= 4 and requested_scale < 2.0:
            values[field] = "bicubic"
            reason = (
                f"preserve native x{native_scale} neural detail while correcting "
                f"to requested x{requested_scale:.3g}"
            )
            sources[field] = {
                "source": "builtin_quality_policy",
                "profile_id": "builtin.auto.sd3.x",
                "profile_name": "IMAGE_GEN Auto - SD3.x",
                "scope": "model_family_upscaler",
                "builtin_version": HIRES_AUTO_PROFILE_VERSION,
                "reason": reason,
            }
            result.update(
                applied=True,
                selected_filter="bicubic",
                reason=reason,
            )
            return result

        result["reason"] = "native/requested scale mismatch does not require SD3.x sharp-reduction policy"
        return result

    def resolve(self, context: HiresResolutionContext) -> HiresResolutionResult:
        family = normalize_family(context.model_family)
        builtin_profile_id, builtin_profile_name = self._builtin_identity(family)
        values = dict(self.service.schema.default_values())
        sources = {key: {"source": "generic_safe_fallback", "scope": "generic"} for key in values}
        builtin_values = builtin_auto_profile_values(family)
        values.update(builtin_values)
        for key in builtin_values:
            sources[key] = {
                "source": "builtin_family_recommendation",
                "profile_id": builtin_profile_id,
                "profile_name": builtin_profile_name,
                "scope": "builtin_family",
                "builtin_version": HIRES_AUTO_PROFILE_VERSION,
            }

        preliminary_layers = [pair for pair in self._profile_layers(context) if str(pair[0].scope).casefold() in {"global", "model_family", "checkpoint"}]
        applied_profiles: list[dict[str, Any]] = [{"profile_id": builtin_profile_id, "name": builtin_profile_name, "scope": "builtin_family", "builtin_version": HIRES_AUTO_PROFILE_VERSION}]
        applied_assignments: list[dict[str, Any]] = []
        for assignment, profile in preliminary_layers:
            self._merge_layer(values, sources, profile, assignment)
            applied_profiles.append({"profile_id": profile.profile_id, "name": profile.name, "scope": assignment.scope})
            applied_assignments.append(assignment.to_dict())

        available = self._available_upscalers(context)
        preferred_assignment_upscaler = None
        preferred_assignment_reason = ""
        if not str(context.explicit_user_upscaler or "").strip():
            preferred_assignment_upscaler, preferred_assignment_reason = self._preferred_assignment_upscaler(
                context, available
            )
        if preferred_assignment_upscaler is not None:
            selected_upscaler = preferred_assignment_upscaler
            upscaler_reason = preferred_assignment_reason
            locked = False
        else:
            selected_upscaler, upscaler_reason, locked = self._choose_upscaler(
                context, values.get("hires_upscaler_id"), available
            )
        unresolved: list[str] = []
        warnings: list[str] = []
        if selected_upscaler is None:
            requested = context.explicit_user_upscaler or values.get("hires_upscaler_id")
            unresolved.append(f"Required upscaler asset {requested!r} is unavailable.")
            selected_upscaler = builtin_resize_descriptor() if not locked else {}
        upscaler_sha = str(selected_upscaler.get("sha256") or "").casefold()

        # Rebuild in exact documented precedence order now that the upscaler identity is known.
        values = dict(self.service.schema.default_values())
        sources = {key: {"source": "generic_safe_fallback", "scope": "generic"} for key in values}
        values.update(builtin_values)
        for key in builtin_values:
            sources[key] = {
                "source": "builtin_family_recommendation",
                "profile_id": builtin_profile_id,
                "profile_name": builtin_profile_name,
                "scope": "builtin_family",
                "builtin_version": HIRES_AUTO_PROFILE_VERSION,
            }
        all_layers = self._profile_layers(context, upscaler_sha=upscaler_sha)
        applied_profiles = [{"profile_id": builtin_profile_id, "name": builtin_profile_name, "scope": "builtin_family", "builtin_version": HIRES_AUTO_PROFILE_VERSION}]
        applied_assignments = []
        for assignment, profile in all_layers:
            self._merge_layer(values, sources, profile, assignment)
            applied_profiles.append({"profile_id": profile.profile_id, "name": profile.name, "scope": assignment.scope})
            applied_assignments.append(assignment.to_dict())

        requested_after_layers = str(values.get("hires_upscaler_id") or AUTO_SELECT).strip()
        if preferred_assignment_upscaler is not None and not str(context.explicit_user_upscaler or "").strip():
            requested_after_layers = str(
                preferred_assignment_upscaler.get("upscaler_id")
                or preferred_assignment_upscaler.get("value")
                or ""
            ).strip()
        selected_after_layers, reason_after_layers, locked_after_layers = self._choose_upscaler(
            context, requested_after_layers, available
        )
        if preferred_assignment_upscaler is not None and selected_after_layers is not None:
            reason_after_layers = preferred_assignment_reason
            locked_after_layers = False
        if selected_after_layers is None:
            if locked_after_layers:
                unresolved.append(f"Required upscaler asset {requested_after_layers!r} is unavailable.")
            else:
                selected_after_layers = builtin_resize_descriptor()
                warnings.append("Auto selected the built-in Bicubic resize fallback because no qualified neural upscaler is available.")
        else:
            selected_upscaler = selected_after_layers
            upscaler_reason = reason_after_layers
        selected_upscaler = selected_after_layers or selected_upscaler or builtin_resize_descriptor()
        selected_id = str(selected_upscaler.get("upscaler_id") or selected_upscaler.get("value") or "")
        values["hires_upscaler_id"] = selected_id
        sources["hires_upscaler_id"] = {"source": "auto_upscaler_resolver", "scope": "resolver", "reason": upscaler_reason}
        strategy = str(selected_upscaler.get("strategy") or "pixel_neural")
        values["hires_strategy"] = strategy
        sources["hires_strategy"] = {"source": "auto_upscaler_resolver", "scope": "resolver", "reason": f"selected {selected_id}"}

        values["hires_sampler_name"] = self._resolve_plugin(
            values.get("hires_sampler_name"),
            runtime_recommendation=context.runtime_profile.get("sampler_name"),
            base_value=context.base_values.get("sampler_name"),
            available=context.available_samplers,
        )
        values["hires_scheduler_name"] = self._resolve_plugin(
            values.get("hires_scheduler_name"),
            runtime_recommendation=context.runtime_profile.get("scheduler_name"),
            base_value=context.base_values.get("scheduler_name"),
            available=context.available_schedulers,
        )
        for key in ("hires_sampler_name", "hires_scheduler_name"):
            sources[key] = {"source": "model_runtime_recommendation", "scope": "resolver"}

        if values.get("hires_cfg_scale") is None:
            values["hires_cfg_scale"] = float(context.base_values.get("cfg_scale") or context.runtime_profile.get("image_gen_cfg_scale") or 7.0)
            sources["hires_cfg_scale"] = {"source": "base_generation_settings", "scope": "resolver"}
        if values.get("hires_cfg_rescale") is None:
            values["hires_cfg_rescale"] = float(context.base_values.get("cfg_rescale") or 0.0)
            sources["hires_cfg_rescale"] = {"source": "base_generation_settings", "scope": "resolver"}
        values["hires_prompt_parser_name"] = str(values.get("hires_prompt_parser_name") or context.base_values.get("prompt_parser_name") or "legacy")
        values["hires_shortcut_profile_name"] = str(values.get("hires_shortcut_profile_name") or context.base_values.get("prompt_shortcut_profile_name") or "legacy_default")

        if context.requested_width > 0 and context.requested_height > 0:
            values["hires_size_mode"] = "explicit_dimensions"
            values["hires_width"] = context.requested_width
            values["hires_height"] = context.requested_height
            sources["hires_size_mode"] = sources["hires_width"] = sources["hires_height"] = {"source": "current_requested_target", "scope": "context"}
        else:
            values["hires_size_mode"] = "scale_from_base"
            values["hires_scale"] = max(1.01, float(context.requested_scale or values.get("hires_scale") or 2.0))
            sources["hires_size_mode"] = sources["hires_scale"] = {"source": "current_requested_target", "scope": "context"}

        correction_policy = self._apply_quality_correction_policy(
            context=context,
            family=family,
            selected_upscaler=selected_upscaler,
            values=values,
            sources=sources,
        )

        values["hires_enabled"] = True
        required = {
            "hires_upscaler_id": selected_id,
            "hires_strategy": values.get("hires_strategy"),
            "hires_sampler_name": values.get("hires_sampler_name"),
            "hires_scheduler_name": values.get("hires_scheduler_name"),
            "hires_steps": values.get("hires_steps"),
            "hires_denoising_strength": values.get("hires_denoising_strength"),
        }
        for key, value in required.items():
            if value in (None, ""):
                unresolved.append(f"Auto could not resolve required setting {key}.")
        if int(values.get("hires_steps") or 0) < 1:
            unresolved.append("Auto resolved an invalid hires step count.")
        strength = float(values.get("hires_denoising_strength") or 0.0)
        if not (0.0 < strength <= 1.0):
            unresolved.append("Auto resolved an invalid hires denoising strength.")

        if selected_id == BUILTIN_PIXEL_RESIZE_ID and not any("built-in Bicubic" in item for item in warnings):
            warnings.append("Auto selected the built-in Bicubic resize fallback because no qualified neural upscaler is available.")

        refinement_policy = self._refinement_policy_summary(values, family=family)

        diagnostics = {
            "family": family,
            "builtin_profile_id": builtin_profile_id,
            "builtin_profile_name": builtin_profile_name,
            "builtin_profile_version": HIRES_AUTO_PROFILE_VERSION,
            "checkpoint_sha256": context.checkpoint_sha256,
            "requested_scale": context.requested_scale,
            "requested_target": {"width": context.requested_width, "height": context.requested_height},
            "selected_upscaler_reason": upscaler_reason,
            "selected_upscaler_sha256": str(selected_upscaler.get("sha256") or ""),
            "selected_upscaler_native_scale": int(selected_upscaler.get("native_scale") or 0),
            "refinement_policy": refinement_policy,
            "quality_correction_policy": correction_policy,
            "vae_contract": context.vae_contract,
        }
        fingerprint_payload = {
            "values": values,
            "field_sources": sources,
            "selected_upscaler": selected_upscaler,
            "applied_assignments": applied_assignments,
        }
        fingerprint = hashlib.sha256(json.dumps(_json_safe(fingerprint_payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return HiresResolutionResult(
            values=values,
            field_sources=sources,
            applied_profiles=tuple(applied_profiles),
            applied_assignments=tuple(applied_assignments),
            selected_upscaler=dict(selected_upscaler),
            valid=not unresolved,
            warnings=tuple(dict.fromkeys(warnings)),
            unresolved_requirements=tuple(dict.fromkeys(unresolved)),
            resolution_fingerprint=fingerprint,
            diagnostics=diagnostics,
        )
