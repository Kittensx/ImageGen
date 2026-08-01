from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import gc
import time
import traceback
from typing import Any, Callable, Iterable

import torch

from .contracts import MemoryEstimate, MemoryManagerSettings, ResidencyPlan
from .diagnostics import memory_failure_bundle
from .oom_recovery import OOMRecoveryState, StageRecoveryContract, is_cuda_oom
from .planner import MemoryEstimator, MemoryPlanner
from .policy import post_stage_offload_candidates, resolve_policy
from .residency import ComponentResidencyRegistry
from .telemetry import MemoryTelemetry


_CREATIVE_KEYS = (
    "width",
    "height",
    "steps",
    "cfg_scale",
    "seed",
    "batch_size",
    "sampler_name",
    "scheduler_name",
    "denoising_strength",
)


class ComponentLease(AbstractContextManager["ComponentLease"]):
    def __init__(
        self,
        manager: "AdaptiveComponentMemoryManager",
        *,
        stage: str,
        required: Iterable[str],
        preferred: Iterable[str] = (),
        optional: Iterable[str] = (),
        estimated_stage_bytes: int = 0,
        preview_requires_vae: bool = False,
        requested_profile_override: str | None = None,
        safety_margin_bytes_override: int | None = None,
    ) -> None:
        self.manager = manager
        self.stage = str(stage)
        self.required = tuple(required)
        self.preferred = tuple(preferred)
        self.optional = tuple(optional)
        self.estimated_stage_bytes = max(0, int(estimated_stage_bytes))
        self.preview_requires_vae = bool(preview_requires_vae)
        self.requested_profile_override = (
            None
            if requested_profile_override is None
            else str(requested_profile_override)
        )
        self.safety_margin_bytes_override = (
            None
            if safety_margin_bytes_override is None
            else max(0, int(safety_margin_bytes_override))
        )
        self.plan: ResidencyPlan | None = None
        self._leased_ids: tuple[str, ...] = ()

    def __enter__(self) -> "ComponentLease":
        self.plan = self.manager._enter_lease(
            stage=self.stage,
            required=self.required,
            preferred=self.preferred,
            optional=self.optional,
            estimated_stage_bytes=self.estimated_stage_bytes,
            preview_requires_vae=self.preview_requires_vae,
            requested_profile_override=self.requested_profile_override,
            safety_margin_bytes_override=self.safety_margin_bytes_override,
        )
        self._leased_ids = tuple(self.plan.selected_for_target)
        self.manager.registry.acquire(self._leased_ids)
        self.manager._emit_status("lease_acquired", stage=self.stage)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.manager.registry.release(self._leased_ids)
        self.manager._exit_lease(self.stage, self.plan)
        self.manager._emit_status("lease_released", stage=self.stage)
        return False


class AdaptiveComponentMemoryManager:
    """Whole-component residency manager for the current txt2img pipeline."""

    def __init__(
        self,
        *,
        target_device: str | torch.device,
        settings: MemoryManagerSettings | None = None,
        state: Any | None = None,
        event_callback: Callable[[dict[str, Any]], Any] | None = None,
        telemetry: MemoryTelemetry | None = None,
    ) -> None:
        self.target_device = torch.device(target_device)
        self.settings = settings or MemoryManagerSettings()
        self.state = state
        self.event_callback = event_callback
        self.registry = ComponentResidencyRegistry()
        self.telemetry = telemetry or MemoryTelemetry(device=self.target_device)
        self.estimator = MemoryEstimator()
        self.planner = MemoryPlanner()
        self.oom = OOMRecoveryState(
            configured_profile=self.settings.oom_retry_profile,
            retry_limit=self.settings.oom_retry_limit,
        )
        self.snapshots: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] = []
        self.estimates: list[dict[str, Any]] = []
        self.automatic_actions: list[dict[str, Any]] = []
        self.active_stage: str | None = None
        self.latest_estimate: dict[str, Any] | None = None
        self.latest_snapshot: dict[str, Any] | None = None
        self.peak_vram_by_stage: dict[str, dict[str, int | None]] = {}
        self.failure_bundle: dict[str, Any] | None = None
        self.preview_image_decode_suspended = False
        self.preview_image_decode_suspension_reason = ""
        self.preview_image_decode_suspension_source = ""
        self.preview_decoder_released = False
        # Phase 14K preview suspension is intentionally one-way for each job.
        # A new manager is created for the next job; this manager never reloads
        # or reattaches a preview decoder after pressure/policy suspension.
        self.preview_suspension_one_way_for_job = True
        self._request_context: Any = None
        self._dimension_plan: Any = None
        self._preview_mode = "disabled"
        self._effective_profile = self.settings.policy
        self.hires_cleanup_reports: list[dict[str, Any]] = []
        self._oom_attention_context_factory: Callable[[str], Any] | None = None
        self._oom_vae_memory_configurator: Callable[..., Any] | None = None

    @classmethod
    def from_state(
        cls,
        *,
        target_device: str | torch.device,
        state: Any | None,
    ) -> "AdaptiveComponentMemoryManager":
        extra = getattr(state, "extra", None)
        values = dict(extra or {}) if isinstance(extra, dict) else {}
        callback = values.get("memory_event_callback")
        return cls(
            target_device=target_device,
            settings=MemoryManagerSettings.from_mapping(values),
            state=state,
            event_callback=callback if callable(callback) else None,
        )

    def register_core_components(self, components: Any) -> None:
        model_identity = str(
            getattr(components, "model_identity", "")
            or getattr(components, "model_hash", "")
            or "active_checkpoint"
        )
        self.registry.invalidate_incompatible(model_identity)
        self.registry.register(
            component_id="text_encoder",
            component_kind="text_encoder",
            model_identity=model_identity,
            module=components.text_encoder,
            preferred_dtype=self._module_dtype(components.text_encoder),
            required_by_stages={"conditioning"},
            pinned_cpu_capable=self.settings.pinned_cpu_memory,
        )
        self.registry.register(
            component_id="unet",
            component_kind="unet",
            model_identity=model_identity,
            module=components.unet,
            preferred_dtype=self._module_dtype(components.unet),
            required_by_stages={"sampling", "hires_second_pass"},
            pinned_cpu_capable=self.settings.pinned_cpu_memory,
        )
        self.registry.register(
            component_id="vae",
            component_kind="vae",
            model_identity=model_identity,
            module=components.vae,
            preferred_dtype=self._module_dtype(components.vae),
            required_by_stages={"final_decode", "source_image_encode", "hires_vae_encode"},
            pinned_cpu_capable=self.settings.pinned_cpu_memory,
        )
        self.capture("components_registered")
        self.apply_initial_policy()

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> str:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return str(parameter.dtype)
        return "torch.float32"

    def set_request_context(
        self,
        *,
        request: Any,
        dimension_plan: Any,
        preview_mode: str,
    ) -> None:
        self._request_context = request
        self._dimension_plan = dimension_plan
        self._preview_mode = str(preview_mode or "disabled").lower()

    def configure_oom_recovery_hooks(
        self,
        *,
        attention_context_factory: Callable[[str], Any] | None = None,
        vae_memory_configurator: Callable[..., Any] | None = None,
    ) -> None:
        """Register job-local memory-control hooks used by maximum recovery.

        The hooks are deliberately generic so the memory manager does not own
        UNet attention or VAE implementation details.
        """

        self._oom_attention_context_factory = (
            attention_context_factory
            if callable(attention_context_factory)
            else None
        )
        self._oom_vae_memory_configurator = (
            vae_memory_configurator
            if callable(vae_memory_configurator)
            else None
        )

    def component_bytes(self) -> dict[str, int]:
        return {
            component_id: component.estimated_total_bytes
            for component_id, component in self.registry.components.items()
        }

    def effective_profile_for_stage(
        self,
        stage: str,
        *,
        fallback: str | None = None,
    ) -> str:
        """Return the most recent effective planner profile for a stage."""

        target = str(stage)
        for plan in reversed(self.plans):
            if str(plan.get("stage") or "") != target:
                continue
            value = str(plan.get("effective_profile") or "").strip()
            if value:
                return value
        return str(fallback or self.settings.policy)

    def capture(self, stage: str) -> dict[str, Any]:
        snapshot = self.telemetry.capture(
            stage,
            component_residency=self.registry.snapshot(),
        ).to_dict()
        self.snapshots.append(snapshot)
        self.latest_snapshot = snapshot
        self._record_stage_peak(self.active_stage, snapshot)
        self._emit_status("memory_snapshot", stage=stage)
        return snapshot

    def _record_stage_peak(
        self,
        logical_stage: str | None,
        snapshot: dict[str, Any],
    ) -> None:
        stage = str(logical_stage or "").strip()
        if not stage:
            return
        cuda = dict(snapshot.get("cuda") or {})
        if not cuda.get("available"):
            return
        current = self.peak_vram_by_stage.setdefault(
            stage,
            {
                "peak_allocated_vram_bytes": None,
                "peak_reserved_vram_bytes": None,
            },
        )
        for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
            value = cuda.get(key)
            if value is None:
                continue
            previous = current.get(key)
            current[key] = max(int(value), int(previous or 0))

    def apply_initial_policy(self) -> None:
        snapshot = self.capture("initial_residency")
        decision = resolve_policy(self.settings.policy, cuda_payload=snapshot.get("cuda"))
        self._effective_profile = decision.effective_profile
        if self.target_device.type != "cuda":
            return
        if self._effective_profile == "high_vram" and self.settings.vae_device != "cpu":
            return
        candidates = {"vae"}
        if self._effective_profile == "low_vram":
            candidates.add("unet")
        for component_id in candidates:
            component = self.registry.components.get(component_id)
            if component is None or component.leased or not component.current_device.startswith("cuda"):
                continue
            self._move_to_cpu(component_id, stage="initial_residency", reason="initial policy residency")
        self.capture("initial_policy_applied")

    def estimate_stage(
        self,
        stage: str,
        *,
        safety_margin_bytes_override: int | None = None,
    ) -> MemoryEstimate | None:
        if self._request_context is None or self._dimension_plan is None:
            return None
        cuda = dict((self.latest_snapshot or {}).get("cuda") or self.telemetry.cuda_payload())
        available = cuda.get("free_vram_bytes")
        estimate = self.estimator.estimate_txt2img(
            request=self._request_context,
            dimension_plan=self._dimension_plan,
            sampler_name=getattr(self._request_context, "sampler_name", ""),
            dtype=getattr(self._request_context, "dtype", None),
            preview_mode=self._preview_mode,
            component_bytes=self.component_bytes(),
            available_bytes=int(available) if available is not None else None,
            safety_margin_bytes=(
                max(0, int(safety_margin_bytes_override))
                if safety_margin_bytes_override is not None
                else self.settings.safety_margin_bytes
            ),
            stage=stage,
        )
        payload = estimate.to_dict()
        self.estimates.append(payload)
        self.latest_estimate = payload
        return estimate

    def lease(
        self,
        *,
        stage: str,
        required: Iterable[str],
        preferred: Iterable[str] = (),
        optional: Iterable[str] = (),
        estimated_stage_bytes: int = 0,
        preview_requires_vae: bool = False,
        requested_profile_override: str | None = None,
        safety_margin_bytes_override: int | None = None,
    ) -> ComponentLease:
        return ComponentLease(
            self,
            stage=stage,
            required=required,
            preferred=preferred,
            optional=optional,
            estimated_stage_bytes=estimated_stage_bytes,
            preview_requires_vae=preview_requires_vae,
            requested_profile_override=requested_profile_override,
            safety_margin_bytes_override=safety_margin_bytes_override,
        )

    def _enter_lease(
        self,
        *,
        stage: str,
        required: tuple[str, ...],
        preferred: tuple[str, ...],
        optional: tuple[str, ...],
        estimated_stage_bytes: int,
        preview_requires_vae: bool,
        requested_profile_override: str | None,
        safety_margin_bytes_override: int | None,
    ) -> ResidencyPlan:
        self.active_stage = stage
        self.telemetry.reset_peak()
        snapshot = self.capture(f"before_{stage}")
        estimate = self.estimate_stage(
            stage,
            safety_margin_bytes_override=safety_margin_bytes_override,
        )
        cuda = dict(snapshot.get("cuda") or {})
        available = cuda.get("free_vram_bytes")
        requested_profile = (
            str(requested_profile_override)
            if requested_profile_override is not None
            else self.settings.policy
        )
        safety_margin_bytes = (
            max(0, int(safety_margin_bytes_override))
            if safety_margin_bytes_override is not None
            else self.settings.safety_margin_bytes
        )
        plan = self.planner.plan(
            stage=stage,
            requested_profile=requested_profile,
            target_device=str(self.target_device),
            required=required,
            preferred=preferred,
            optional=optional,
            component_bytes=self.component_bytes(),
            available_bytes=int(available) if available is not None else None,
            safety_margin_bytes=safety_margin_bytes,
            estimated_stage_bytes=(
                int(estimate.estimated_expected_bytes)
                if estimate is not None
                else max(0, int(estimated_stage_bytes))
            ),
            preview_requires_vae=preview_requires_vae,
        )
        self._effective_profile = plan.effective_profile
        self.plans.append(plan.to_dict())

        selected = set(plan.selected_for_target)
        for component_id in plan.selected_for_cpu:
            component = self.registry.components.get(component_id)
            if component is None or component.leased:
                continue
            if component.current_device.startswith("cuda"):
                self._move_to_cpu(component_id, stage=stage, reason="stage plan eviction")

        default_target = "cpu" if plan.effective_profile == "cpu_fallback" else str(self.target_device)
        for component_id in selected:
            target = default_target
            if component_id == "vae":
                requested_vae_device = str(self.settings.vae_device or "auto").lower()
                if requested_vae_device == "cpu":
                    target = "cpu"
                elif requested_vae_device == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "VAE device cuda was requested, but CUDA is unavailable."
                        )
                    target = (
                        str(self.target_device)
                        if self.target_device.type == "cuda"
                        else "cuda"
                    )
            component = self.registry.get(component_id)
            if not self._device_matches(component.current_device, target):
                self.registry.move(
                    component_id,
                    device=target,
                    stage=stage,
                    reason=(
                        f"VAE device override {self.settings.vae_device}"
                        if component_id == "vae" and self.settings.vae_device != "auto"
                        else "required by stage lease" if component_id in required else "retained by stage plan"
                    ),
                )
                self.automatic_actions.append({
                    "stage": stage,
                    "action": "component_transfer",
                    "component_id": component_id,
                    "target_device": target,
                    "vae_device_override": (
                        self.settings.vae_device if component_id == "vae" else None
                    ),
                })

        if plan.preview_image_decode_suspended:
            self.suspend_preview_image_decode(
                "The active stage could not retain the VAE preview decoder without violating the VRAM safety margin.",
                source="memory_pressure",
            )
        self.capture(f"lease_ready_{stage}")
        return plan

    @staticmethod
    def _device_matches(current: str, target: str) -> bool:
        if current == target:
            return True
        return current.startswith("cuda") and target == "cuda"

    def _move_to_cpu(self, component_id: str, *, stage: str, reason: str) -> None:
        self.registry.move(component_id, device="cpu", stage=stage, reason=reason)
        self.automatic_actions.append({
            "stage": stage,
            "action": "component_offload",
            "component_id": component_id,
            "reason": reason,
        })

    def _exit_lease(self, stage: str, plan: ResidencyPlan | None) -> None:
        self.capture(f"after_{stage}")
        effective_profile = (
            plan.effective_profile if plan is not None else self._effective_profile
        )
        candidates = post_stage_offload_candidates(effective_profile, stage)
        if stage == "final_decode":
            if self.settings.retain_vae_between_jobs:
                candidates.discard("vae")
            else:
                candidates.add("vae")
            if not self.settings.retain_checkpoint_between_jobs:
                candidates.add("unet")
        for component_id in candidates:
            component = self.registry.components.get(component_id)
            if component is None or component.leased:
                continue
            if component.current_device.startswith("cuda"):
                self._move_to_cpu(component_id, stage=stage, reason="post-stage policy")
        self.capture(f"post_policy_{stage}")
        self.active_stage = None
        self._effective_profile = self.settings.policy

    def offload_inactive_components(
        self,
        component_ids: Iterable[str],
        *,
        stage: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for component_id in dict.fromkeys(str(value) for value in component_ids):
            component = self.registry.components.get(component_id)
            if component is None:
                actions.append({
                    "action": "component_not_registered",
                    "component_id": component_id,
                })
                continue
            if component.leased:
                actions.append({
                    "action": "component_offload_skipped",
                    "component_id": component_id,
                    "reason": "active lease",
                })
                continue
            if not component.current_device.startswith("cuda"):
                actions.append({
                    "action": "component_already_offloaded",
                    "component_id": component_id,
                    "device": component.current_device,
                })
                continue
            self._move_to_cpu(component_id, stage=stage, reason=reason)
            actions.append({
                "action": "component_offload",
                "component_id": component_id,
                "target_device": "cpu",
                "reason": reason,
            })
        return actions

    def release_preview_work_for_hires(self) -> list[dict[str, Any]]:
        extra = getattr(self.state, "extra", None)
        if not isinstance(extra, dict):
            return []
        writer = extra.get("live_preview_frame_writer")
        if writer is None:
            return []
        actions: list[dict[str, Any]] = []
        drain = getattr(writer, "drain", None)
        if callable(drain):
            try:
                drain()
                actions.append({"action": "drain_preview_queue", "applied": True})
            except Exception as exc:
                actions.append({
                    "action": "drain_preview_queue",
                    "applied": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        release_history = getattr(writer, "release_nonfinal_history", None)
        if callable(release_history):
            try:
                removed = int(release_history() or 0)
                actions.append({
                    "action": "release_nonfinal_preview_history",
                    "removed_files": removed,
                })
            except Exception as exc:
                actions.append({
                    "action": "release_nonfinal_preview_history",
                    "applied": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        extra["live_preview_hires_queue_released"] = True
        return actions

    def record_hires_cleanup(self, report: dict[str, Any]) -> None:
        payload = dict(report or {})
        self.hires_cleanup_reports.append(payload)
        self.automatic_actions.append({
            "stage": "pre_hires_cleanup",
            "action": "hires_cleanup_complete",
            "profile": payload.get("profile"),
            "reclaimed_allocated_bytes": payload.get("reclaimed_allocated_bytes", 0),
            "reclaimed_reserved_bytes": payload.get("reclaimed_reserved_bytes", 0),
        })
        self._emit_status("hires_cleanup_complete", stage="pre_hires_cleanup")

    def suspend_preview_image_decode(
        self,
        reason: str,
        *,
        source: str = "automatic",
        release_decoder: bool = True,
    ) -> bool:
        if self.preview_image_decode_suspended:
            return True
        extra = getattr(self.state, "extra", None)
        if not isinstance(extra, dict):
            return False
        writer = extra.get("live_preview_frame_writer")
        suspended = False
        method = getattr(writer, "suspend_image_decode", None)
        if callable(method):
            try:
                method(
                    reason=reason,
                    source=source,
                    release_decoder=release_decoder,
                )
                suspended = True
            except TypeError:
                try:
                    method(reason=reason)
                    suspended = True
                except Exception:
                    suspended = False
            except Exception:
                suspended = False
        if suspended:
            self.preview_image_decode_suspended = True
            self.preview_image_decode_suspension_reason = str(reason)
            self.preview_image_decode_suspension_source = str(source or "automatic")
            self.preview_decoder_released = bool(release_decoder)
            extra["live_preview_image_decode_suspended"] = True
            extra["live_preview_image_decode_suspension_reason"] = str(reason)
            extra["live_preview_image_decode_suspension_source"] = self.preview_image_decode_suspension_source
            extra["live_preview_decoder_released"] = self.preview_decoder_released
            action = {
                "stage": self.active_stage,
                "action": "preview_image_decode_suspended",
                "reason": str(reason),
                "source": self.preview_image_decode_suspension_source,
                "preview_decoder_released": self.preview_decoder_released,
                "cfg_telemetry_continues": True,
            }
            self.automatic_actions.append(action)
            self.oom.record_action(
                self.active_stage or "unknown",
                action["action"],
                reason=reason,
                source=self.preview_image_decode_suspension_source,
            )
            self._emit_status("preview_suspended", stage=self.active_stage)
        return suspended

    def _release_preview_history_for_recovery(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        extra = getattr(self.state, "extra", None)
        writer = extra.get("live_preview_frame_writer") if isinstance(extra, dict) else None
        drain = getattr(writer, "drain", None)
        if callable(drain):
            try:
                drain()
                actions.append({"action": "drain_preview_queue", "applied": True})
            except Exception as exc:
                actions.append({
                    "action": "drain_preview_queue_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        release_history = getattr(writer, "release_nonfinal_history", None)
        if callable(release_history):
            try:
                removed = int(release_history() or 0)
                actions.append({
                    "action": "release_nonfinal_preview_history",
                    "removed_files": removed,
                })
            except Exception as exc:
                actions.append({
                    "action": "release_nonfinal_preview_history_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return actions

    def _recover_from_oom(
        self,
        *,
        stage: str,
        profile: str,
        required: set[str],
        boundary: StageRecoveryContract | None,
        retry_index: int,
    ) -> list[dict[str, Any]]:
        """Apply one bounded recovery profile after the failed lease is released."""

        selected = str(profile)
        actions: list[dict[str, Any]] = []
        before = self.capture(f"oom_before_recovery_{stage}")

        if boundary is not None:
            try:
                details = boundary.prepare(selected, retry_index)
                actions.append({
                    "action": "restore_valid_stage_boundary",
                    "boundary_id": boundary.boundary_id,
                    "restart_mode": boundary.restart_mode,
                    "details": details,
                })
            except Exception as exc:
                actions.append({
                    "action": "restore_valid_stage_boundary_failed",
                    "boundary_id": boundary.boundary_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                for action in actions:
                    self.oom.record_action(
                        stage,
                        action["action"],
                        **{key: value for key, value in action.items() if key != "action"},
                    )
                raise RuntimeError(
                    f"OOM retry boundary {boundary.boundary_id!r} could not be restored."
                ) from exc

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                actions.append({"action": "cuda_synchronize", "applied": True})
            except Exception as exc:
                actions.append({
                    "action": "cuda_synchronize_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        collected = int(gc.collect())
        actions.append({"action": "garbage_collect", "collected": collected})

        if selected in {"low_vram", "maximum"}:
            if self.settings.allow_preview_suspension_on_oom:
                if self.suspend_preview_image_decode(
                    "CUDA OOM recovery suspended image preview decoding for the remainder of this job.",
                    source="oom_recovery",
                ):
                    actions.append({
                        "action": "suspend_preview_image_decode",
                        "one_way_for_job": True,
                    })
            actions.extend(self._release_preview_history_for_recovery())

            for component_id, component in self.registry.components.items():
                if component_id in required or component.leased:
                    continue
                if component.current_device.startswith("cuda"):
                    self._move_to_cpu(
                        component_id,
                        stage=stage,
                        reason=f"OOM {selected} recovery",
                    )
                    actions.append({
                        "action": "offload_inactive_component",
                        "component_id": component_id,
                        "fallback_profile": selected,
                    })

        if selected == "maximum":
            configurator = self._oom_vae_memory_configurator
            if callable(configurator):
                try:
                    report = configurator(tiling=True, slicing=True)
                    actions.append({
                        "action": "enable_vae_tiling_and_slicing",
                        "report": dict(report or {}),
                    })
                except Exception as exc:
                    actions.append({
                        "action": "enable_vae_tiling_and_slicing_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            else:
                actions.append({
                    "action": "enable_vae_tiling_and_slicing_unavailable",
                })

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                actions.append({"action": "empty_released_allocator_cache"})
            except Exception as exc:
                actions.append({
                    "action": "allocator_cleanup_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        after = self.capture(f"oom_after_recovery_{stage}")
        actions.append({
            "action": "recovery_memory_snapshot",
            "profile": selected,
            "before": before,
            "after": after,
        })
        for action in actions:
            self.oom.record_action(
                stage,
                action["action"],
                **{key: value for key, value in action.items() if key != "action"},
            )
        self._emit_status("oom_recovery_applied", stage=stage)
        return actions

    def _oom_retry_context(self, *, stage: str, profile: str):
        if profile != "maximum" or stage not in {"sampling", "hires_second_pass"}:
            return nullcontext({
                "requested": "off",
                "applied": False,
                "reason": "maximum attention recovery not required for this stage",
            })
        factory = self._oom_attention_context_factory
        if not callable(factory):
            return nullcontext({
                "requested": "max",
                "applied": False,
                "reason": "attention slicing hook unavailable",
            })
        return factory("max")

    @staticmethod
    def _creative_snapshot(request: Any) -> dict[str, Any]:
        return {key: getattr(request, key, None) for key in _CREATIVE_KEYS}

    def observe_stage(self, stage: str, operation: Callable[[], Any]) -> Any:
        self.active_stage = str(stage)
        self.telemetry.reset_peak()
        self.capture(f"before_{stage}")
        try:
            return operation()
        finally:
            self.capture(f"after_{stage}")
            self.active_stage = None

    def run_stage(
        self,
        *,
        stage: str,
        operation: Callable[[], Any],
        required: Iterable[str],
        preferred: Iterable[str] = (),
        optional: Iterable[str] = (),
        estimated_stage_bytes: int = 0,
        preview_requires_vae: bool = False,
        request: Any = None,
        requested_profile_override: str | None = None,
        safety_margin_bytes_override: int | None = None,
        recovery_contract: StageRecoveryContract | None = None,
    ) -> Any:
        target_stage = str(stage)
        required_set = set(required)
        preferred_set = set(preferred)
        optional_set = set(optional)
        creative_before = self._creative_snapshot(request) if request is not None else {}

        contract = recovery_contract
        if contract is None and not self.oom.sampler_stage(target_stage):
            # Non-sampler stages in this pipeline are invoked from stable input
            # tensors or immutable request data, so rerunning the operation is a
            # valid same-stage boundary. Sampling must opt in explicitly.
            contract = StageRecoveryContract(
                boundary_id=f"{target_stage}:stable_stage_inputs",
                operation_factory=lambda _attempt: operation,
                metadata={"implicit_idempotent_stage_boundary": True},
            )

        def operation_for_attempt(attempt_index: int) -> Callable[[], Any]:
            if contract is not None:
                return contract.operation_for_attempt(attempt_index)
            return operation

        original_effective_profile = str(
            requested_profile_override or self.settings.policy
        )
        try:
            with self.lease(
                stage=target_stage,
                required=required_set,
                preferred=preferred_set,
                optional=optional_set,
                estimated_stage_bytes=estimated_stage_bytes,
                preview_requires_vae=preview_requires_vae,
                requested_profile_override=requested_profile_override,
                safety_margin_bytes_override=safety_margin_bytes_override,
            ) as initial_lease:
                if initial_lease.plan is not None:
                    original_effective_profile = str(
                        initial_lease.plan.effective_profile
                    )
                result = operation_for_attempt(0)()
            if contract is not None:
                contract.release()
            return result
        except Exception as exc:
            if not is_cuda_oom(exc):
                if contract is not None:
                    contract.release()
                raise

            has_boundary = contract is not None
            if not self.oom.may_retry(
                target_stage,
                has_valid_boundary=has_boundary,
            ):
                reason = "retry_disabled_or_limit_reached"
                if self.oom.sampler_stage(target_stage) and not has_boundary:
                    reason = "sampler_stage_has_no_valid_saved_boundary"
                self.oom.record_action(
                    target_stage,
                    "oom_retry_not_permitted",
                    reason=reason,
                    configured_profile=self.oom.configured_profile,
                    retry_limit=self.oom.retry_limit,
                    total_retry_count=self.oom.total_retry_count,
                )
                self.oom.final_result_by_stage[target_stage] = "failed_without_retry"
                self.failure_bundle = memory_failure_bundle(
                    request=request,
                    stage=target_stage,
                    error=exc,
                    manager_summary=self.summary(),
                    dimension_plan=self._dimension_plan,
                )
                if contract is not None:
                    contract.release()
                raise

            fallback_profile = self.oom.configured_profile
            attempt = self.oom.record_retry(
                target_stage,
                original_profile=original_effective_profile,
                fallback_profile=fallback_profile,
                boundary=contract,
                error=exc,
            )
            failed_traceback = exc.__traceback__
            cleared_traceback_frames = 0
            if failed_traceback is not None:
                frames = []
                cursor = failed_traceback
                while cursor is not None:
                    frames.append(cursor.tb_frame)
                    cursor = cursor.tb_next
                cleared_traceback_frames = len(frames)
                traceback.clear_frames(failed_traceback)
                exc.__traceback__ = None
            self.oom.record_action(
                target_stage,
                "release_failed_stage_temporaries",
                traceback_frames_cleared=cleared_traceback_frames,
                failed_lease_released=True,
            )
            self._emit_status("oom_retry_scheduled", stage=target_stage)

            creative_after_failure = (
                self._creative_snapshot(request) if request is not None else {}
            )
            if creative_after_failure != creative_before:
                self.oom.record_result(
                    target_stage,
                    "blocked_creative_settings_changed",
                    error=exc,
                )
                if contract is not None:
                    contract.release()
                raise RuntimeError(
                    "OOM recovery changed creative request settings, which is not allowed."
                ) from exc

            try:
                recovery_required = set(required_set)
                if (
                    fallback_profile in {"low_vram", "maximum"}
                    and self.oom.sampler_stage(target_stage)
                ):
                    # Sampling-stage VAE residency exists only for image preview.
                    # The fallback suspends preview one-way, so retry requires UNet only.
                    recovery_required.discard("vae")
                self._recover_from_oom(
                    stage=target_stage,
                    profile=fallback_profile,
                    required=recovery_required,
                    boundary=contract,
                    retry_index=int(attempt["retry_index"]),
                )
            except Exception:
                if contract is not None:
                    contract.release()
                raise

        retry_profile_override = requested_profile_override
        retry_required = required_set
        retry_preferred = preferred_set
        retry_optional = optional_set
        retry_preview_requires_vae = preview_requires_vae
        if fallback_profile in {"low_vram", "maximum"}:
            retry_profile_override = "low_vram"
            retry_required = set(required_set)
            if self.oom.sampler_stage(target_stage):
                retry_required.discard("vae")
            retry_preferred = set()
            retry_optional = set()
            retry_preview_requires_vae = False

        context_report: dict[str, Any] = {}
        try:
            with self._oom_retry_context(
                stage=target_stage,
                profile=fallback_profile,
            ) as active_context_report:
                if isinstance(active_context_report, dict):
                    context_report = active_context_report
                with self.lease(
                    stage=target_stage,
                    required=retry_required,
                    preferred=retry_preferred,
                    optional=retry_optional,
                    estimated_stage_bytes=estimated_stage_bytes,
                    preview_requires_vae=retry_preview_requires_vae,
                    requested_profile_override=retry_profile_override,
                    safety_margin_bytes_override=safety_margin_bytes_override,
                ):
                    result = operation_for_attempt(1)()
            self.oom.record_action(
                target_stage,
                "oom_retry_context",
                fallback_profile=fallback_profile,
                attention_slicing=dict(context_report),
            )
            creative_final = (
                self._creative_snapshot(request) if request is not None else {}
            )
            if creative_final != creative_before:
                raise RuntimeError(
                    "OOM recovery retry changed creative request settings, which is not allowed."
                )
            self.oom.record_result(target_stage, "retry_succeeded")
            self._emit_status("oom_retry_succeeded", stage=target_stage)
            return result
        except Exception as retry_exc:
            self.oom.record_action(
                target_stage,
                "oom_retry_context",
                fallback_profile=fallback_profile,
                attention_slicing=dict(context_report),
            )
            self.oom.record_result(
                target_stage,
                "retry_failed",
                error=retry_exc,
            )
            self.failure_bundle = memory_failure_bundle(
                request=request,
                stage=target_stage,
                error=retry_exc,
                manager_summary=self.summary(),
                dimension_plan=self._dimension_plan,
            )
            self._emit_status("oom_retry_failed", stage=target_stage)
            raise
        finally:
            if contract is not None:
                contract.release()

    def _emit_status(self, event: str, *, stage: str | None = None) -> None:
        callback = self.event_callback
        if not callable(callback):
            return
        try:
            callback({
                "event": str(event),
                "stage": stage,
                "active_stage": self.active_stage,
                "status": self.status_payload(),
            })
        except Exception:
            pass

    def status_payload(self) -> dict[str, Any]:
        components = self.registry.snapshot()
        gpu_components = [item["component_id"] for item in components if str(item.get("current_device", "")).startswith("cuda")]
        offloaded_components = [item["component_id"] for item in components if item["component_id"] not in gpu_components]
        latest_cuda = dict((self.latest_snapshot or {}).get("cuda") or {})
        return {
            "requested_policy": self.settings.policy,
            "effective_policy": self._effective_profile,
            "active_stage": self.active_stage,
            "latest_snapshot": self.latest_snapshot,
            "latest_estimate": self.latest_estimate,
            "active_gpu_components": gpu_components,
            "offloaded_components": offloaded_components,
            "automatic_actions": list(self.automatic_actions[-20:]),
            "component_transfer_count": len(self.registry.transfer_records),
            "oom_retry_count_by_stage": dict(self.oom.retry_count_by_stage),
            "oom_recovery_count": self.oom.total_retry_count,
            "oom_recovery_profile": self.oom.configured_profile,
            "oom_retry_limit": self.oom.retry_limit,
            "oom_runtime_path_changed": self.oom.runtime_path_changed,
            "preview_policy": self.settings.preview_policy,
            "preview_image_decode_suspended": self.preview_image_decode_suspended,
            "preview_image_decode_suspension_reason": self.preview_image_decode_suspension_reason,
            "preview_image_decode_suspension_source": self.preview_image_decode_suspension_source,
            "preview_decoder_released": self.preview_decoder_released,
            "preview_suspension_one_way_for_job": self.preview_suspension_one_way_for_job,
            "cfg_telemetry_continues_during_preview_suspension": True,
            "peak_allocated_vram_bytes": latest_cuda.get("peak_allocated_vram_bytes"),
            "peak_reserved_vram_bytes": latest_cuda.get("peak_reserved_vram_bytes"),
            "peak_vram_by_stage": {
                stage: dict(values)
                for stage, values in self.peak_vram_by_stage.items()
            },
            "settings": self.settings.to_dict(),
            "hires_cleanup_reports": list(self.hires_cleanup_reports),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "format": "image-gen-memory-management-v1",
            **self.status_payload(),
            "component_residency": self.registry.snapshot(),
            "transfers": [record.to_dict() for record in self.registry.transfer_records],
            "plans": list(self.plans),
            "estimates": list(self.estimates),
            "snapshots": list(self.snapshots),
            "oom_retry_count_by_stage": dict(self.oom.retry_count_by_stage),
            "oom_recovery_actions": list(self.oom.actions),
            "oom_recovery": self.oom.summary(),
            "hires_cleanup_reports": list(self.hires_cleanup_reports),
            "failure_bundle": self.failure_bundle,
        }
