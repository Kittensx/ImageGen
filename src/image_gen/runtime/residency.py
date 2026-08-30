from __future__ import annotations

import copy
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Mapping

import torch

from image_gen.program_metadata import PRODUCT_NAME
from image_gen.contracts import resolve_latent_vae_contract
from image_gen.systems.model_loading.system import LoadedModel
from image_gen.runtime.model_load_variant import (
    model_load_variant_fingerprint,
    model_load_variant_payload,
    model_load_variant_payload_fingerprint,
    resolved_model_load_variant_payload,
)
from image_gen.systems.memory.policy import normalize_policy
from image_gen.systems.memory.oom_recovery import is_cuda_oom
from image_gen.runtime.residency_policy import MODEL_RESIDENCY_MODE_HOT, normalize_model_residency_mode
from image_gen.runtime.component_residency import (
    build_resident_component_inventory,
    public_transition_report,
    resident_reuse_bundle,
)
from image_gen.runtime.composition_leases import (
    CompositionExecutionLease,
    normalize_planned_schedule,
)
from image_gen.runtime.composition_transitions import (
    plan_execution_lease_transition,
)
from modules.registry.family_providers import DEFAULT_FAMILY_PROVIDER_REGISTRY
from modules.registry.contracts import CompositionIdentity
from modules.component_placement import place_component
from modules.txt2img.request_loader import payload_to_generation_request


class ResidencyMixin:
    @staticmethod
    def _runtime_component_entries(components: Any) -> list[tuple[str, Any]]:
        if components is None:
            return []
        denoiser_kind = str(getattr(components, "denoiser_kind", "unet") or "unet").strip().lower()
        entries: list[tuple[str, Any]] = []
        if denoiser_kind == "transformer":
            entries.append(("transformer", getattr(components, "denoiser", None)))
        else:
            entries.append(("unet", getattr(components, "unet", None)))
        entries.extend([
            ("text_encoder", getattr(components, "text_encoder", None)),
            ("text_encoder_2", getattr(components, "text_encoder_2", None)),
            ("text_encoder_3", getattr(components, "text_encoder_3", None)),
            ("vae", getattr(components, "vae", None)),
        ])
        seen: set[int] = set()
        unique: list[tuple[str, Any]] = []
        for name, module in entries:
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            unique.append((name, module))
        return unique

    def resident_component_reuse_bundle(self) -> dict[str, Any]:
        lora_manager = getattr(self, "lora_runtime_manager", None)
        adapter_dirty = bool(
            getattr(lora_manager, "_loaded_adapters", {})
            or getattr(lora_manager, "_active_signature", ())
        )
        return resident_reuse_bundle(
            self.last_loaded_model,
            adapter_state_dirty=adapter_dirty,
        )

    def composition_lease_status(self) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None:
            return {
                "schema_version": 1,
                "state": "inactive",
                "generation": int(getattr(self, "_composition_lease_generation", 0) or 0),
                "component_pool_count": 0,
                "warm_states": {},
            }
        return lease.public_status()

    def _resolve_composition_lease_schedule(self, raw_schedule: Any) -> list[dict[str, Any]]:
        schedule = normalize_planned_schedule(raw_schedule)
        if not schedule:
            return []
        registry = getattr(getattr(self, "model_loader", None), "asset_registry", None)
        get_asset = getattr(registry, "get_asset_by_path", None)
        get_snapshots = getattr(registry, "get_component_snapshots", None)
        if not callable(get_asset) or not callable(get_snapshots):
            raise RuntimeError("CNRR-06 requires read access to registered component snapshots.")
        resolved: list[dict[str, Any]] = []
        for index, raw in enumerate(schedule):
            item = dict(raw)
            model_path = str(Path(str(item.get("model_path") or "")).expanduser().resolve(strict=False))
            asset = get_asset(model_path)
            if asset is None:
                raise RuntimeError(f"Planned composition is not registered: {model_path}")
            family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(getattr(asset, "architecture", ""))
            provider = DEFAULT_FAMILY_PROVIDER_REGISTRY.get(family) if family else None
            provider_version = str(getattr(provider, "version", "") or "")
            role_hashes: dict[str, str] = {}
            role_bytes: dict[str, int] = {}
            ambiguous: set[str] = set()
            for snapshot in tuple(get_snapshots(int(asset.id)) or ()):
                role = str(getattr(snapshot, "component_role", "") or "").strip()
                digest = str(getattr(snapshot, "component_sha256", "") or "").strip().lower()
                if not role or len(digest) != 64:
                    continue
                previous = role_hashes.get(role)
                if previous and previous != digest:
                    ambiguous.add(role)
                    continue
                role_hashes[role] = digest
                role_bytes[role] = int(getattr(snapshot, "total_bytes", 0) or 0)
            for role in ambiguous:
                role_hashes.pop(role, None)
                role_bytes.pop(role, None)
            if not family or not provider_version or not role_hashes:
                raise RuntimeError(f"Planned composition lacks complete registered identity evidence: {model_path}")
            identity = CompositionIdentity.derive(
                family=family,
                provider_version=provider_version,
                components=role_hashes,
            )
            try:
                stat = Path(model_path).stat()
                source_signature = {
                    "file_size_bytes": int(stat.st_size),
                    "modified_ns": int(stat.st_mtime_ns),
                }
            except OSError:
                source_signature = {}
            resolved.append({
                **item,
                "index": index,
                "model_path": model_path,
                "asset_id": int(asset.id),
                "family": family,
                "provider_version": provider_version,
                "components": role_hashes,
                "component_bytes": role_bytes,
                "composition_sha256": identity.composition_sha256,
                "whole_checkpoint_sha256": str(getattr(asset, "sha256", "") or "").strip().lower(),
                "source_signature": source_signature,
            })
        return resolved

    def ensure_composition_execution_lease(
        self,
        raw_schedule: Any,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_planned_schedule(raw_schedule)
        requested_paths = [
            str(Path(str(item.get("model_path") or "")).expanduser().resolve(strict=False))
            for item in normalized
        ]
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is not None and lease.state == "active":
            existing_paths = [
                str(Path(str(item.get("model_path") or "")).expanduser().resolve(strict=False))
                for item in lease.schedule
            ]
            if requested_paths and requested_paths == existing_paths:
                return {"established": False, "reused": True, **lease.public_status()}
        return self.establish_composition_execution_lease(raw_schedule, settings)

    def establish_composition_execution_lease(
        self,
        raw_schedule: Any,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del settings
        if self.last_loaded_model is None:
            return {"established": False, "reason": "no_active_resident_composition"}
        schedule = self._resolve_composition_lease_schedule(raw_schedule)
        if not schedule:
            return {"established": False, "reason": "planned_schedule_empty"}
        status = self.resident_model_status()
        current_path = str(Path(str(status.get("model_path") or "")).expanduser().resolve(strict=False))
        current_family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(status.get("architecture"))
        active_index = next((
            index for index, item in enumerate(schedule)
            if str(Path(str(item.get("model_path") or "")).expanduser().resolve(strict=False)) == current_path
        ), None)
        if active_index is None:
            return {
                "established": False,
                "reason": "active_model_not_present_in_planned_schedule",
                "active_model_path": current_path,
            }
        families = {str(item.get("family") or "") for item in schedule}
        if len(families) != 1 or current_family not in families:
            return {
                "established": False,
                "reason": "cnrr06_same_family_schedule_required",
                "families": sorted(families),
                "active_family": current_family,
            }
        provider_versions = {str(item.get("provider_version") or "") for item in schedule}
        if len(provider_versions) != 1:
            return {
                "established": False,
                "reason": "provider_contract_version_mismatch",
                "provider_versions": sorted(provider_versions),
            }
        previous = getattr(self, "_composition_execution_lease", None)
        if previous is not None:
            previous.invalidate("replaced_by_new_execution_schedule")
        self._composition_lease_generation = int(getattr(self, "_composition_lease_generation", 0) or 0) + 1
        lease = CompositionExecutionLease(
            generation=self._composition_lease_generation,
            schedule=schedule,
            active_index=int(active_index),
            family=current_family,
            provider_version=next(iter(provider_versions)),
        )
        lora_manager = getattr(self, "lora_runtime_manager", None)
        adapter_dirty = bool(
            getattr(lora_manager, "_loaded_adapters", {})
            or getattr(lora_manager, "_active_signature", ())
        )
        lease.register_active_loaded(self.last_loaded_model, adapter_state_dirty=adapter_dirty)
        self._composition_execution_lease = lease
        return {"established": True, **lease.public_status()}

    def execution_lease_reuse_bundle(self) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or str(getattr(lease, "state", "")) != "active":
            return self.resident_component_reuse_bundle()
        lora_manager = getattr(self, "lora_runtime_manager", None)
        adapter_dirty = bool(
            getattr(lora_manager, "_loaded_adapters", {})
            or getattr(lora_manager, "_active_signature", ())
        )
        return lease.reusable_bundle(
            self.last_loaded_model,
            adapter_state_dirty=adapter_dirty,
        )

    def _warm_planned_composition_worker(
        self,
        *,
        lease_generation: int,
        index: int,
        settings: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or lease.generation != lease_generation or lease.state != "active":
            return {"warmed": False, "reason": "lease_no_longer_active"}
        item = dict(lease.schedule[index])
        model_path = str(item.get("model_path") or "")
        lease.mark_warming(index, model_path)
        started = time.perf_counter()
        try:
            if bool(dict(item.get("runtime_settings") or {}).get("advanced_models_enabled")):
                raise RuntimeError("CNRR-06 background warm currently requires a whole-checkpoint same-family target.")
            source_signature = dict(item.get("source_signature") or {})
            if source_signature:
                stat = Path(model_path).stat()
                if (
                    int(source_signature.get("file_size_bytes") or -1) != int(stat.st_size)
                    or int(source_signature.get("modified_ns") or -1) != int(stat.st_mtime_ns)
                ):
                    raise RuntimeError("planned_checkpoint_source_changed_before_warm")

            from modules.load_safetensors_model import LoadModel

            loader = LoadModel(project_context=self.project_context)
            extras = dict(settings or {})
            target_runtime_settings = dict(item.get("runtime_settings") or {})
            # The settings passed to the warm worker originate from the currently
            # active segment and may already contain an automatically-resolved
            # runtime profile (for example SDXL Lightning 2-step).  That resolved
            # profile is evidence about the active checkpoint, not authority for a
            # different future checkpoint.  Future segments infer their own profile
            # unless the schedule entry explicitly declares an override.
            for profile_field in (
                "sd2_runtime_profile_override",
                "sdxl_runtime_profile_override",
                "sd3_runtime_profile_override",
            ):
                if profile_field not in target_runtime_settings:
                    extras.pop(profile_field, None)
            extras.update(target_runtime_settings)
            extras["_component_transition_requested"] = True
            extras["_resident_component_reuse_bundle"] = lease.reusable_bundle(self.last_loaded_model)
            extras["_component_warm_stage_only"] = True
            extras["model_runtime_execution_device"] = "cpu"
            prepare_kwargs: dict[str, Any] = {
                "require_generation_support": True,
                "explicit_sd2_runtime_profile": str(extras.get("sd2_runtime_profile_override") or "").strip() or None,
                "explicit_sdxl_runtime_profile": str(extras.get("sdxl_runtime_profile_override") or "").strip() or None,
                "explicit_sd3_runtime_profile": str(extras.get("sd3_runtime_profile_override") or "").strip() or None,
                "request_extras": extras,
            }
            plan = loader.prepare_load_plan(model_path, **prepare_kwargs)
            built = loader.build_components_from_plan(
                plan,
                dtype=self.dtype,
                device=torch.device("cpu"),
                request_extras=extras,
            )
            current = getattr(self, "_composition_execution_lease", None)
            if current is None or current.generation != lease_generation or current.state != "active":
                return {"warmed": False, "reason": "lease_invalidated_during_warm"}
            registration = lease.register_warmed_components(
                index=index,
                model_path=model_path,
                components=built,
                target_components=dict(item.get("components") or {}),
                load_plan=plan,
                runtime_settings=extras,
            )
            result = {
                "warmed": True,
                "index": index,
                "model_path": model_path,
                "warm_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "runtime_source_plan": dict(extras.get("_runtime_component_source_plan") or {}),
                "component_transition_report": public_transition_report(
                    extras.get("_component_transition_plan") or {}
                ),
                **registration,
            }
            state = dict(lease.warm_states.get(index) or {})
            state.update({key: value for key, value in result.items() if key not in {"runtime_source_plan", "component_transition_report"}})
            lease.warm_states[index] = state
            return result
        except BaseException as exc:
            lease.mark_warm_failed(index, model_path, exc)
            return {
                "warmed": False,
                "index": index,
                "model_path": model_path,
                "warm_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def prime_composition_prefetch(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        depth: int | None = None,
    ) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or lease.state != "active":
            return {"scheduled": [], "reason": "no_active_execution_lease"}
        values = dict(settings or {})
        requested_depth = int(depth if depth is not None else values.get("model_runtime_prefetch_depth", 1) or 1)
        requested_depth = max(0, min(requested_depth, 4))
        if requested_depth <= 0:
            return {"scheduled": [], "reason": "prefetch_disabled"}
        if getattr(self, "_composition_warm_executor", None) is None:
            self._composition_warm_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="imagegen-cnrr06-warm",
            )
        scheduled: list[int] = []
        cursor = lease.active_index
        for _ in range(requested_depth):
            next_index = None
            for index in range(cursor + 1, len(lease.schedule)):
                state = str(dict(lease.warm_states.get(index) or {}).get("state") or "")
                if state in {"queued", "warming", "warm"}:
                    cursor = index
                    continue
                target = dict(lease.schedule[index].get("components") or {})
                if all(
                    lease._pool_key(str(role), str(digest).lower()) in lease.component_pool
                    for role, digest in target.items() if digest
                ):
                    lease.warm_states[index] = {
                        "state": "warm",
                        "model_path": str(lease.schedule[index].get("model_path") or ""),
                        "reason": "all_components_already_leased",
                        "new_component_roles": [],
                    }
                    cursor = index
                    continue
                next_index = index
                break
            if next_index is None:
                break
            lease.mark_queued(
                next_index,
                str(lease.schedule[next_index].get("model_path") or ""),
            )
            future = self._composition_warm_executor.submit(
                self._warm_planned_composition_worker,
                lease_generation=lease.generation,
                index=next_index,
                settings=values,
            )
            self._composition_warm_futures[next_index] = future
            scheduled.append(next_index)
            cursor = next_index
        return {
            "scheduled": scheduled,
            "prefetch_depth": requested_depth,
            "worker_count": 1,
            "lease_generation": lease.generation,
        }

    def wait_for_composition_prefetch(
        self,
        index: int,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        future = dict(getattr(self, "_composition_warm_futures", {}) or {}).get(int(index))
        if future is None:
            lease = getattr(self, "_composition_execution_lease", None)
            state = dict(getattr(lease, "warm_states", {}).get(int(index)) or {}) if lease is not None else {}
            return {"completed": state.get("state") == "warm", "state": state}
        result = future.result(timeout=timeout)
        return {"completed": bool(result.get("warmed")), "result": result}

    def on_composition_committed_to_lease(self, model_path: str) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or lease.state != "active":
            return {"updated": False, "reason": "no_active_execution_lease"}
        index = lease.index_for_model(model_path, after_current=True)
        if index is None:
            # Same-model rebuild/reuse does not advance the execution schedule.
            current_path = str(lease.schedule[lease.active_index].get("model_path") or "")
            if Path(str(current_path)).resolve(strict=False) == Path(str(model_path)).resolve(strict=False):
                lease.register_active_loaded(self.last_loaded_model)
                return {"updated": True, "active_index": lease.active_index, "reason": "same_segment_refresh"}
            return {"updated": False, "reason": "committed_model_not_next_in_schedule"}
        lease.set_active_index(index, self.last_loaded_model)
        return {"updated": True, "active_index": index, "reason": "planned_composition_committed"}

    def invalidate_composition_execution_lease(self, reason: str) -> dict[str, Any]:
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None:
            return {"invalidated": False, "reason": "no_active_execution_lease"}
        lease.invalidate(reason)
        for future in list(dict(getattr(self, "_composition_warm_futures", {}) or {}).values()):
            if not future.done():
                future.cancel()
        self._composition_warm_futures = {}
        self._cnrr07_prepared_transition_context = {}
        self._last_cnrr07_transition = {}
        return {"invalidated": True, "reason": str(reason), "generation": lease.generation}

    def plan_prepared_composition_transition(
        self,
        target: int | str,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a serializable CNRR-07 transition plan without mutating runtime state."""

        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or lease.state != "active":
            return plan_execution_lease_transition(
                lease, self.last_loaded_model, target_index=int(target) if isinstance(target, int) else -1,
                required_device="cpu",
            ).to_dict()
        if isinstance(target, int):
            target_index = int(target)
        else:
            target_index = lease.index_for_model(str(target), after_current=True)
            if target_index is None:
                target_index = -1
        values = dict(settings or {})
        try:
            device, _ = self._resolve_execution_device(values)
            required_device = str(device)
        except Exception:
            required_device = "cpu"
        return plan_execution_lease_transition(
            lease,
            self.last_loaded_model,
            target_index=target_index,
            required_device=required_device,
        ).to_dict()

    @staticmethod
    def _cnrr07_set_role(container: Any, role: str, module: Any) -> None:
        if role == "transformer":
            setattr(container, "denoiser", module)
            if hasattr(container, "transformer"):
                setattr(container, "transformer", module)
            return
        setattr(container, role, module)
        if role == "unet":
            setattr(container, "denoiser", module)

    def _cnrr07_assemble_prepared_loaded(
        self,
        *,
        lease: CompositionExecutionLease,
        target_index: int,
        transition_plan: Mapping[str, Any],
    ) -> LoadedModel:
        prepared = lease.prepared_composition(target_index)
        if prepared is None:
            raise RuntimeError("CNRR-07 target prepared composition is unavailable.")
        if prepared.loaded_model is not None:
            loaded = prepared.loaded_model
        else:
            current = self.last_loaded_model
            if current is None:
                raise RuntimeError("CNRR-07 requires an active resident composition.")
            current_components = getattr(current, "components", None)
            built = prepared.built_components
            plan = prepared.load_plan
            if current_components is None or built is None or plan is None:
                raise RuntimeError("CNRR-07 prepared composition lacks required process-local build metadata.")
            components = copy.copy(current_components)
            target_item = dict(lease.schedule[target_index] or {})
            target_hashes = dict(target_item.get("components") or {})
            for role, digest in target_hashes.items():
                pool_entry = lease.component_pool.get(lease._pool_key(str(role), str(digest)))
                if pool_entry is None or pool_entry.module is None:
                    raise RuntimeError(f"CNRR-07 prepared component missing from lease pool: {role}")
                self._cnrr07_set_role(components, str(role), pool_entry.module)

            # Preserve target-local tokenizers prepared by the canonical builder.
            for name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
                value = getattr(built, name, None)
                if value is not None:
                    setattr(components, name, value)

            report = getattr(plan, "report", None)
            prediction_type, prediction_source = self.model_loading_system._prediction_contract(plan)
            components.prediction_type = prediction_type
            components.prediction_type_source = prediction_source
            components.architecture = str(getattr(report, "architecture", "") or lease.family)
            components.model_identity = str(
                target_item.get("whole_checkpoint_sha256")
                or target_item.get("composition_sha256")
                or target_item.get("model_path")
                or ""
            )
            components.model_hash = str(target_item.get("whole_checkpoint_sha256") or "")

            identity = CompositionIdentity.derive(
                family=lease.family,
                provider_version=lease.provider_version,
                components=target_hashes,
            )
            contract = identity.to_dict()
            components.composition_sha256 = identity.composition_sha256
            components.composition_identity_version = identity.identity_version
            components.composition_contract = contract
            components.composition_projection = {
                "schema_version": 1,
                "status": "complete",
                "reason": "cnrr07_prepared_execution_lease_commit",
                "composition_sha256": identity.composition_sha256,
                "composition_contract": contract,
            }
            components.component_sources = {
                str(role): {
                    **dict(getattr(lease.component_pool.get(lease._pool_key(str(role), str(digest))), "source", {}) or {}),
                    "source_kind": str(
                        dict(transition_plan.get("source_plan") or {}).get(str(role), {}).get("selected_source_kind")
                        or "execution_lease"
                    ),
                    "component_sha256": str(digest),
                }
                for role, digest in target_hashes.items()
            }
            components.runtime_component_source_plan = {
                "schema_version": 1,
                "mode": "cnrr07_prepared_transition",
                "roles": {str(k): dict(v) for k, v in dict(transition_plan.get("source_plan") or {}).items()},
                "active_transaction_hydration_roles": [],
            }
            components.component_transition_report = {
                "schema_version": 1,
                "transition_classification": str(transition_plan.get("transition_class") or ""),
                "role_diff": {str(k): dict(v) for k, v in dict(transition_plan.get("role_diff") or {}).items()},
                "requested_composition_sha256": identity.composition_sha256,
                "cnrr07_direct_commit": True,
            }

            sd3_contract = getattr(plan, "sd3_contract", None)
            sdxl_contract = getattr(plan, "sdxl_contract", None)
            sd2_contract = getattr(plan, "sd2_contract", None)
            if sd3_contract is not None:
                runtime_profile = dict(sd3_contract.profile.to_dict())
                runtime_profile["text_encoder_sources"] = dict(getattr(built, "sd3_text_encoder_sources", {}) or {})
            elif sdxl_contract is not None:
                runtime_profile = dict(sdxl_contract.profile.to_dict())
            elif sd2_contract is not None:
                runtime_profile = dict(sd2_contract.profile.to_dict())
            else:
                runtime_profile = {}
            components.model_runtime_profile = runtime_profile

            old_contract = dict(getattr(current_components, "composition_contract", {}) or {})
            old_vae_sha = str(dict(old_contract.get("components") or {}).get("vae") or "")
            new_vae_sha = str(target_hashes.get("vae") or "")
            if new_vae_sha and new_vae_sha != old_vae_sha:
                active_vae = getattr(components, "vae", None)
                vae_scaling_factor = float(
                    sd3_contract.assets.vae_payload().get("scaling_factor", 1.5305)
                    if sd3_contract is not None
                    else getattr(sdxl_contract, "vae_scaling_factor", 0.18215)
                    if sdxl_contract is not None
                    else 0.18215
                )
                latent = resolve_latent_vae_contract(SimpleNamespace(
                    vae=active_vae,
                    latent_channels=int(getattr(sd3_contract, "latent_channels", 16)) if sd3_contract is not None else 4,
                    latent_scale_factor=8,
                    vae_scaling_factor=vae_scaling_factor,
                    vae_shift_factor=float(sd3_contract.assets.vae_payload().get("shift_factor", 0.0609)) if sd3_contract is not None else 0.0,
                    vae_force_upcast=bool(getattr(sdxl_contract, "vae_force_upcast", False)) if sdxl_contract is not None else bool(getattr(built, "vae_force_upcast", False)),
                ))
                components.latent_channels = latent.latent_channels
                components.latent_scale_factor = latent.latent_scale_factor
                components.vae_scaling_factor = latent.scaling_factor
                components.vae_shift_factor = latent.shift_factor
                components.vae_force_upcast = latent.force_upcast
                components.vae_use_quant_conv = latent.use_quant_conv
                components.vae_use_post_quant_conv = latent.use_post_quant_conv

            built_copy = copy.copy(built)
            for role, digest in target_hashes.items():
                pool_entry = lease.component_pool.get(lease._pool_key(str(role), str(digest)))
                self._cnrr07_set_role(built_copy, str(role), pool_entry.module if pool_entry is not None else None)
            loaded = LoadedModel(
                components=components,
                load_plan=plan,
                built_components=built_copy,
                memory_telemetry=None,
            )
            prepared.loaded_model = loaded
        return loaded

    def execute_prepared_composition_transition(
        self,
        target: int | str,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit an already-prepared same-family composition without preload/build.

        This is the CNRR-07 internal transition boundary intended for later sampler
        step scheduling. It does not itself change sampler state or conditioning.
        """

        started = time.perf_counter()
        lease = getattr(self, "_composition_execution_lease", None)
        if lease is None or lease.state != "active":
            raise RuntimeError("CNRR-07 requires an active composition execution lease.")
        if isinstance(target, int):
            target_index = int(target)
        else:
            found = lease.index_for_model(str(target), after_current=True)
            if found is None:
                raise RuntimeError("CNRR-07 target is not the next planned composition.")
            target_index = int(found)
        values = dict(settings or {})
        execution_device, fallback_reason = self._resolve_execution_device(values)
        plan = plan_execution_lease_transition(
            lease,
            self.last_loaded_model,
            target_index=target_index,
            required_device=str(execution_device),
        )
        public_plan = plan.to_dict()
        if not plan.ready:
            raise RuntimeError("CNRR-07 prepared transition is not ready: " + ", ".join(plan.reasons))

        target_loaded = self._cnrr07_assemble_prepared_loaded(
            lease=lease,
            target_index=target_index,
            transition_plan=public_plan,
        )
        target_components_runtime = getattr(target_loaded, "components", None)
        prepared = lease.prepared_composition(target_index)
        prepared_plan = getattr(prepared, "load_plan", None) if prepared is not None else None
        profile_ids: dict[str, str] = {}
        for field, contract in (
            ("sd2_runtime_profile_override", getattr(prepared_plan, "sd2_contract", None)),
            ("sdxl_runtime_profile_override", getattr(prepared_plan, "sdxl_contract", None)),
            ("sd3_runtime_profile_override", getattr(prepared_plan, "sd3_contract", None)),
        ):
            profile = getattr(contract, "profile", None)
            profile_id = str(getattr(profile, "profile_id", "") or "").strip()
            if profile_id:
                profile_ids[field] = profile_id
        raw_variant = model_load_variant_payload(values)
        effective_variant = resolved_model_load_variant_payload(
            values,
            profile_ids=profile_ids,
            sd3_text_encoder_sources=dict(
                dict(getattr(target_components_runtime, "model_runtime_profile", {}) or {}).get("text_encoder_sources") or {}
            ),
        )
        if target_components_runtime is not None:
            target_components_runtime.runtime_load_variant = raw_variant
            target_components_runtime.runtime_load_variant_fingerprint = model_load_variant_payload_fingerprint(raw_variant)
            target_components_runtime.runtime_effective_load_variant = effective_variant
            target_components_runtime.runtime_effective_load_variant_fingerprint = model_load_variant_payload_fingerprint(effective_variant)

            # A prepared atomic commit changes checkpoint authority just as surely as
            # a canonical preload.  Rebind the target path/SHA/source signature at
            # the same transaction boundary as the component composition; otherwise
            # the next ordinary request sees B's model path paired with A's stale
            # checkpoint proof and correctly forces a conventional reload.
            target_path = Path(str(lease.schedule[target_index].get("model_path") or "")).expanduser().resolve(strict=False)
            target_signature = dict(lease.schedule[target_index].get("source_signature") or {})
            if target_path.is_file() and (
                not target_signature.get("file_size_bytes")
                or not target_signature.get("modified_ns")
            ):
                target_stat = target_path.stat()
                target_signature = {
                    "file_size_bytes": int(target_stat.st_size),
                    "modified_ns": int(target_stat.st_mtime_ns),
                }
            target_sha256 = str(
                lease.schedule[target_index].get("whole_checkpoint_sha256")
                or getattr(getattr(prepared_plan, "report", None), "sha256", "")
                or ""
            ).strip().lower()
            target_components_runtime.runtime_checkpoint_identity = {
                "path": str(target_path),
                "sha256": target_sha256,
                "file_size_bytes": target_signature.get("file_size_bytes"),
                "modified_ns": target_signature.get("modified_ns"),
                "proof": (
                    "resident_sha256_bound_to_source_file_signature"
                    if target_sha256
                    else "source_file_signature"
                ),
            }
        current_loaded = self.last_loaded_model
        current_inventory = build_resident_component_inventory(current_loaded) if current_loaded is not None else {}
        target_item = dict(lease.schedule[target_index] or {})
        target_hashes = dict(target_item.get("components") or {})

        moves: list[dict[str, Any]] = []
        rollback: list[tuple[Any, str]] = []
        try:
            # First demote only outgoing changed GPU modules. Exact shared modules are
            # untouched, and leased modules required again remain hydrated on CPU.
            for role, diff in public_plan["role_diff"].items():
                if str(diff.get("action") or "") not in {"replace", "remove"}:
                    continue
                entry = current_inventory.get(role)
                module = getattr(entry, "module", None) if entry is not None else None
                if module is None or not hasattr(module, "to"):
                    continue
                before = str(getattr(entry, "device", "") or "")
                if before.startswith("cuda"):
                    rollback.append((module, before))
                    module.to(torch.device("cpu"))
                    moves.append({"role": role, "direction": "demote_outgoing", "from": before, "to": "cpu"})

            # Promote only changed/add target modules. Retained shared modules keep
            # their current placement and therefore cannot be disturbed by the swap.
            for role, diff in public_plan["role_diff"].items():
                if str(diff.get("action") or "") not in {"replace", "add"}:
                    continue
                digest = str(target_hashes.get(role) or "")
                entry = lease.component_pool.get(lease._pool_key(role, digest))
                module = getattr(entry, "module", None) if entry is not None else None
                if module is None or not hasattr(module, "to"):
                    raise RuntimeError(f"CNRR-07 target live module unavailable during commit: {role}")
                before = str(getattr(entry, "device", "") or "")
                if execution_device.type == "cuda" and not before.startswith("cuda"):
                    rollback.append((module, before))
                    module.to(device=execution_device, dtype=self.dtype)
                    moves.append({"role": role, "direction": "promote_incoming", "from": before, "to": str(execution_device)})
                elif execution_device.type == "cpu" and not before.startswith("cpu"):
                    rollback.append((module, before))
                    module.to(device=execution_device)
                    moves.append({"role": role, "direction": "restage_incoming", "from": before, "to": str(execution_device)})

            previous_loaded = self.last_loaded_model
            previous_cache = dict(self._loaded_model_cache)
            try:
                self.last_loaded_model = target_loaded
                self._loaded_model_cache.clear()
                self._loaded_model_cache[("cnrr07-prepared", lease.generation, target_index)] = target_loaded
                lease.set_active_index(target_index, target_loaded)
                self._cnrr07_prepared_transition_context = {
                    "lease_generation": lease.generation,
                    "target_index": target_index,
                    "model_path": str(target_item.get("model_path") or ""),
                    "requested_load_variant_fingerprint": model_load_variant_fingerprint(values),
                    "transition_plan": public_plan,
                    "source_signature": dict(target_item.get("source_signature") or {}),
                }
            except Exception:
                self.last_loaded_model = previous_loaded
                self._loaded_model_cache.clear()
                self._loaded_model_cache.update(previous_cache)
                raise
        except BaseException:
            # Best-effort placement rollback keeps the previous composition usable if
            # promotion fails before the atomic resident pointer/cache commit.
            for module, original in reversed(rollback):
                try:
                    if original and original not in {"missing", "unknown"}:
                        module.to(torch.device(original))
                except Exception:
                    pass
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        result = {
            "committed": True,
            "transition_mode": "cnrr07_prepared_atomic_commit",
            "target_index": target_index,
            "target_model_path": str(target_item.get("model_path") or ""),
            "transition_time_ms": elapsed_ms,
            "checkpoint_hydration_roles": [],
            "pipeline_rebuild_performed": False,
            "device_moves": moves,
            "execution_device": str(execution_device),
            "cpu_fallback_reason": fallback_reason,
            "plan": public_plan,
            "lease": lease.public_status(),
        }
        self._last_cnrr07_transition = dict(result)
        return result

    def component_transition_eligibility(self, model_path: str) -> dict[str, Any]:
        """Cheaply decide whether preserving the current composition is worthwhile.

        This gate only uses resident CNRR-03 identity plus already-indexed target
        registry component snapshots.  It never scans or hashes the target model.
        Unknown evidence fails closed to the historical full-rebuild path.
        """

        bundle = self.resident_component_reuse_bundle()
        resident_family = str(bundle.get("family") or "")
        inventory = dict(bundle.get("entries") or {})
        if not resident_family or not inventory:
            return {"eligible": False, "reason": "resident_component_identity_unavailable"}
        if not any(bool(getattr(entry, "reuse_eligible", False)) for entry in inventory.values()):
            return {"eligible": False, "reason": "resident_components_not_reuse_eligible"}

        registry = getattr(getattr(self, "model_loader", None), "asset_registry", None)
        get_asset = getattr(registry, "get_asset_by_path", None)
        get_snapshots = getattr(registry, "get_component_snapshots", None)
        if not callable(get_asset) or not callable(get_snapshots):
            return {"eligible": False, "reason": "registry_transition_evidence_unavailable"}
        try:
            asset = get_asset(str(Path(model_path).expanduser().resolve(strict=False)))
        except OSError:
            asset = get_asset(str(model_path))
        if asset is None:
            return {"eligible": False, "reason": "target_checkpoint_not_registered"}
        target_family = DEFAULT_FAMILY_PROVIDER_REGISTRY.canonicalize(
            getattr(asset, "architecture", "")
        )
        if not target_family or target_family != resident_family:
            return {
                "eligible": False,
                "reason": "architecture_family_mismatch",
                "resident_family": resident_family,
                "target_family": target_family,
            }
        target_hashes = {
            str(getattr(item, "component_role", "") or ""): str(
                getattr(item, "component_sha256", "") or ""
            ).strip().lower()
            for item in tuple(get_snapshots(int(asset.id)) or ())
            if str(getattr(item, "component_role", "") or "").strip()
            and str(getattr(item, "component_sha256", "") or "").strip()
        }
        shared_roles = sorted(
            role
            for role, entry in inventory.items()
            if bool(getattr(entry, "reuse_eligible", False))
            and target_hashes.get(role) == str(getattr(entry, "component_sha256", "") or "")
        )
        if not shared_roles:
            return {
                "eligible": False,
                "reason": "no_known_exact_component_overlap",
                "resident_family": resident_family,
                "target_family": target_family,
            }
        return {
            "eligible": True,
            "reason": "same_family_exact_component_overlap",
            "resident_family": resident_family,
            "target_family": target_family,
            "shared_roles": shared_roles,
            "target_asset_id": int(asset.id),
        }

    def clear_model_cache(self, *, move_components_to_cpu: bool = True) -> dict[str, Any]:
        """Release cached checkpoint components and return before/after telemetry.

        CUDA's process context and third-party allocations may remain visible in system
        monitors, but IMAGE_GEN-owned module parameters are moved to CPU and allocator
        caches are explicitly released before the status response is produced.
        """
        started = time.perf_counter()
        lease_invalidation = self.invalidate_composition_execution_lease("model_cache_cleared")
        cached_entries = len(self._loaded_model_cache)
        previous = self.resident_model_status()
        cuda_before = dict(previous.get("cuda_memory") or {})
        released_components: list[str] = []
        directly_released_components: list[str] = []
        placement_errors: list[dict[str, str]] = []

        loaded_objects: list[Any] = list(self._loaded_model_cache.values())
        if self.last_loaded_model is not None:
            loaded_objects.append(self.last_loaded_model)
        seen: set[int] = set()
        for loaded in loaded_objects:
            if id(loaded) in seen:
                continue
            seen.add(id(loaded))
            components = getattr(loaded, "components", None)
            if components is None:
                continue
            for name, module in self._runtime_component_entries(components):
                if not callable(getattr(module, "to", None)):
                    continue
                if not move_components_to_cpu:
                    directly_released_components.append(name)
                    continue
                try:
                    module.to(device=torch.device("cpu"))
                    released_components.append(name)
                except Exception as exc:
                    placement_errors.append(
                        {"component": name, "error_type": type(exc).__name__, "error": str(exc)}
                    )

        self._loaded_model_cache.clear()
        self.last_loaded_model = None
        self.lora_runtime_manager.reset()
        gc.collect()
        cuda_cleanup_errors: list[str] = []
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                cuda_cleanup_errors.append(f"synchronize: {type(exc).__name__}: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cuda_cleanup_errors.append(f"empty_cache: {type(exc).__name__}: {exc}")
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception as exc:
                    cuda_cleanup_errors.append(f"ipc_collect: {type(exc).__name__}: {exc}")

        after = self.resident_model_status()
        return {
            "cached_entries_released": cached_entries,
            "previous_model_path": previous.get("model_path"),
            "components_moved_to_cpu": sorted(set(released_components)),
            "components_released_without_cpu_stage": sorted(set(directly_released_components)),
            "move_components_to_cpu": bool(move_components_to_cpu),
            "component_release_errors": placement_errors,
            "cuda_cleanup_errors": cuda_cleanup_errors,
            "cuda_memory_before": cuda_before,
            "cuda_memory_after": dict(after.get("cuda_memory") or {}),
            "note": (
                f"System GPU monitors may still show the CUDA context or non-{PRODUCT_NAME} allocations; "
                f"allocated_bytes is the authoritative {PRODUCT_NAME}/PyTorch tensor allocation value."
            ),
            "lease_invalidation": lease_invalidation,
            "unload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def resident_model_status(self) -> dict[str, Any]:
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        report = getattr(getattr(loaded, "load_plan", None), "report", None)
        model_path = str(getattr(report, "model_path", "") or "")
        component_devices: dict[str, str] = {}
        if components is not None:
            for name, module in self._runtime_component_entries(components):
                device = "unknown"
                try:
                    parameter = next(module.parameters())
                    device = str(parameter.device)
                except (StopIteration, AttributeError, TypeError):
                    device = str(getattr(module, "device", "unknown"))
                component_devices[name] = device
        known_component_devices = [
            value for value in component_devices.values() if value and value != "unknown"
        ]
        gpu_loaded = bool(known_component_devices) and all(
            value.startswith("cuda") for value in known_component_devices
        )
        cpu_loaded = any(value.startswith("cpu") for value in known_component_devices)
        denoiser_component_name = "transformer" if str(getattr(components, "denoiser_kind", "unet") or "unet").strip().lower() == "transformer" else "unet"
        denoiser_device = str(component_devices.get(denoiser_component_name) or "unknown")
        denoiser_gpu_ready = denoiser_device.startswith("cuda")
        cuda_memory = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
            "device_name": None,
        }
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                cuda_memory = {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                }
            except Exception:
                pass
        resident = loaded is not None and bool(self._loaded_model_cache)
        architecture = str(getattr(components, "architecture", "") or "") if components is not None else ""
        staged_runtime = architecture.strip().lower() in {"sd3", "sd3.x", "stable-diffusion-3.x"}
        generation_ready = bool(
            resident
            and known_component_devices
            and all(value != "unknown" and not value.startswith("meta") for value in known_component_devices)
        )
        lora_manager = getattr(self, "lora_runtime_manager", None)
        adapter_dirty = bool(
            getattr(lora_manager, "_loaded_adapters", {})
            or getattr(lora_manager, "_active_signature", ())
        )
        resident_inventory = build_resident_component_inventory(
            loaded, adapter_state_dirty=adapter_dirty
        ) if loaded is not None else {}
        return {
            "resident": resident,
            "architecture": architecture,
            "staged_runtime": staged_runtime,
            "generation_ready": generation_ready,
            "model_path": model_path or None,
            "model_identity": str(getattr(components, "model_identity", "") or "") if components is not None else "",
            "composition_sha256": str(getattr(components, "composition_sha256", "") or "") if components is not None else "",
            "composition_identity_version": str(getattr(components, "composition_identity_version", "") or "") if components is not None else "",
            "composition_contract": dict(getattr(components, "composition_contract", {}) or {}) if components is not None else {},
            "component_sources": {
                str(role): dict(source)
                for role, source in dict(getattr(components, "component_sources", {}) or {}).items()
            } if components is not None else {},
            "composition_projection": dict(getattr(components, "composition_projection", {}) or {}) if components is not None else {},
            "resident_component_inventory": {
                role: entry.public_dict() for role, entry in resident_inventory.items()
            },
            "component_transition_report": public_transition_report(
                getattr(components, "component_transition_report", {}) if components is not None else {}
            ),
            "runtime_component_source_plan": dict(
                getattr(components, "runtime_component_source_plan", {}) if components is not None else {}
            ),
            "composition_execution_lease": self.composition_lease_status(),
            "advanced_model_composition_sha256": str(getattr(components, "advanced_model_composition_sha256", "") or "") if components is not None else "",
            "runtime_load_variant": dict(getattr(components, "runtime_load_variant", {}) or {}) if components is not None else {},
            "runtime_load_variant_fingerprint": str(getattr(components, "runtime_load_variant_fingerprint", "") or "") if components is not None else "",
            "runtime_effective_load_variant": dict(getattr(components, "runtime_effective_load_variant", {}) or {}) if components is not None else {},
            "runtime_effective_load_variant_fingerprint": str(getattr(components, "runtime_effective_load_variant_fingerprint", "") or "") if components is not None else "",
            "runtime_checkpoint_identity": dict(getattr(components, "runtime_checkpoint_identity", {}) or {}) if components is not None else {},
            "cache_entries": len(self._loaded_model_cache),
            "cpu_loaded": cpu_loaded,
            "gpu_loaded": gpu_loaded,
            "denoiser_component": denoiser_component_name,
            "denoiser_device": denoiser_device,
            "denoiser_gpu_ready": denoiser_gpu_ready,
            "hot_gpu_ready": bool(denoiser_gpu_ready and not staged_runtime),
            "component_devices": component_devices,
            "cuda_memory": cuda_memory,
        }

    def apply_resident_retention(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Keep the selected checkpoint components resident between jobs."""
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        if components is None:
            return {"applied": False, "reason": "no cached model"}
        values = dict(settings or {})
        architecture = str(getattr(components, "architecture", "") or "").strip().lower()
        staged_sd3 = architecture in {"sd3", "sd3.x", "stable-diffusion-3.x"}
        checkpoint_retain = bool(values.get("memory_retain_checkpoint_between_jobs", True))
        vae_retain = bool(values.get("memory_retain_vae_between_jobs", True))
        text_retain = bool(values.get("model_runtime_retain_text_encoder_between_jobs", True))
        memory_policy = normalize_policy(values.get("memory_policy"))
        advanced_models_enabled = bool(values.get("advanced_models_enabled"))
        retain = {}
        for name, _module in self._runtime_component_entries(components):
            if name in {"unet", "transformer"}:
                retain[name] = checkpoint_retain
            elif name == "vae":
                retain[name] = vae_retain
            elif name.startswith("text_encoder"):
                retain[name] = text_retain
        retention_policy = str(values.get("model_runtime_retention_device") or "cuda_preferred").strip().lower()
        execution_device, fallback_reason = self._resolve_execution_device(values)
        staged_advanced = bool(advanced_models_enabled and memory_policy != "high_vram")
        staged_low_vram = memory_policy in {"low_vram", "cpu_fallback"}
        if staged_sd3:
            # SD3 is generation-qualified with component-at-a-time CUDA residency.
            # Keep hydrated modules cached on CPU between jobs and let the memory
            # lifecycle stage only the component needed by the active phase.
            target_device = torch.device("cpu")
            retention_policy = "staged_cpu"
        elif staged_advanced:
            # Advanced Models composes independently selected components, often
            # from several digital checkpoint donors. Unless high-VRAM mode is
            # explicitly requested, keeping every retained component on CUDA here
            # defeats the stage memory manager and can freeze low-memory systems
            # before generation starts. Keep the selected modules hydrated on CPU;
            # leases will promote only the conditioning/denoising/decode working set.
            target_device = torch.device("cpu")
            retention_policy = "advanced_staged_cpu"
        elif staged_low_vram:
            target_device = torch.device("cpu")
            retention_policy = "low_vram_staged_cpu"
        elif retention_policy == "cpu":
            target_device = torch.device("cpu")
        elif retention_policy == "cuda":
            if not torch.cuda.is_available():
                target_device = torch.device("cpu")
                fallback_reason = "GPU retention was requested, but CUDA is unavailable; retained on CPU."
            else:
                target_device = torch.device("cuda")
        else:
            target_device = execution_device
        moves: list[dict[str, Any]] = []
        for name, module in self._runtime_component_entries(components):
            keep_on_target = bool(retain.get(name, False))
            if not hasattr(module, "to"):
                continue
            target = target_device if keep_on_target else torch.device("cpu")
            before = "unknown"
            try:
                before = str(next(module.parameters()).device)
            except (StopIteration, AttributeError, TypeError):
                pass
            if before == str(target) or (before.startswith("cuda") and target.type == "cuda"):
                continue
            module.to(target)
            moves.append({"component": name, "from": before, "to": str(target)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "applied": True,
            "retain": retain,
            "retention_device": str(target_device),
            "staged_runtime": staged_sd3,
            "advanced_staged_runtime": staged_advanced,
            "memory_policy": memory_policy,
            "execution_device": str(execution_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_fallback_reason": fallback_reason,
            "moves": moves,
            "status": self.resident_model_status(),
        }

    def apply_hot_residency(
        self,
        settings: dict[str, Any] | None = None,
        *,
        reason: str = "hot_policy",
    ) -> dict[str, Any]:
        """Establish the reusable working set for persistent Hot residency.

        Hot is deliberately architecture-aware. SD3/SD3.5 and constrained
        low-VRAM/advanced compositions remain hydrated but staged on CPU, while
        compatible SD1/SD2/SDXL runtimes keep the expensive denoiser CUDA-ready.
        Request-scoped state is not consulted or retained by this operation.
        """

        started = time.perf_counter()
        loaded = self.last_loaded_model
        components = getattr(loaded, "components", None)
        if components is None:
            return {
                "applied": False,
                "reason": "no cached model",
                "residency_action": "forced_release",
                "effective_hot_state": "empty",
                "moves": [],
                "kept": [],
                "target_devices": {},
                "promotion_time_ms": 0.0,
                "hot_residency_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "status": self.resident_model_status(),
            }

        values = dict(settings or {})
        architecture = str(getattr(components, "architecture", "") or "").strip().lower()
        staged_sd3 = architecture in {"sd3", "sd3.x", "stable-diffusion-3.x"}
        memory_policy = normalize_policy(values.get("memory_policy"))
        advanced_models_enabled = bool(values.get("advanced_models_enabled"))
        staged_advanced = bool(advanced_models_enabled and memory_policy != "high_vram")
        staged_low_vram = memory_policy in {"low_vram", "cpu_fallback"}
        retention_policy = str(values.get("model_runtime_retention_device") or "cuda_preferred").strip().lower()
        execution_device, fallback_reason = self._resolve_execution_device(values)
        cuda_available = bool(torch.cuda.is_available())
        cpu_retention_forced = retention_policy == "cpu"
        staged_for_safety = bool(
            staged_sd3
            or staged_advanced
            or staged_low_vram
            or cpu_retention_forced
            or execution_device.type != "cuda"
            or not cuda_available
        )

        vae_retain = bool(values.get("memory_retain_vae_between_jobs", True))
        text_retain = bool(values.get("model_runtime_retain_text_encoder_between_jobs", True))
        high_vram = memory_policy == "high_vram"
        target_devices: dict[str, str] = {}
        moves: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        promotion_time_ms = 0.0
        degradation_reason: str | None = None
        staging_reason: str | None = None
        promotion_error: dict[str, str] | None = None
        if staged_sd3:
            staging_reason = "architecture_staged_runtime"
        elif staged_advanced:
            staging_reason = "advanced_model_staging"
        elif staged_low_vram:
            staging_reason = f"memory_policy_{memory_policy}"
        elif cpu_retention_forced:
            staging_reason = "cpu_retention_policy"
        elif not cuda_available:
            staging_reason = "cuda_unavailable"
        elif execution_device.type != "cuda":
            staging_reason = str(fallback_reason or "execution_device_cpu")

        def _stage_all_components_to_cpu(*, reason: str) -> None:
            nonlocal degradation_reason, staging_reason
            degradation_reason = str(reason)
            staging_reason = str(reason)
            for component_name, component in self._runtime_component_entries(components):
                if not callable(getattr(component, "to", None)):
                    continue
                before_device = "unknown"
                try:
                    before_device = str(next(component.parameters()).device)
                except (StopIteration, AttributeError, TypeError):
                    before_device = str(getattr(component, "device", "unknown"))
                target_devices[component_name] = "cpu"
                if before_device.startswith("cpu"):
                    continue
                try:
                    move_started = time.perf_counter()
                    component.to(torch.device("cpu"))
                    moves.append({
                        "component": component_name,
                        "from": before_device,
                        "to": "cpu",
                        "reason": reason,
                        "move_time_ms": round((time.perf_counter() - move_started) * 1000.0, 3),
                    })
                except Exception as cleanup_exc:
                    moves.append({
                        "component": component_name,
                        "from": before_device,
                        "to": "cpu",
                        "reason": reason,
                        "error_type": type(cleanup_exc).__name__,
                        "error": str(cleanup_exc),
                    })
            if cuda_available:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        for name, module in self._runtime_component_entries(components):
            if not callable(getattr(module, "to", None)):
                continue
            is_denoiser = name in {"unet", "transformer"}
            if staged_for_safety:
                target = torch.device("cpu")
                target_reason = "architecture_or_memory_staging"
            elif is_denoiser:
                target = torch.device("cuda")
                target_reason = "hot_denoiser_working_set"
            elif high_vram and name == "vae" and vae_retain:
                target = torch.device("cuda")
                target_reason = "high_vram_hot_vae"
            elif high_vram and name.startswith("text_encoder") and text_retain:
                target = torch.device("cuda")
                target_reason = "high_vram_hot_text_encoder"
            else:
                target = torch.device("cpu")
                target_reason = "nonessential_hot_component_staged"

            target_devices[name] = str(target)
            before = "unknown"
            try:
                before = str(next(module.parameters()).device)
            except (StopIteration, AttributeError, TypeError):
                before = str(getattr(module, "device", "unknown"))
            already_target = before == str(target) or (before.startswith("cuda") and target.type == "cuda")
            if already_target:
                kept.append({
                    "component": name,
                    "device": before,
                    "reason": target_reason,
                })
                continue

            move_started = time.perf_counter()
            try:
                module.to(target)
            except Exception as exc:
                if target.type != "cuda" or not is_cuda_oom(exc):
                    raise
                promotion_error = {
                    "component": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _stage_all_components_to_cpu(reason="hot_gpu_promotion_oom")
                staged_for_safety = True
                break
            move_ms = (time.perf_counter() - move_started) * 1000.0
            if target.type == "cuda":
                promotion_time_ms += move_ms
            moves.append(
                {
                    "component": name,
                    "from": before,
                    "to": str(target),
                    "reason": target_reason,
                    "move_time_ms": round(move_ms, 3),
                }
            )

        if cuda_available:
            torch.cuda.empty_cache()

        status = self.resident_model_status()
        hot_gpu_ready = bool(status.get("hot_gpu_ready"))
        effective_hot_state = "hot_staged" if staged_for_safety or not hot_gpu_ready else "hot_gpu"
        if effective_hot_state == "hot_staged":
            residency_action = "hot_staged_hold"
            if not staging_reason:
                staging_reason = "hot_gpu_not_ready"
        elif any(str(move.get("to") or "").startswith("cuda") for move in moves):
            residency_action = "hot_restore"
        else:
            residency_action = "hot_hold"

        return {
            "applied": True,
            "reason": str(reason or "hot_policy"),
            "residency_action": residency_action,
            "effective_hot_state": effective_hot_state,
            "architecture": architecture,
            "memory_policy": memory_policy,
            "retention_device_policy": retention_policy,
            "execution_device": str(execution_device),
            "cuda_available": cuda_available,
            "cpu_fallback_reason": fallback_reason,
            "degradation_reason": degradation_reason,
            "staging_reason": staging_reason,
            "promotion_error": promotion_error,
            "staged_runtime": staged_sd3,
            "advanced_staged_runtime": staged_advanced,
            "low_vram_staged_runtime": staged_low_vram,
            "cpu_retention_forced": cpu_retention_forced,
            "moves": moves,
            "kept": kept,
            "target_devices": target_devices,
            "promotion_time_ms": round(promotion_time_ms, 3),
            "hot_residency_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "status": status,
        }

    def preload_model(
        self,
        model_path: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load and retain checkpoint components without running a generation."""
        trace_started = time.perf_counter()
        trace: dict[str, Any] = {
            "schema_version": 1,
            "kind": "preload_model",
            "stages": [],
        }

        def record_stage(name: str, started: float, **details: Any) -> float:
            elapsed = round((time.perf_counter() - started) * 1000.0, 3)
            item = {"name": str(name), "elapsed_ms": elapsed}
            if details:
                item.update(details)
            trace["stages"].append(item)
            return elapsed

        stage_started = time.perf_counter()
        preload_extras = dict(extras or {})
        preload_extras["model_path"] = str(model_path)
        trace["load_variant_before_preflight"] = {
            "payload": model_load_variant_payload(preload_extras),
            "fingerprint": model_load_variant_fingerprint(preload_extras),
        }
        record_stage("copy_preload_extras", stage_started)

        stage_started = time.perf_counter()
        defaults = dict(self.project_context.generation_defaults() or {})
        defaults.update({
            "positive_prompt": defaults.get("positive_prompt") or "warmup",
            "negative_prompt": defaults.get("negative_prompt") or "",
            "model_path": str(model_path),
            "save_images": False,
        })
        record_stage("generation_defaults", stage_started)

        stage_started = time.perf_counter()
        request, payload_extras = payload_to_generation_request(defaults)
        payload_extras.update(preload_extras)
        record_stage("payload_to_generation_request", stage_started)

        stage_started = time.perf_counter()
        effective_config_fn = getattr(self.project_context, "effective_config", None)
        effective_config = effective_config_fn() if callable(effective_config_fn) else {
            "project_root": str(getattr(self.project_context, "project_root", "."))
        }
        record_stage("effective_config", stage_started)

        stage_started = time.perf_counter()
        session = self.diagnostics_system.start(
            request,
            effective_config=effective_config,
            request_extras=payload_extras,
        )
        record_stage("diagnostics_start", stage_started)
        started = time.perf_counter()
        try:
            stage_started = time.perf_counter()
            request.device = str(self.device)
            self._configure_runtime_state(payload_extras, session)
            record_stage("configure_runtime_state", stage_started)

            # Model activation must apply architecture/profile mutations before registry
            # resolution. SDXL preflight may change the sampler/scheduler and clears
            # descriptors that were resolved for the previous/default request values.
            # Running it here keeps WebUI preload ordering identical to generation.
            stage_started = time.perf_counter()
            self._apply_sdxl_runtime_preflight(request, payload_extras)
            record_stage("sdxl_runtime_preflight", stage_started)

            stage_started = time.perf_counter()
            self._apply_sd3_runtime_preflight(request, payload_extras)
            record_stage(
                "sd3_runtime_preflight",
                stage_started,
                resolved_profile=str(payload_extras.get("sd3_runtime_profile_override") or ""),
            )
            sd3_preflight_trace = dict(payload_extras.get("sd3_preflight_trace") or {})
            if sd3_preflight_trace:
                trace["sd3_preflight_trace"] = sd3_preflight_trace
            trace["load_variant_after_preflight"] = {
                "payload": model_load_variant_payload(payload_extras),
                "fingerprint": model_load_variant_fingerprint(payload_extras),
            }

            stage_started = time.perf_counter()
            request, payload_extras = self.registry_system.apply_resolution(request, payload_extras)
            record_stage("registry_resolution", stage_started)

            stage_started = time.perf_counter()
            self._build_pipeline(request, payload_extras, session)
            record_stage("build_pipeline", stage_started)
            pipeline_trace = dict(payload_extras.get("pipeline_build_trace") or {})
            if pipeline_trace:
                trace["pipeline_build_trace"] = pipeline_trace

            stage_started = time.perf_counter()
            self.diagnostics_system.complete(session)
            record_stage("diagnostics_complete", stage_started)

            stage_started = time.perf_counter()
            status = self.resident_model_status()
            record_stage("resident_status_after", stage_started)
            status.update({
                "preload_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "model_provenance": dict(payload_extras.get("model_provenance") or {}),
            })
            trace["total_ms"] = round((time.perf_counter() - trace_started) * 1000.0, 3)
            status["preload_trace"] = trace
            return status
        except Exception as exc:
            trace["failed"] = {"error_type": type(exc).__name__, "error": str(exc)}
            trace["total_ms"] = round((time.perf_counter() - trace_started) * 1000.0, 3)
            raise self.diagnostics_system.fail_unassigned(
                session, exc, system="model_loading", operation="preload_model"
            ) from exc
        finally:
            self.reset_runtime_state()

    def _resolve_execution_device(self, extras: dict[str, Any]) -> tuple[torch.device, str | None]:
        policy = str(extras.get("model_runtime_execution_device") or "cuda_preferred").strip().lower()
        if policy == "cpu":
            return torch.device("cpu"), "CPU execution was explicitly selected."
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        if policy == "cuda_required":
            raise RuntimeError("CUDA execution is required, but torch.cuda.is_available() is false in this worker.")
        return torch.device("cpu"), "CUDA is unavailable in the model runtime; CPU fallback was activated."

    def _place_loaded_components(
        self,
        loaded: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        settings: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        components = getattr(loaded, "components", None)
        if components is None:
            return []
        reports: list[dict[str, Any]] = []
        architecture = str(getattr(components, "architecture", "") or "").strip().lower()
        values = dict(settings or {})
        memory_policy = normalize_policy(values.get("memory_policy"))
        hot_requested = normalize_model_residency_mode(values.get("model_residency_mode")) == MODEL_RESIDENCY_MODE_HOT
        advanced_staged = bool(values.get("advanced_models_enabled") and memory_policy != "high_vram")
        hot_cuda_working_set = bool(
            hot_requested
            and device.type == "cuda"
            and architecture not in {"sd3.x", "sd3", "stable-diffusion-3.x"}
            and memory_policy not in {"low_vram", "cpu_fallback"}
            and not advanced_staged
            and str(values.get("model_runtime_retention_device") or "cuda_preferred").strip().lower() != "cpu"
        )
        sequential_cpu_first = (
            architecture in {"sdxl", "sd3.x", "sd3", "stable-diffusion-3.x"}
            and device.type == "cuda"
        )
        denoiser_kind = str(getattr(components, "denoiser_kind", "unet") or "unet").strip().lower()
        denoiser_name = "unet" if denoiser_kind == "unet" else "denoiser"
        component_names = (
            denoiser_name,
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        )
        seen_modules: set[int] = set()
        for name in component_names:
            module = getattr(components, name, None)
            if module is None or id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            preserve_hot_denoiser = bool(
                sequential_cpu_first
                and hot_cuda_working_set
                and name in {"unet", "denoiser"}
            )
            target_device = (
                device
                if preserve_hot_denoiser
                else torch.device("cpu")
                if sequential_cpu_first
                else device
            )
            target_dtype = dtype
            if name == "vae" and bool(getattr(components, "vae_force_upcast", False)):
                target_dtype = torch.float32
            reports.append(
                place_component(
                    module,
                    device=target_device,
                    dtype=target_dtype,
                    owner=(
                        "Txt2ImgRunner.hot_cached_denoiser"
                        if preserve_hot_denoiser
                        else "Txt2ImgRunner.sdxl_cached_cpu_first"
                        if sequential_cpu_first and architecture == "sdxl"
                        else "Txt2ImgRunner.sd3_sequential_cpu_first"
                        if sequential_cpu_first
                        else "Txt2ImgRunner.execution_promotion"
                    ),
                    component_name=name,
                ).to_dict()
            )
        return reports
