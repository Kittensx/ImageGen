from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import torch

from modules.config_options import ConfigOptions
from modules.asset_discovery import resolve_nested_asset
from modules.checkpoint_inspector import CheckpointInspector, CheckpointReport
from modules.config_resolver import ConfigResolver, ResolvedConfigs
from modules.state_dict_mapper import StateDictMapper, MappedStateDict
from modules.registry import (
    AssetRegistry,
    ComponentRegistryRefresher,
    ComponentSnapshotRecord,
    SafetensorsComponentSnapshotter,
)
from modules.component_builder import ComponentBuilder, BuiltComponents
from modules.project_context import ProjectContext
from modules.sd2_model_contract import SD2ResolvedModelContract, resolve_sd2_model_contract
from modules.sdxl_model_contract import SDXLResolvedModelContract, resolve_sdxl_model_contract
from modules.sd3_model_contract import SD3ResolvedModelContract, resolve_sd3_model_contract
from modules.sd3_component_sources import prepare_sd3_text_encoder_states
from image_gen.systems.validation.capabilities import capability_for


@dataclass
class LoadPlan:
    report: CheckpointReport
    configs: ResolvedConfigs
    mapped_state: MappedStateDict
    sd2_contract: SD2ResolvedModelContract | None = None
    sdxl_contract: SDXLResolvedModelContract | None = None
    sd3_contract: SD3ResolvedModelContract | None = None
    component_snapshots: tuple[ComponentSnapshotRecord, ...] = ()


class LoadModel(ConfigOptions):
    """
    Phase-1 orchestrator:
    - inspect checkpoint
    - resolve local configs
    - split state dict by component
    - register and log asset metadata
    """

    def __init__(
        self,
        project_context: ProjectContext | None = None,
        asset_registry: AssetRegistry | None = None,
    ):
        super().__init__(project_context=project_context)
        self.inspector = CheckpointInspector()
        self.mapper = StateDictMapper()
        self.config_resolver = ConfigResolver(
            local_config_root=self.local_config_dir
        )
        self.asset_registry = asset_registry or AssetRegistry(self.registry_db_path)
        self.component_snapshotter = SafetensorsComponentSnapshotter(self.mapper)
        self.component_registry_refresher = ComponentRegistryRefresher(
            self.context,
            registry=self.asset_registry,
            inspector=self.inspector,
            snapshotter=self.component_snapshotter,
        )
        self._last_component_registry_refresh: dict[str, object] = {}

    def build_components_from_plan(
        self,
        plan: LoadPlan,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        request_extras: dict[str, Any] | None = None,
    ) -> BuiltComponents:
        requested_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cpu_first_sdxl = bool(plan.sdxl_contract is not None and requested_device.type == "cuda")
        cpu_first_sd3 = bool(plan.sd3_contract is not None)
        cpu_first_hydration = bool(cpu_first_sdxl or cpu_first_sd3)
        # Preserve the qualified SDXL CPU-first expression exactly, then layer
        # the SD3 transformer rule on top. This avoids changing established
        # SDXL behavior/contracts while SD3 remains CPU-first in Phase 4.
        build_device = torch.device("cpu") if cpu_first_sdxl else requested_device
        if cpu_first_sd3:
            build_device = torch.device("cpu")
        sd3_text_encoder_sources: dict[str, Any] = {}
        if plan.sd3_contract is not None:
            sd3_text_encoder_sources = prepare_sd3_text_encoder_states(
                plan,
                context=self.context,
                request_extras=request_extras,
            )
        builder = ComponentBuilder(
            device=str(build_device),
            dtype=dtype,
            defer_attention_configuration=cpu_first_sdxl,
        )
        built = builder.build_components(
            configs=plan.configs,
            mapped_state=plan.mapped_state,
        )
        built.cpu_first_hydration = cpu_first_sdxl
        if cpu_first_sd3:
            built.cpu_first_hydration = True
        built.runtime_target_device = str(requested_device)
        if sd3_text_encoder_sources:
            setattr(built, "sd3_text_encoder_sources", dict(sd3_text_encoder_sources))
        return built
        
    def debug_print_components(self, built) -> None:
        for result in [built.unet_result, built.text_encoder_result, getattr(built, "text_encoder_2_result", None)]:
            if result is None:
                continue
            print(f"=== Component: {result.name} ===")
            print(f"Success: {result.success}")
            print(f"Loaded keys: {result.loaded_keys}")
            print(f"Missing keys: {len(result.missing_keys)}")
            print(f"Unexpected keys: {len(result.unexpected_keys)}")
            if result.error:
                print(f"Error: {result.error}")

        print("=== Component: vae ===")
        print(f"Success: {built.vae_result.success}")
        print(f"Encoder channels: {built.vae_result.encoder_channels}")
        print(f"Decoder specs: {built.vae_result.decoder_specs}")
        print(f"Missing keys: {len(built.vae_result.missing_keys)}")
        print(f"Unexpected keys: {len(built.vae_result.unexpected_keys)}")
        if built.vae_result.error:
            print(f"Error: {built.vae_result.error}")


    def debug_print_component_key_samples(self, built, limit: int = 40) -> None:
        for result in [built.unet_result, built.text_encoder_result, getattr(built, "text_encoder_2_result", None)]:
            if result is None:
                continue
            print(f"\n=== Key samples for {result.name} ===")
            if result.missing_keys:
                print("Missing:")
                for k in result.missing_keys[:limit]:
                    print(f"  {k}")
            if result.unexpected_keys:
                print("Unexpected:")
                for k in result.unexpected_keys[:limit]:
                    print(f"  {k}")

        print("\n=== Key samples for vae ===")
        if built.vae_result.missing_keys:
            print("Missing:")
            for k in built.vae_result.missing_keys[:limit]:
                print(f"  {k}")
        if built.vae_result.unexpected_keys:
            print("Unexpected:")
            for k in built.vae_result.unexpected_keys[:limit]:
                print(f"  {k}")
                
    
    def get_managed_roots(self) -> dict[str, str]:
        return {
            "CheckPoints": self.checkpoints_dir,
            "Lora": self.lora_dir,
            "VAE": self.vae_dir,
            "VAE_approx": self.vae_approx_dir,
            "BLIP": self.blip_dir,
            "Codeformer": self.codeformer_dir,
            "ESRGAN": self.esrgan_dir,
            "GFPGAN": self.gfpgan_dir,
            "RealESRGAN": self.realesrgan_dir,
            "ControlNet": self.controlnet_dir,
            "Embeddings": self.embeddings_dir,
            "Hypernetworks": self.hypernetworks_dir,
        }

    def _ensure_component_snapshots(
        self,
        *,
        asset_id: int,
        asset_quick_fingerprint: str | None,
        checkpoint_path: str,
        report: CheckpointReport,
    ) -> tuple[ComponentSnapshotRecord, ...]:
        """Ensure a loaded Safetensors checkpoint has a complete registry snapshot.

        Registry maintenance is now delegated to :class:`ComponentRegistryRefresher`.
        A file is not considered cache-complete merely because *some* snapshots exist:
        the current header-derived component role set must be represented and bound to
        the file's current quick fingerprint. This means loading a new or previously
        partial asset automatically repairs its registry entry without requiring a
        separate full-library inventory run.
        """
        if Path(checkpoint_path).suffix.lower() != ".safetensors":
            self._last_component_registry_refresh = {
                "status": "skipped_non_safetensors",
                "changed": False,
                "path": str(checkpoint_path),
            }
            return ()

        asset = self.asset_registry.get_asset_by_id(asset_id)
        if asset is None:
            raise RuntimeError(f"Loaded asset registry row disappeared before component refresh: asset_id={asset_id}")

        records, evidence = self.component_registry_refresher.ensure_checkpoint(
            asset=asset,
            checkpoint_path=checkpoint_path,
            report=report,
            source="model_load",
        )
        self._last_component_registry_refresh = dict(evidence)
        return tuple(records)

    def _infer_asset_type_from_report(self, report: CheckpointReport) -> str:
        if report.checkpoint_kind == "lora":
            return "lora"
        if report.checkpoint_kind == "full":
            return "checkpoint"
        if report.has_vae and not report.has_unet and not report.has_text_encoder:
            return "vae"
        return "unknown"

    def prepare_load_plan(
        self,
        checkpoint_path: str | None = None,
        *,
        require_generation_support: bool = True,
        explicit_sd2_runtime_profile: str | None = None,
        explicit_sdxl_runtime_profile: str | None = None,
        explicit_sd3_runtime_profile: str | None = None,
    ) -> LoadPlan:
        checkpoint_path = checkpoint_path if checkpoint_path is not None else self.MODEL_PATH
        if not checkpoint_path:
            raise ValueError(
                "No checkpoint path was supplied and defaults.model_path is not configured."
            )
        requested_checkpoint = str(checkpoint_path)
        direct_checkpoint = Path(requested_checkpoint).expanduser()
        if not direct_checkpoint.is_absolute():
            direct_checkpoint = self.context.resolve_project_path(direct_checkpoint)
        if direct_checkpoint.is_file():
            checkpoint_path = str(direct_checkpoint.resolve())
        else:
            nested_checkpoint = resolve_nested_asset(
                self.checkpoints_dir,
                requested_checkpoint,
                extensions={".safetensors", ".ckpt", ".pt"},
            )
            checkpoint_path = str(nested_checkpoint) if nested_checkpoint is not None else os.path.abspath(requested_checkpoint)

        start_time = time.perf_counter()
        asset_id = None

        try:
            managed_roots = self.get_managed_roots()
            library_root, managed_category, path_kind = self.asset_registry.classify_path(
                checkpoint_path,
                managed_roots=managed_roots,
            )

            asset_record = self.asset_registry.register_file(
                checkpoint_path,
                library_root=library_root,
                managed_category=managed_category,
                path_kind=path_kind,
            )
            asset_id = asset_record.id

            report = self.inspector.inspect(checkpoint_path)
            self.asset_registry.update_asset_sha256(asset_id, report.sha256)
            component_snapshots = self._ensure_component_snapshots(
                asset_id=asset_id,
                asset_quick_fingerprint=asset_record.quick_fingerprint,
                checkpoint_path=checkpoint_path,
                report=report,
            )

            inspection_payload = {
                "asset_type": self._infer_asset_type_from_report(report),
                "format_type": Path(checkpoint_path).suffix.lower().lstrip(".") or "other",
                "architecture": report.architecture,
                "checkpoint_kind": report.checkpoint_kind,
                "has_unet": report.has_unet,
                "has_vae": report.has_vae,
                "has_text_encoder": report.has_text_encoder,
                "has_text_encoder_2": report.has_sdxl_text_encoder_2,
                "key_count": report.total_keys,
                "prefix_summary": {"prefixes": report.key_prefixes},
                "example_keys": report.example_keys,
                "dtype_summary": dict(report.dtype_summary),
                "tensor_shape_summary": dict(report.tensor_shape_summary),
                "metadata": {
                    "file_name": report.file_name,
                    "checkpoint_kind": report.checkpoint_kind,
                    "file_size_bytes": report.file_size_bytes,
                    "sha256": report.sha256,
                    "architecture_evidence": list(report.architecture_evidence),
                    "safetensors_metadata": dict(report.safetensors_metadata),
                    "library_root": library_root,
                    "managed_category": managed_category,
                    "path_kind": path_kind,
                },
                "inspector_version": "3",
            }
            self.asset_registry.store_inspection(asset_id, inspection_payload)

            if report.checkpoint_kind != "full":
                raise ValueError(
                    f"Expected a full checkpoint, but got '{report.checkpoint_kind}'."
                )

            capability = capability_for(report.architecture)
            if require_generation_support:
                if not capability.generation_supported:
                    requirements = "; ".join(capability.requirements)
                    raise ValueError(
                        f"Checkpoint architecture {report.architecture!r} is not enabled for generation: "
                        f"{capability.reason} Required work: {requirements}"
                    )
            elif not (capability.validation_supported or capability.generation_supported):
                requirements = "; ".join(capability.requirements)
                raise ValueError(
                    f"Checkpoint architecture {report.architecture!r} is not enabled for validation: "
                    f"{capability.reason} Required work: {requirements}"
                )

            sd2_contract = None
            sdxl_contract = None
            sd3_contract = None
            if report.architecture == "sd2.x":
                configured_profile = explicit_sd2_runtime_profile
                if configured_profile is None:
                    configured_profile = str(
                        ((self.config.get("defaults") or {}).get("sd2_runtime_profile") or "")
                    ).strip() or None
                sd2_contract = resolve_sd2_model_contract(
                    self.context,
                    checkpoint_filename=report.file_name,
                    checkpoint_sha256=report.sha256,
                    checkpoint_prediction_type=report.prediction_type,
                    checkpoint_prediction_source=report.prediction_type_source,
                    explicit_profile_id=configured_profile,
                )
                report.prediction_type = sd2_contract.prediction_type
                report.prediction_type_source = sd2_contract.prediction_type_source
                report.architecture_summary = (
                    f"SD 2.x / {sd2_contract.prediction_type} / {sd2_contract.conditioning_dimension}"
                )
                report.architecture_source = "qualified_sd2_runtime_profile"
                configs = self.config_resolver.resolve_explicit(
                    architecture=report.architecture,
                    root_dir=str(sd2_contract.assets.root),
                    manifest_path=str(sd2_contract.assets.model_index or ""),
                    unet_config_path=str(sd2_contract.assets.unet_config),
                    vae_config_path=str(sd2_contract.assets.vae_config),
                    text_encoder_config_path=str(sd2_contract.assets.text_encoder_config),
                )
            elif report.architecture == "sdxl":
                configured_profile = explicit_sdxl_runtime_profile
                if configured_profile is None:
                    configured_profile = str(
                        ((self.config.get("defaults") or {}).get("sdxl_runtime_profile") or "")
                    ).strip() or None
                sdxl_contract = resolve_sdxl_model_contract(
                    self.context,
                    checkpoint_filename=report.file_name,
                    explicit_profile_id=configured_profile,
                )
                report.prediction_type = sdxl_contract.prediction_type
                report.prediction_type_source = sdxl_contract.prediction_type_source
                report.architecture_summary = (
                    f"SDXL / {sdxl_contract.profile.profile_id} / "
                    f"{sdxl_contract.prediction_type} / 2048"
                )
                report.architecture_source = "qualified_sdxl_runtime_profile"
                configs = sdxl_contract.assets.to_resolved_configs(self.config_resolver)
            elif report.architecture == "sd3.x":
                configured_profile = explicit_sd3_runtime_profile
                if configured_profile is None:
                    configured_profile = str(
                        ((self.config.get("defaults") or {}).get("sd3_runtime_profile") or "")
                    ).strip() or None
                sd3_contract = resolve_sd3_model_contract(
                    self.context,
                    checkpoint_variant=report.architecture_variant,
                    explicit_profile_id=configured_profile,
                )
                report.prediction_type = "not_applicable_flow_match"
                report.prediction_type_source = "sd3_flow_match_contract"
                report.architecture_summary = (
                    f"SD3.x / {sd3_contract.profile.profile_id} / flow_match / "
                    f"{sd3_contract.model_dimension}"
                )
                report.architecture_source = "qualified_sd3_runtime_profile"
                configs = sd3_contract.assets.to_resolved_configs(self.config_resolver)
            else:
                configs = self.config_resolver.resolve(report.architecture)
            state_dict = self.inspector.load_state_dict(checkpoint_path)
            mapped_state = self.mapper.split_checkpoint(state_dict, architecture=report.architecture)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            self.asset_registry.log_load_attempt(
                asset_id=asset_id,
                status="success",
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="unknown",
                load_time_ms=elapsed_ms,
                context={
                    "stage": "prepare_load_plan",
                    "architecture": report.architecture,
                    "path_kind": path_kind,
                    "managed_category": managed_category,
                    "mode": "generation" if require_generation_support else "validation",
                    "component_snapshot_count": len(component_snapshots),
                    "component_snapshot_version": (
                        component_snapshots[0].snapshot_version if component_snapshots else None
                    ),
                    "component_registry_refresh": dict(self._last_component_registry_refresh),
                },
            )

            return LoadPlan(
                report=report,
                configs=configs,
                mapped_state=mapped_state,
                sd2_contract=sd2_contract,
                sdxl_contract=sdxl_contract,
                sd3_contract=sd3_contract,
                component_snapshots=component_snapshots,
            )

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            if asset_id is not None:
                self.asset_registry.log_load_attempt(
                    asset_id=asset_id,
                    status="failed",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    precision="unknown",
                    load_time_ms=elapsed_ms,
                    error_message=str(e),
                    context={
                        "stage": "prepare_load_plan",
                        "checkpoint_path": checkpoint_path,
                    },
                )
            raise

    def prepare_validation_plan(
        self,
        checkpoint_path: str | None = None,
        *,
        explicit_sd2_runtime_profile: str | None = None,
        explicit_sdxl_runtime_profile: str | None = None,
        explicit_sd3_runtime_profile: str | None = None,
    ) -> LoadPlan:
        return self.prepare_load_plan(
            checkpoint_path=checkpoint_path,
            require_generation_support=False,
            explicit_sd2_runtime_profile=explicit_sd2_runtime_profile,
            explicit_sdxl_runtime_profile=explicit_sdxl_runtime_profile,
            explicit_sd3_runtime_profile=explicit_sd3_runtime_profile,
        )

    def debug_print_plan(self, plan: LoadPlan) -> None:
        print("=== Load Plan ===")
        print(f"File: {plan.report.file_name}")
        print(f"Architecture: {plan.report.architecture}")
        print(f"SHA-256: {plan.report.sha256}")
        print(f"Checkpoint kind: {plan.report.checkpoint_kind}")
        print(f"Total keys: {plan.report.total_keys}")
        print(f"UNet keys: {len(plan.mapped_state.unet)}")
        print(f"Transformer keys: {len(plan.mapped_state.transformer)}")
        print(f"VAE keys: {len(plan.mapped_state.vae)}")
        print(f"Text encoder keys: {len(plan.mapped_state.text_encoder)}")
        print(f"Text encoder 2 keys: {len(plan.mapped_state.text_encoder_2)}")
        print(f"Text encoder 3 keys: {len(plan.mapped_state.text_encoder_3)}")
        print(f"Extra keys: {len(plan.mapped_state.extras)}")
        print(f"Config root: {plan.configs.root_dir}")