# prompt_conditioning_adapter.py

# -----------------------------------------------------------------------------
# REGION ATTRIBUTION
#
# REGION syntax and the original regional-conditioning design originate from
# work by GitHub user Konpr:
#   https://github.com/Konpr/whats-/tree/main/new_version3
#
# The original author granted permission to use/adapt the code with credit.
# This adapter/bridge is IMAGE_GEN-specific: it separates REGION extraction from
# the selected prompt parser and routes the resulting regional prompts through
# IMAGE_GEN's parser registry and native conditioning/runtime contracts.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch

from modules.contracts import ConditioningOutput, PromptAdapterProtocol

PromptAdapter = PromptAdapterProtocol
from modules.parser.prompt_parser_class import (
    MulticondLearnedConditioning,
    ScheduledPromptConditioning,
)
from modules.prompt_parsers import (
    CANONICAL_PROMPT_CONTRACT_VERSION,
    PromptParseRequest,
    PromptParseResult,
    PromptParserError,
    PromptParserRegistry,
    default_prompt_parser_registry,
)
from modules.prompt_parsers.semantic_replay import (
    PromptSemanticReplayError,
    build_superhybrid_semantic_record,
    validate_recorded_superhybrid_semantic_record,
)
from modules.prompt_shortcuts import (
    PROMPT_SHORTCUT_CONTRACT_VERSION,
    PromptShortcutError,
    PromptShortcutProfileDescriptor,
    PromptShortcutTranslator,
    default_prompt_shortcut_registry,
    validate_prompt_shortcut_profile,
)
from modules.adapters.local_clip_conditioning_wrapper import LocalCLIPConditioningWrapper
from modules.adapters.sd2_openclip_conditioning import SD2OpenCLIPConditioningRuntime
from modules.adapters.sdxl_conditioning import SDXLConditioningRuntime
from modules.adapters.sd3_conditioning import SD3ConditioningRuntime
from image_gen.systems.guidance import (
    PromptCFGScheduleError,
    finalize_prompt_cfg_payload,
    validate_recorded_prompt_cfg_payload,
)
from image_gen.systems.prompt_expansion import (
    PromptExpansionError,
    expand_superhybrid_prompt_batch,
)
from image_gen.systems.regional_prompting import (
    RegionalPromptError,
    RuntimeRegionSpec,
    build_region_record,
    extract_superhybrid_region_slot,
    validate_recorded_region_record,
)
from modules.pipeline.regional_conditioning import (
    RegionConditioningEntry,
    RegionalConditioningResolver,
)
from modules.txt2img.seed_utils import resolve_seed_sequence

@dataclass
class StepConditioningResolver:
    positive_multicond: MulticondLearnedConditioning
    negative_multicond: MulticondLearnedConditioning
    total_steps: int

    def _select_schedule_cond(
        self,
        schedules: list[ScheduledPromptConditioning],
        step_index: int,
    ) -> Any:
        """
        step_index is expected to be 0-based from the sampler loop.
        Prompt scheduling in your parser is effectively 1-based, so we convert here.
        """
        step_1_based = step_index + 1

        if not schedules:
            raise ValueError("No schedules found for conditioning item.")

        for sched in schedules:
            if step_1_based <= sched.end_at_step:
                return sched.cond

        return schedules[-1].cond

    def _resolve_multicond_for_step(
        self,
        multicond: MulticondLearnedConditioning,
        step_index: int,
        *,
        value_key: str | None = None,
    ) -> torch.Tensor:
        """
        Resolves the active conditioning tensor for a given step.

        Output shape:
          [batch, tokens, dim]  or whatever the underlying text encoder returns
        """
        batch_outputs = []

        for batch_items in multicond.batch:
            item_tensors = []
            item_weights = []

            for item in batch_items:
                schedules = item.schedules

                # Defensive unwrap for legacy nesting:
                # some existing code stores one ScheduledPromptConditioning whose
                # .cond is itself a list[ScheduledPromptConditioning].
                if (
                    len(schedules) == 1
                    and isinstance(getattr(schedules[0], "cond", None), list)
                    and schedules[0].cond
                    and hasattr(schedules[0].cond[0], "end_at_step")
                    and hasattr(schedules[0].cond[0], "cond")
                ):
                    schedules = schedules[0].cond

                resolved_value = self._select_schedule_cond(schedules, step_index)
                if isinstance(resolved_value, dict):
                    selected_key = value_key or "cross_attention"
                    if selected_key not in resolved_value:
                        raise KeyError(
                            f"Structured conditioning is missing required field {selected_key!r}."
                        )
                    cond_tensor = resolved_value[selected_key]
                else:
                    if value_key not in (None, "cross_attention"):
                        raise TypeError(
                            f"Conditioning field {value_key!r} was requested from an unstructured tensor schedule."
                        )
                    cond_tensor = resolved_value

                if not isinstance(cond_tensor, torch.Tensor):
                    raise TypeError(
                        f"Resolved conditioning must be a tensor, got {type(cond_tensor)}"
                    )

                item_tensors.append(cond_tensor)
                weight_at_step = getattr(item, "weight_at_step", None)
                if callable(weight_at_step):
                    item_weights.append(float(weight_at_step(step_index + 1)))
                else:
                    item_weights.append(float(getattr(item, "weight", 1.0)))

            if not item_tensors:
                raise ValueError("No composable conditioning tensors found for batch item.")

            if len(item_tensors) == 1:
                combined = item_tensors[0]
            else:
                weight_tensor = torch.tensor(
                    item_weights,
                    device=item_tensors[0].device,
                    dtype=item_tensors[0].dtype,
                )
                weight_sum = weight_tensor.sum().clamp(min=1e-8)
                weight_tensor = weight_tensor / weight_sum

                stacked = torch.stack(item_tensors, dim=0)
                view_shape = [len(item_tensors)] + [1] * (stacked.ndim - 1)
                combined = (stacked * weight_tensor.view(*view_shape)).sum(dim=0)

            batch_outputs.append(combined)

        return torch.stack(batch_outputs, dim=0)

    def resolve(self, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._resolve_multicond_for_step(
            self.positive_multicond, step_index, value_key="cross_attention"
        )
        uncond = self._resolve_multicond_for_step(
            self.negative_multicond, step_index, value_key="cross_attention"
        )
        return cond, uncond

    def resolve_pooled(self, step_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._resolve_multicond_for_step(
            self.positive_multicond, step_index, value_key="pooled"
        )
        uncond = self._resolve_multicond_for_step(
            self.negative_multicond, step_index, value_key="pooled"
        )
        return cond, uncond


class PromptConditioningAdapter(PromptAdapter):
    """
    Pipeline-facing adapter for:
      prompt text -> parsed schedules -> learned conditioning -> step resolver
    """

    def __init__(
        self,
        hires_steps: Optional[int] = None,
        parser_registry: PromptParserRegistry | None = None,
    ):
        self.hires_steps = hires_steps
        self.parser_registry = parser_registry or default_prompt_parser_registry()
        self.shortcut_registry = default_prompt_shortcut_registry()
        self.shortcut_translator = PromptShortcutTranslator()

    def _get_conditioning_model(self, components, state: Optional[Any]):
        text_encoder = getattr(components, "text_encoder", None)
        tokenizer = getattr(components, "tokenizer", None)


        if text_encoder is not None and tokenizer is not None:
            config = getattr(text_encoder, "config", None)
            hidden_size = int(getattr(config, "hidden_size", 0) or 0)
            hidden_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
            architecture = str(getattr(text_encoder, "_image_gen_architecture", "") or "").strip().lower()
            is_sd3 = architecture in {"sd3", "sd3.x", "sd3.5", "stable-diffusion-3", "stable-diffusion-3.x"}
            if is_sd3:
                text_encoder_2 = getattr(components, "text_encoder_2", None)
                tokenizer_2 = getattr(components, "tokenizer_2", None)
                if text_encoder_2 is None or tokenizer_2 is None:
                    raise ValueError(
                        "SD3 CLIP-only conditioning requires CLIP-G text_encoder_2 and tokenizer_2."
                    )
                runtime_profile = dict(getattr(components, "model_runtime_profile", {}) or {})
                denoiser = getattr(components, "denoiser", None)
                denoiser_config = getattr(denoiser, "config", None)
                component_joint_attention_dim = int(
                    getattr(denoiser_config, "joint_attention_dim", 0) or 0
                )
                joint_attention_dim = component_joint_attention_dim or int(
                    runtime_profile.get(
                        "transformer_joint_attention_dim",
                        runtime_profile.get("joint_attention_dim", 4096),
                    )
                    or 4096
                )
                return SD3ConditioningRuntime(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    text_encoder_3=getattr(components, "text_encoder_3", None),
                    tokenizer_3=getattr(components, "tokenizer_3", None),
                    joint_attention_dim=joint_attention_dim,
                )

            is_sdxl = architecture in {"sdxl", "stable-diffusion-xl", "stable-diffusion-xl-base"}
            if is_sdxl:
                text_encoder_2 = getattr(components, "text_encoder_2", None)
                tokenizer_2 = getattr(components, "tokenizer_2", None)
                if text_encoder_2 is None or tokenizer_2 is None:
                    raise ValueError(
                        "Qualified SDXL conditioning requires text_encoder_2 and tokenizer_2."
                    )
                return SDXLConditioningRuntime(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                )

            is_sd2 = architecture in {"sd2", "sd2.1", "sd2.x", "stable-diffusion-2.x"}
            if is_sd2:
                if hidden_size != 1024 or hidden_layers != 23:
                    raise ValueError(
                        "Qualified SD2 text encoder must use the 23-layer / 1024-wide runtime contract; "
                        f"got layers={hidden_layers}, hidden_size={hidden_size}."
                    )
                return SD2OpenCLIPConditioningRuntime(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                )
            wrapper = LocalCLIPConditioningWrapper(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=next(text_encoder.parameters()).device,
            )
            return wrapper

        if text_encoder is not None and hasattr(text_encoder, "encode"):
            return text_encoder

        if state is not None:
            cfgm = getattr(state, "cfgm", None)
            if cfgm is not None and hasattr(cfgm, "model"):
                model = cfgm.model
                if hasattr(model, "get_learned_conditioning") or hasattr(model, "encode"):
                    return model

        raise AttributeError("Could not find or construct a conditioning model.")

    @staticmethod
    def _parser_selection(request: Any) -> tuple[str, str, dict[str, Any]]:
        explicit = str(getattr(request, "prompt_parser_name", "") or "").strip()
        prompt_parser_kwargs = dict(getattr(request, "prompt_parser_kwargs", {}) or {})
        legacy_kwargs = dict(getattr(request, "parser_kwargs", {}) or {})
        alias = str(
            legacy_kwargs.get("prompt_parser")
            or legacy_kwargs.get("prompt_parser_name")
            or ""
        ).strip()
        if explicit:
            parser_name = explicit
            source = "prompt_parser_name"
        elif alias:
            parser_name = alias
            source = "parser_kwargs_compatibility_alias"
        else:
            parser_name = "legacy"
            source = "default"
        parser_options = dict(legacy_kwargs)
        parser_options.pop("prompt_parser", None)
        parser_options.pop("prompt_parser_name", None)
        parser_options.pop("hires_steps", None)
        parser_options.update(prompt_parser_kwargs)
        return parser_name, source, parser_options

    def _shortcut_profile(self, request: Any, *, parser_id: str) -> PromptShortcutProfileDescriptor:
        snapshot = dict(getattr(request, "prompt_shortcut_profile_snapshot", {}) or {})
        if snapshot:
            profile = PromptShortcutProfileDescriptor.from_dict(
                snapshot,
                builtin=bool(snapshot.get("builtin", False)),
            )
            validation = validate_prompt_shortcut_profile(profile)
            if not validation.valid:
                raise PromptShortcutError(
                    "Embedded prompt shortcut profile snapshot is invalid.",
                    error_kind="invalid_shortcut_profile_snapshot",
                    diagnostics=validation.to_dict(),
                )
        else:
            fallback = (
                "legacy_default"
                if parser_id == "legacy"
                else "parser21_native"
                if parser_id == "parser21"
                else "superhybrid_native"
                if parser_id == "superhybrid"
                else "canonical"
            )
            profile_name = str(getattr(request, "prompt_shortcut_profile_name", "") or fallback)
            if (
                (parser_id in {"parser21", "superhybrid"} and profile_name == "legacy_default")
                or (parser_id == "legacy" and profile_name in {"parser21_native", "superhybrid_native"})
                or (parser_id == "parser21" and profile_name == "superhybrid_native")
                or (parser_id == "superhybrid" and profile_name == "parser21_native")
            ):
                profile_name = fallback
            profile = self.shortcut_registry.get(profile_name)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21", "superhybrid"))
        )
        if not compatible:
            raise PromptShortcutError(
                f"Shortcut profile {profile.profile_id!r} is not compatible with parser {parser_id!r}.",
                error_kind="shortcut_profile_parser_incompatible",
                diagnostics={"profile_id": profile.profile_id, "parser_id": parser_id},
            )
        request.prompt_shortcut_profile_name = profile.profile_id
        request.prompt_shortcut_profile_snapshot = profile.snapshot()
        return profile

    def _parse_prompt(
        self,
        *,
        parser: Any,
        state: Any,
        prompt: str,
        prompt_role: str,
        steps: int,
        hires_steps: Optional[int],
        model: Any,
        parser_options: dict[str, Any],
        width: int,
        height: int,
        seed: int | None,
        recorded_route_plan: dict[str, Any] | None = None,
    ):
        return parser.parse(
            PromptParseRequest(
                raw_prompt=prompt,
                prompt_role=prompt_role,
                steps=steps,
                hires_steps=hires_steps,
                parser_options=parser_options,
                model_context=model,
                shared_state=state,
                width=width,
                height=height,
                seed=seed,
                recorded_route_plan=recorded_route_plan,
            )
        )

    @staticmethod
    def _combine_parse_results(results: list[PromptParseResult], *, prompt_role: str) -> PromptParseResult:
        if not results:
            raise ValueError("At least one prompt parse result is required.")
        if len(results) == 1:
            return results[0]
        first = results[0]
        batches: list[Any] = []
        schedules: list[Any] = []
        flat_list: list[str] = []
        warnings: list[str] = []
        directives_by_slot: list[dict[str, Any]] = []
        canonical_by_slot: list[str] = []
        structures_by_slot: list[dict[str, Any]] = []
        diagnostics_by_slot: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            source = result.conditioning_source
            multicond = source.multicond
            slot_batches = list(getattr(multicond, "batch", []) or [])
            if len(slot_batches) != 1:
                raise PromptParserError(
                    "Per-image prompt parsing expected exactly one conditioning batch item per image slot.",
                    parser_id=result.parser_id,
                    prompt_role=prompt_role,
                    error_kind="invalid_per_image_conditioning_batch",
                    diagnostics={"slot_index": index, "batch_item_count": len(slot_batches)},
                )
            batches.extend(slot_batches)
            schedules.extend(list(getattr(source, "schedules", []) or []))
            flat_list.extend([str(item) for item in list(getattr(source, "flat_list", []) or [])])
            warnings.extend(result.warnings)
            directives_by_slot.append(dict(result.directives or {}))
            canonical_by_slot.append(str(result.canonical_prompt or ""))
            structures_by_slot.append(dict(result.canonical_structure or {}))
            diagnostics_by_slot.append(dict(result.diagnostics or {}))

        cfg_by_slot = [dict(item.get("cfg") or {}) for item in directives_by_slot]
        comparable_cfg = [item for item in cfg_by_slot if item]
        if comparable_cfg and any(item != comparable_cfg[0] for item in comparable_cfg[1:]):
            raise PromptParserError(
                "Per-image SuperHybrid expansion produced different CFG directives across batch slots. Phase 4 requires one shared CFG schedule per sampling batch.",
                parser_id=first.parser_id,
                prompt_role=prompt_role,
                error_kind="per_image_cfg_schedule_mismatch",
                diagnostics={"cfg_directives_by_slot": cfg_by_slot},
            )
        if comparable_cfg and len(comparable_cfg) != len(cfg_by_slot):
            raise PromptParserError(
                "Per-image SuperHybrid expansion produced a CFG directive for only part of the batch. Phase 4 requires one shared CFG schedule per sampling batch.",
                parser_id=first.parser_id,
                prompt_role=prompt_role,
                error_kind="partial_per_image_cfg_schedule",
                diagnostics={"cfg_directives_by_slot": cfg_by_slot},
            )

        multicond_type = type(first.conditioning_source.multicond)
        combined_multicond = multicond_type(shape=(len(batches),), batch=batches)
        combined_source = SimpleNamespace(
            multicond=combined_multicond,
            schedules=schedules,
            flat_list=flat_list,
        )
        directives = {"cfg": comparable_cfg[0]} if comparable_cfg else {}
        return PromptParseResult(
            parser_id=first.parser_id,
            parser_version=first.parser_version,
            parser_contract_version=first.parser_contract_version,
            raw_prompt=str(first.raw_prompt or ""),
            canonical_prompt=canonical_by_slot[0],
            canonical_structure=structures_by_slot[0],
            schedules=schedules,
            conditioning_source=combined_source,
            warnings=warnings,
            diagnostics={
                **dict(first.diagnostics or {}),
                "batch_slot_count": len(results),
                "canonical_prompts_by_slot": canonical_by_slot,
                "canonical_structures_by_slot": structures_by_slot,
                "slot_diagnostics": diagnostics_by_slot,
                "semantic_fingerprints_by_slot": [
                    dict(item.get("semantic_fingerprint") or {}) for item in diagnostics_by_slot
                ],
            },
            directives=directives,
        )

    def _parse_prompt_slots(
        self,
        *,
        parser: Any,
        state: Any,
        prompts: list[str],
        prompt_role: str,
        steps: int,
        hires_steps: Optional[int],
        model: Any,
        parser_options: dict[str, Any],
        width: int,
        height: int,
        seeds: list[int],
        parser_seeds: list[int] | None = None,
        recorded_route_plan: dict[str, Any] | None = None,
    ) -> PromptParseResult:
        if len(prompts) != len(seeds):
            raise ValueError("Prompt slot count must match the resolved image seed count.")
        effective_parser_seeds = list(parser_seeds or seeds)
        if len(effective_parser_seeds) != len(seeds):
            raise ValueError("Parser seed count must match the resolved image seed count.")
        results = [
            self._parse_prompt(
                parser=parser,
                state=state,
                prompt=prompt,
                prompt_role=prompt_role,
                steps=steps,
                hires_steps=hires_steps,
                model=model,
                parser_options={
                    **parser_options,
                    **({"seed": effective_parser_seeds[index]} if parser.descriptor.parser_id == "superhybrid" else {}),
                },
                width=width,
                height=height,
                seed=effective_parser_seeds[index],
                recorded_route_plan=recorded_route_plan,
            )
            for index, prompt in enumerate(prompts)
        ]
        return self._combine_parse_results(results, prompt_role=prompt_role)

    def encode(
        self,
        components,
        request,
        state: Optional[Any] = None,
    ) -> ConditioningOutput:
        cond_model = self._get_conditioning_model(components, state)
        
        
        legacy_parser_kwargs = dict(getattr(request, "parser_kwargs", {}) or {})
        hires_steps = legacy_parser_kwargs.get("hires_steps", self.hires_steps)
        parser_name, parser_selection_source, parser_options = self._parser_selection(request)
        parser = self.parser_registry.get(parser_name)
        descriptor = parser.descriptor
        request.prompt_parser_name = descriptor.parser_id
        request.base_prompt_parser_name = descriptor.parser_id
        shortcut_profile = self._shortcut_profile(request, parser_id=descriptor.parser_id)
        request.base_shortcut_profile_name = shortcut_profile.profile_id
        pass_name = "hires" if bool(getattr(request, "_is_hires_request", False)) else "base"
        expansion_record: dict[str, Any] = {}
        expanded_positive_prompt = str(request.positive_prompt or "")
        expanded_negative_prompt = str(request.negative_prompt or "")
        resolved_seeds = [int(value) for value in list(getattr(request, "resolved_seeds", []) or [])]
        if len(resolved_seeds) != int(request.batch_size):
            resolved_seeds = resolve_seed_sequence(request.seed, request.batch_size)
            request.seed = resolved_seeds[0]
            request.resolved_seeds = list(resolved_seeds)
        expansion_replay_mode = str(
            getattr(request, "prompt_expansion_replay_mode", "reconstruct") or "reconstruct"
        ).strip().lower()
        if expansion_replay_mode not in {"reconstruct", "recorded_exact"}:
            raise PromptParserError(
                "prompt_expansion_replay_mode must be reconstruct or recorded_exact.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="invalid_prompt_expansion_replay_mode",
            )
        expanded_positive_slots = [expanded_positive_prompt for _ in resolved_seeds]
        expanded_negative_slots = [expanded_negative_prompt for _ in resolved_seeds]
        expansion_scope = str(parser_options.get("prompt_expansion_scope", "per_batch") or "per_batch")
        parser_slot_seeds = list(resolved_seeds)
        if descriptor.parser_id == "superhybrid":
            recorded_expansions = dict(
                getattr(request, "prompt_expansion_recorded", {}) or {}
            )
            parser_seed = parser_options.get("seed")
            if parser_seed in (None, "") or int(parser_seed) < 0:
                expansion_selection_seeds = list(resolved_seeds)
            else:
                expansion_selection_seeds = resolve_seed_sequence(int(parser_seed), request.batch_size)
            parser_slot_seeds = (
                list(expansion_selection_seeds)
                if expansion_scope == "per_image"
                else [int(expansion_selection_seeds[0])] * len(resolved_seeds)
            )
            try:
                expansion_record = expand_superhybrid_prompt_batch(
                    expanded_positive_prompt,
                    expanded_negative_prompt,
                    resolved_seeds=resolved_seeds,
                    selection_seeds=expansion_selection_seeds,
                    pass_name=pass_name,
                    parser_version=descriptor.version,
                    scope=expansion_scope,
                    wildcard_directory=str(parser_options.get("wildcard_directory", "wildcards") or "wildcards"),
                    recorded=dict(recorded_expansions.get(pass_name) or {}),
                    replay_mode=expansion_replay_mode,
                )
            except (PromptExpansionError, TypeError, ValueError) as exc:
                raise PromptParserError(
                    f"SuperHybrid prompt expansion failed: {exc}",
                    parser_id=descriptor.parser_id,
                    prompt_role="positive",
                    error_kind="superhybrid_prompt_expansion_failed",
                    diagnostics={
                        "pass": pass_name,
                        "replay_mode": expansion_replay_mode,
                        "scope": expansion_scope,
                    },
                ) from exc
            expanded_positive_slots = [
                str(value or "") for value in list(expansion_record.get("expanded_positive_by_slot") or [])
            ]
            expanded_negative_slots = [
                str(value or "") for value in list(expansion_record.get("expanded_negative_by_slot") or [])
            ]
            if len(expanded_positive_slots) != len(resolved_seeds) or len(expanded_negative_slots) != len(resolved_seeds):
                raise PromptParserError(
                    "SuperHybrid prompt expansion did not produce one prompt pair per resolved image seed.",
                    parser_id=descriptor.parser_id,
                    prompt_role="positive",
                    error_kind="invalid_prompt_expansion_slot_count",
                    diagnostics={
                        "expected": len(resolved_seeds),
                        "positive_slots": len(expanded_positive_slots),
                        "negative_slots": len(expanded_negative_slots),
                    },
                )
            expanded_positive_prompt = expanded_positive_slots[0]
            expanded_negative_prompt = expanded_negative_slots[0]
        elif expansion_replay_mode == "recorded_exact" and getattr(request, "prompt_expansion_recorded", None):
            raise PromptParserError(
                "Recorded SuperHybrid prompt expansion cannot be applied to a different prompt parser.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="prompt_expansion_parser_mismatch",
            )

        request.prompt_expansion_record = dict(expansion_record)
        expansion_pass_records = dict(getattr(request, "prompt_expansion_pass_records", {}) or {})
        if expansion_record:
            expansion_pass_records[pass_name] = dict(expansion_record)
        else:
            expansion_pass_records.pop(pass_name, None)
        request.prompt_expansion_pass_records = expansion_pass_records
        if isinstance(getattr(request, "diagnostics", None), dict):
            if expansion_record:
                request.diagnostics["prompt_expansion"] = dict(expansion_record)
            else:
                request.diagnostics.pop("prompt_expansion", None)

        region_runtime_specs_by_slot: list[list[RuntimeRegionSpec]] = [
            [] for _ in resolved_seeds
        ]
        region_slot_records: list[dict[str, Any]] = [
            {
                "slot_index": index,
                "source_prompt": expanded_positive_slots[index],
                "base_prompt": expanded_positive_slots[index],
                "region_count": 0,
                "regions": [],
            }
            for index in range(len(resolved_seeds))
        ]
        region_replay_mode = str(
            getattr(request, "region_replay_mode", "reconstruct") or "reconstruct"
        ).strip().lower()
        if region_replay_mode not in {"reconstruct", "recorded_exact"}:
            raise PromptParserError(
                "region_replay_mode must be reconstruct or recorded_exact.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="invalid_region_replay_mode",
            )
        try:
            base_region_reference = {}
            if pass_name == "hires":
                if region_replay_mode == "recorded_exact":
                    # Exact replay honors the recorded hires geometry itself.
                    # This preserves older manifests generated before hires
                    # REGION scaling was corrected, while new manifests replay
                    # their corrected geometry exactly.
                    base_region_reference = dict(
                        (getattr(request, "region_recorded", {}) or {}).get("hires") or {}
                    )
                if not base_region_reference:
                    base_region_reference = dict(
                        (getattr(request, "region_pass_records", {}) or {}).get("base") or {}
                    )
            base_reference_slots = [
                dict(item or {}) for item in list(base_region_reference.get("slots") or [])
            ]
            base_reference_width = int(base_region_reference.get("width") or 0)
            base_reference_height = int(base_region_reference.get("height") or 0)
            for slot_index, prompt in enumerate(list(expanded_positive_slots)):
                coordinate_reference_slot = None
                if slot_index < len(base_reference_slots):
                    candidate = base_reference_slots[slot_index]
                    # Only inherit resolved base geometry when hires is using the
                    # same REGION source prompt. An explicitly different hires
                    # prompt remains authoritative for its own coordinates.
                    if str(candidate.get("source_prompt") or "") == str(prompt or ""):
                        coordinate_reference_slot = candidate
                base_prompt, runtime_specs, slot_record = extract_superhybrid_region_slot(
                    prompt,
                    slot_index=slot_index,
                    steps=int(request.steps),
                    seed=int(parser_slot_seeds[slot_index]),
                    width=int(getattr(request, "generation_width", request.width)),
                    height=int(getattr(request, "generation_height", request.height)),
                    coordinate_reference_slot=coordinate_reference_slot,
                    coordinate_reference_width=base_reference_width if coordinate_reference_slot else None,
                    coordinate_reference_height=base_reference_height if coordinate_reference_slot else None,
                )
                expanded_positive_slots[slot_index] = base_prompt
                region_runtime_specs_by_slot[slot_index] = runtime_specs
                region_slot_records[slot_index] = slot_record
        except RegionalPromptError as exc:
            raise PromptParserError(
                f"REGION planning failed: {exc}",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="region_plan_failed",
                diagnostics={"pass": pass_name},
            ) from exc
        expanded_positive_prompt = expanded_positive_slots[0]

        positive_translations = [
            self.shortcut_translator.translate(
                prompt,
                profile=shortcut_profile,
                parser_id=descriptor.parser_id,
                prompt_role="positive",
            )
            for prompt in expanded_positive_slots
        ]
        negative_translations = [
            self.shortcut_translator.translate(
                prompt,
                profile=shortcut_profile,
                parser_id=descriptor.parser_id,
                prompt_role="negative",
            )
            for prompt in expanded_negative_slots
        ]
        positive_translation = positive_translations[0]
        negative_translation = negative_translations[0]
        if state is not None and hasattr(state, "p"):
            state.p.steps = request.steps
            state.p.batch_size = request.batch_size
            state.p.cfg_scale = request.cfg_scale
            state.p.width = int(getattr(request, "generation_width", request.width))
            state.p.height = int(getattr(request, "generation_height", request.height))
            state.p.positive_prompt = positive_translation.parser_input
            state.p.negative_prompt = negative_translation.parser_input
            state.p.positive_prompts_by_slot = [item.parser_input for item in positive_translations]
            state.p.negative_prompts_by_slot = [item.parser_input for item in negative_translations]
            if request.seed is not None:
                state.p.seed = request.seed
        generation_width = int(getattr(request, "generation_width", request.width))
        generation_height = int(getattr(request, "generation_height", request.height))
        recorded_route_plans = dict(getattr(request, "prompt_route_plan", {}) or {})
        try:
            pos_result = self._parse_prompt_slots(
                parser=parser,
                state=state,
                prompts=[item.parser_input for item in positive_translations],
                prompt_role="positive",
                steps=request.steps,
                hires_steps=hires_steps,
                model=cond_model,
                parser_options=parser_options,
                width=generation_width,
                height=generation_height,
                seeds=resolved_seeds,
                parser_seeds=parser_slot_seeds,
                recorded_route_plan=dict(recorded_route_plans.get("positive") or {}),
            )

            neg_result = self._parse_prompt_slots(
                parser=parser,
                state=state,
                prompts=[item.parser_input for item in negative_translations],
                prompt_role="negative",
                steps=request.steps,
                hires_steps=hires_steps,
                model=cond_model,
                parser_options=parser_options,
                width=generation_width,
                height=generation_height,
                seeds=resolved_seeds,
                parser_seeds=parser_slot_seeds,
                recorded_route_plan=dict(recorded_route_plans.get("negative") or {}),
            )
        except PromptParserError as exc:
            exc.diagnostics.setdefault("shortcut_profile", shortcut_profile.snapshot())
            exc.diagnostics.setdefault(
                "positive_translations", [item.metadata() for item in positive_translations]
            )
            exc.diagnostics.setdefault(
                "negative_translations", [item.metadata() for item in negative_translations]
            )
            raise

        region_entries: list[RegionConditioningEntry] = []
        region_semantic_digests_by_slot: list[list[str]] = [
            [] for _ in resolved_seeds
        ]
        region_translation_metadata: list[dict[str, Any]] = []
        for slot_index, runtime_specs in enumerate(region_runtime_specs_by_slot):
            for spec in runtime_specs:
                translation = self.shortcut_translator.translate(
                    spec.prompt,
                    profile=shortcut_profile,
                    parser_id=descriptor.parser_id,
                    prompt_role="positive",
                )
                try:
                    result = self._parse_prompt(
                        parser=parser,
                        state=state,
                        prompt=translation.parser_input,
                        prompt_role="positive",
                        steps=request.steps,
                        hires_steps=hires_steps,
                        model=cond_model,
                        parser_options={
                            **parser_options,
                            **({"seed": parser_slot_seeds[slot_index]} if descriptor.parser_id == "superhybrid" else {}),
                        },
                        width=generation_width,
                        height=generation_height,
                        seed=parser_slot_seeds[slot_index],
                        recorded_route_plan=None,
                    )
                except PromptParserError as exc:
                    exc.diagnostics.setdefault("region_slot_index", slot_index)
                    exc.diagnostics.setdefault("region_index", spec.region_index)
                    raise
                if dict((result.directives or {}).get("cfg") or {}):
                    raise PromptParserError(
                        "SuperHybrid CFG directives are not allowed inside REGION branches because one sampling batch owns one canonical CFG schedule.",
                        parser_id=descriptor.parser_id,
                        prompt_role="positive",
                        error_kind="region_cfg_directive_unsupported",
                        diagnostics={"slot_index": slot_index, "region_index": spec.region_index},
                    )
                slot_batches = list(getattr(result.conditioning_source.multicond, "batch", []) or [])
                if len(slot_batches) != 1:
                    raise PromptParserError(
                        "A REGION branch must resolve to exactly one conditioning batch item.",
                        parser_id=descriptor.parser_id,
                        prompt_role="positive",
                        error_kind="invalid_region_conditioning_batch",
                        diagnostics={"slot_index": slot_index, "region_index": spec.region_index},
                    )
                semantic = dict(result.diagnostics.get("semantic_fingerprint") or {})
                digest = str(semantic.get("digest") or "")
                region_semantic_digests_by_slot[slot_index].append(digest)
                region_entries.append(
                    RegionConditioningEntry(
                        spec=spec,
                        positive_multicond=result.conditioning_source.multicond,
                        semantic_digest=digest,
                    )
                )
                region_translation_metadata.append({
                    "slot_index": int(slot_index),
                    "region_index": int(spec.region_index),
                    "translation": translation.metadata(),
                    "canonical_prompt": str(result.canonical_prompt or ""),
                    "semantic_fingerprint": semantic,
                })

        try:
            current_region_record = build_region_record(
                parser_id=descriptor.parser_id,
                parser_version=descriptor.version,
                pass_name=pass_name,
                width=generation_width,
                height=generation_height,
                steps=int(request.steps),
                overlap_policy=str(parser_options.get("region_overlap_policy", "additive") or "additive"),
                slots=region_slot_records,
                semantic_digests_by_slot=region_semantic_digests_by_slot,
            )
            if region_replay_mode == "recorded_exact":
                recorded_regions = dict(getattr(request, "region_recorded", {}) or {})
                recorded_region = dict(recorded_regions.get(pass_name) or {})
                if not recorded_region:
                    raise RegionalPromptError(
                        f"Exact REGION replay was requested, but no recorded {pass_name} REGION plan was supplied."
                    )
                region_record = validate_recorded_region_record(
                    recorded_region,
                    current=current_region_record,
                )
            else:
                region_record = current_region_record
        except RegionalPromptError as exc:
            raise PromptParserError(
                f"REGION replay validation failed: {exc}",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="region_replay_failed",
                diagnostics={"pass": pass_name, "replay_mode": region_replay_mode},
            ) from exc
        region_pass_records = dict(getattr(request, "region_pass_records", {}) or {})
        if int(region_record.get("region_count", 0) or 0) > 0 or region_replay_mode == "recorded_exact":
            region_pass_records[pass_name] = dict(region_record)
        else:
            region_pass_records.pop(pass_name, None)
        request.region_pass_records = region_pass_records
        if isinstance(getattr(request, "diagnostics", None), dict):
            if int(region_record.get("region_count", 0) or 0) > 0:
                request.diagnostics["regional_prompting"] = dict(region_record)
            else:
                request.diagnostics.pop("regional_prompting", None)

        positive_semantic_fingerprints = list(
            pos_result.diagnostics.get("semantic_fingerprints_by_slot")
            or [pos_result.diagnostics.get("semantic_fingerprint")]
        )
        negative_semantic_fingerprints = list(
            neg_result.diagnostics.get("semantic_fingerprints_by_slot")
            or [neg_result.diagnostics.get("semantic_fingerprint")]
        )
        semantic_replay_mode = str(
            getattr(request, "prompt_semantic_replay_mode", "reconstruct")
            or "reconstruct"
        ).strip().lower()
        if semantic_replay_mode not in {"reconstruct", "recorded_exact"}:
            raise PromptParserError(
                "prompt_semantic_replay_mode must be reconstruct or recorded_exact.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="invalid_prompt_semantic_replay_mode",
            )
        semantic_record: dict[str, Any] = {}
        if descriptor.parser_id == "superhybrid":
            try:
                current_semantic_record = build_superhybrid_semantic_record(
                    parser_version=descriptor.version,
                    pass_name=pass_name,
                    scope=expansion_scope,
                    resolved_seeds=resolved_seeds,
                    selection_seeds=parser_slot_seeds,
                    positive_fingerprints=positive_semantic_fingerprints,
                    negative_fingerprints=negative_semantic_fingerprints,
                )
                recorded_semantics = dict(
                    getattr(request, "prompt_semantic_recorded", {}) or {}
                )
                if semantic_replay_mode == "recorded_exact":
                    recorded_semantic_record = dict(
                        recorded_semantics.get(pass_name) or {}
                    )
                    if not recorded_semantic_record:
                        raise PromptSemanticReplayError(
                            f"Exact SuperHybrid semantic replay was requested, but no recorded {pass_name} semantic contract was supplied."
                        )
                    semantic_record = validate_recorded_superhybrid_semantic_record(
                        recorded_semantic_record,
                        current=current_semantic_record,
                    )
                else:
                    semantic_record = current_semantic_record
            except PromptSemanticReplayError as exc:
                raise PromptParserError(
                    f"SuperHybrid semantic replay validation failed: {exc}",
                    parser_id=descriptor.parser_id,
                    prompt_role="positive",
                    error_kind="superhybrid_semantic_replay_failed",
                    diagnostics={
                        "pass": pass_name,
                        "replay_mode": semantic_replay_mode,
                    },
                ) from exc
            semantic_pass_records = dict(
                getattr(request, "prompt_semantic_pass_records", {}) or {}
            )
            semantic_pass_records[pass_name] = dict(semantic_record)
            request.prompt_semantic_pass_records = semantic_pass_records
        elif semantic_replay_mode == "recorded_exact" and getattr(
            request, "prompt_semantic_recorded", None
        ):
            raise PromptParserError(
                "Recorded SuperHybrid semantic data cannot be applied to a different prompt parser.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="prompt_semantic_parser_mismatch",
            )

        parsed_prompt_cfg = dict((pos_result.directives or {}).get("cfg") or {})
        replay_mode = str(
            getattr(request, "prompt_cfg_replay_mode", "reconstruct") or "reconstruct"
        ).strip().lower()
        if replay_mode not in {"reconstruct", "recorded_exact"}:
            raise PromptParserError(
                "prompt_cfg_replay_mode must be reconstruct or recorded_exact.",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="invalid_prompt_cfg_replay_mode",
            )
        recorded_passes = dict(
            getattr(request, "prompt_cfg_recorded_schedules", {}) or {}
        )
        recorded_prompt_cfg = dict(recorded_passes.get(pass_name) or {})
        try:
            if replay_mode == "recorded_exact" and recorded_prompt_cfg:
                prompt_cfg_schedule = validate_recorded_prompt_cfg_payload(
                    recorded_prompt_cfg,
                    total_steps=int(request.steps),
                    pass_name=pass_name,
                )
                prompt_cfg_schedule["parsed_directive"] = dict(parsed_prompt_cfg)
            elif parsed_prompt_cfg:
                prompt_cfg_schedule = finalize_prompt_cfg_payload(
                    parsed_prompt_cfg,
                    ui_cfg_scale=float(request.cfg_scale),
                    total_steps=int(request.steps),
                    pass_name=pass_name,
                )
            elif replay_mode == "recorded_exact":
                raise PromptCFGScheduleError(
                    f"Exact prompt CFG replay was requested, but no recorded {pass_name} schedule was supplied."
                )
            else:
                prompt_cfg_schedule = {}
        except PromptCFGScheduleError as exc:
            raise PromptParserError(
                f"Prompt CFG schedule resolution failed: {exc}",
                parser_id=descriptor.parser_id,
                prompt_role="positive",
                error_kind="prompt_cfg_schedule_resolution_failed",
                diagnostics={"pass": pass_name, "replay_mode": replay_mode},
            ) from exc

        request.prompt_cfg_schedule = prompt_cfg_schedule
        pass_schedules = dict(getattr(request, "prompt_cfg_pass_schedules", {}) or {})
        if prompt_cfg_schedule:
            pass_schedules[pass_name] = dict(prompt_cfg_schedule)
        else:
            pass_schedules.pop(pass_name, None)
        request.prompt_cfg_pass_schedules = pass_schedules
        if isinstance(getattr(request, "diagnostics", None), dict):
            if prompt_cfg_schedule:
                request.diagnostics["prompt_cfg_schedule"] = dict(prompt_cfg_schedule)
            else:
                request.diagnostics.pop("prompt_cfg_schedule", None)
        if state is not None and hasattr(state, "p"):
            state.p.prompt_cfg_schedule = dict(prompt_cfg_schedule)

        base_resolver = StepConditioningResolver(
            positive_multicond=pos_result.conditioning_source.multicond,
            negative_multicond=neg_result.conditioning_source.multicond,
            total_steps=request.steps,
        )
        regional_resolver = None
        if region_entries:
            regional_resolver = RegionalConditioningResolver(
                base_resolver=base_resolver,
                entries=region_entries,
                total_steps=int(request.steps),
                generation_width=generation_width,
                generation_height=generation_height,
                overlap_policy=str(region_record.get("overlap_policy") or "additive"),
                region_counts_by_slot={
                    int(item.get("slot_index", index)): int(item.get("region_count", 0) or 0)
                    for index, item in enumerate(region_slot_records)
                },
                pass_name=pass_name,
            )
        resolver = regional_resolver or base_resolver

        # Initial tensors for step 0 so the pipeline has direct cond/uncond fields.
        cond0, uncond0 = resolver.resolve(step_index=0)
        pooled_cond0 = None
        pooled_uncond0 = None
        has_pooled_conditioning = isinstance(cond_model, (SDXLConditioningRuntime, SD3ConditioningRuntime))
        if has_pooled_conditioning:
            # Pooled SDXL and SD3 conditioning follows the same parsed prompt
            # schedules and composable weights as token conditioning. Regional
            # guidance remains spatially applied only to token/cross-attention
            # conditioning; pooled conditioning resolves from the base schedule.
            pooled_cond0, pooled_uncond0 = base_resolver.resolve_pooled(step_index=0)

        prompt_schedules = {
            "positive": list(pos_result.conditioning_source.flat_list),
            "negative": list(neg_result.conditioning_source.flat_list),
        }
        parser_warnings = [
            *(warning for item in positive_translations for warning in item.warnings),
            *(warning for item in negative_translations for warning in item.warnings),
            *pos_result.warnings,
            *neg_result.warnings,
        ]
        parser_metadata = {
            "name": descriptor.parser_id,
            "label": descriptor.label,
            "version": descriptor.version,
            "contract_version": descriptor.contract_version,
            "experimental": descriptor.experimental,
            "credit": descriptor.credit,
            "source_url": descriptor.source_url,
            "capabilities": dict(descriptor.capabilities),
            "selection_source": parser_selection_source,
            "options_used": dict(pos_result.diagnostics.get("options_used") or parser_options),
            "warnings": parser_warnings,
            "warning_count": len(parser_warnings),
            "positive_diagnostics": dict(pos_result.diagnostics),
            "negative_diagnostics": dict(neg_result.diagnostics),
            "directives": {
                "positive": dict(pos_result.directives or {}),
                "negative": dict(neg_result.directives or {}),
            },
            "prompt_cfg_schedule": dict(prompt_cfg_schedule),
            "prompt_cfg_replay_mode": replay_mode,
            "prompt_cfg_replay_locked": bool(prompt_cfg_schedule.get("replay_locked", False)),
            "prompt_expansion": dict(expansion_record),
            "prompt_expansion_replay_mode": expansion_replay_mode,
            "prompt_expansion_replay_locked": bool(expansion_record.get("replay_locked", False)),
            "prompt_expansion_scope": str(expansion_record.get("scope") or expansion_scope),
            "prompt_slot_count": len(resolved_seeds),
            "semantic_fingerprints_by_slot": {
                "positive": positive_semantic_fingerprints,
                "negative": negative_semantic_fingerprints,
            },
            "semantic_record": dict(semantic_record),
            "semantic_replay_mode": semantic_replay_mode,
            "semantic_replay_locked": bool(semantic_record.get("replay_locked", False)),
            "regional_prompting": dict(region_record),
            "region_replay_mode": region_replay_mode,
            "region_replay_locked": bool(region_record.get("replay_locked", False)),
            "region_translations": list(region_translation_metadata),
            "route_plan": {
                "positive": dict(pos_result.diagnostics.get("route_plan") or {}),
                "negative": dict(neg_result.diagnostics.get("route_plan") or {}),
            },
            "shadow_comparison": {
                "positive": dict((getattr(request, "prompt_preflight", {}) or {}).get("base", {}).get("positive", {}).get("shadow_comparison") or {}),
                "negative": dict((getattr(request, "prompt_preflight", {}) or {}).get("base", {}).get("negative", {}).get("shadow_comparison") or {}),
            },
        }
        shortcut_profile_metadata = {
            "name": shortcut_profile.profile_id,
            "label": shortcut_profile.label,
            "version": shortcut_profile.version,
            "contract_version": PROMPT_SHORTCUT_CONTRACT_VERSION,
            "mapping_hash": shortcut_profile.mapping_hash,
            "builtin": shortcut_profile.builtin,
            "source": shortcut_profile.source,
            "credit": shortcut_profile.credit,
            "effective_mapping": shortcut_profile.snapshot(),
            "parser_preset_name": str(getattr(request, "prompt_parser_preset_name", "") or ""),
        }
        prompt_translation = {
            "contract_version": PROMPT_SHORTCUT_CONTRACT_VERSION,
            "positive": positive_translation.metadata(),
            "negative": negative_translation.metadata(),
            "positive_slots": [item.metadata() for item in positive_translations],
            "negative_slots": [item.metadata() for item in negative_translations],
            "substitution_count": sum(len(item.substitutions) for item in positive_translations + negative_translations),
            "warnings": [
                *(warning for item in positive_translations for warning in item.warnings),
                *(warning for item in negative_translations for warning in item.warnings),
            ],
        }
        prompt_preflight = dict(getattr(request, "prompt_preflight", {}) or {})
        hires_processing = dict(prompt_preflight.get("hires") or {})
        hires_routing = {
            "parser_mode": str(getattr(request, "hires_prompt_parser_mode", "same_as_base") or "same_as_base"),
            "parser_name": str(getattr(request, "hires_prompt_parser_name", descriptor.parser_id) or descriptor.parser_id),
            "parser_kwargs": dict(getattr(request, "hires_prompt_parser_kwargs", {}) or {}),
            "shortcut_profile_mode": str(getattr(request, "hires_shortcut_profile_mode", "same_as_base") or "same_as_base"),
            "shortcut_profile_name": str(getattr(request, "hires_shortcut_profile_name", shortcut_profile.profile_id) or shortcut_profile.profile_id),
            "shortcut_profile_snapshot": dict(getattr(request, "hires_shortcut_profile_snapshot", {}) or {}),
            "positive_prompt": str(getattr(request, "hires_positive_prompt", request.positive_prompt) or request.positive_prompt),
            "negative_prompt": str(getattr(request, "hires_negative_prompt", "") or request.negative_prompt or ""),
            "preflight": hires_processing,
        }
        prompt_contract = {
            "canonical_contract_version": CANONICAL_PROMPT_CONTRACT_VERSION,
            "shortcut_contract_version": PROMPT_SHORTCUT_CONTRACT_VERSION,
            "raw_positive": request.positive_prompt,
            "raw_negative": request.negative_prompt or "",
            "expanded_positive": expanded_positive_prompt,
            "expanded_negative": expanded_negative_prompt,
            "expanded_positive_by_slot": list(expanded_positive_slots),
            "expanded_negative_by_slot": list(expanded_negative_slots),
            "prompt_expansion": dict(expansion_record),
            "translated_positive": positive_translation.parser_input,
            "translated_negative": negative_translation.parser_input,
            "translated_positive_by_slot": [item.parser_input for item in positive_translations],
            "translated_negative_by_slot": [item.parser_input for item in negative_translations],
            "shortcut_canonical_positive": positive_translation.canonical_prompt,
            "shortcut_canonical_negative": negative_translation.canonical_prompt,
            "shortcut_canonical_positive_structure": positive_translation.canonical_structure,
            "shortcut_canonical_negative_structure": negative_translation.canonical_structure,
            "canonical_positive": pos_result.canonical_prompt,
            "canonical_negative": neg_result.canonical_prompt,
            "canonical_positive_by_slot": list(pos_result.diagnostics.get("canonical_prompts_by_slot") or [pos_result.canonical_prompt]),
            "canonical_negative_by_slot": list(neg_result.diagnostics.get("canonical_prompts_by_slot") or [neg_result.canonical_prompt]),
            "canonical_positive_structure": pos_result.canonical_structure,
            "canonical_negative_structure": neg_result.canonical_structure,
            "shortcut_profile": shortcut_profile_metadata,
            "route_plan": parser_metadata.get("route_plan") or {},
            "shadow_comparison": parser_metadata.get("shadow_comparison") or {},
            "hires": hires_routing,
            "prompt_cfg_schedule": dict(prompt_cfg_schedule),
            "regional_prompting": dict(region_record),
        }
        prompt_processing = {
            "contract_version": "image-gen-prompt-processing-v1",
            "base": {
                "parser": parser_metadata,
                "shortcut_profile": shortcut_profile_metadata,
                "raw": {"positive": request.positive_prompt, "negative": request.negative_prompt or ""},
                "expanded": {"positive": expanded_positive_prompt, "negative": expanded_negative_prompt},
                "expanded_by_slot": {
                    "positive": list(expanded_positive_slots),
                    "negative": list(expanded_negative_slots),
                },
                "prompt_expansion": dict(expansion_record),
                "canonical": {"positive": pos_result.canonical_prompt, "negative": neg_result.canonical_prompt},
                "canonical_by_slot": {
                    "positive": list(pos_result.diagnostics.get("canonical_prompts_by_slot") or [pos_result.canonical_prompt]),
                    "negative": list(neg_result.diagnostics.get("canonical_prompts_by_slot") or [neg_result.canonical_prompt]),
                },
                "route_plan": parser_metadata.get("route_plan") or {},
                "shadow_comparison": parser_metadata.get("shadow_comparison") or {},
                "prompt_cfg_schedule": dict(prompt_cfg_schedule),
                "regional_prompting": dict(region_record),
            },
            "hires": hires_routing,
            "preflight": prompt_preflight,
        }

        if state is not None and hasattr(state, "p"):
            state.p.cond = cond0
            state.p.uncond = uncond0
            state.p.prompt_schedules = prompt_schedules
            
       
            cond = state.p.cond
            uncond = state.p.uncond
            diff = (cond - uncond).abs().mean().item()
            
            
           
        
        return ConditioningOutput(
            cond=cond0,
            uncond=uncond0,
            prompt_schedules=prompt_schedules,
            pooled_cond=pooled_cond0,
            pooled_uncond=pooled_uncond0,
            extra={
                "positive_parsed": pos_result.conditioning_source,
                "negative_parsed": neg_result.conditioning_source,
                "positive_parse_result": pos_result,
                "negative_parse_result": neg_result,
                "resolver": resolver,
                "pooled_resolver": base_resolver if has_pooled_conditioning else None,
                "conditioning_architecture": (
                    "sd3.x"
                    if isinstance(cond_model, SD3ConditioningRuntime)
                    else ("sdxl" if isinstance(cond_model, SDXLConditioningRuntime) else "legacy")
                ),
                "conditioning_metadata": (
                    cond_model.contract_metadata()
                    if isinstance(cond_model, SD3ConditioningRuntime)
                    else {}
                ),
                "prompt_parser": parser_metadata,
                "prompt_shortcut_profile": shortcut_profile_metadata,
                "prompt_translation": prompt_translation,
                "prompt_contract": prompt_contract,
                "prompt_processing": prompt_processing,
                "prompt_preflight": prompt_preflight,
                "prompt_cfg_schedule": dict(prompt_cfg_schedule),
                "regional_prompting": dict(region_record),
                "regional_resolver": regional_resolver,
            },
        )
