# prompt_conditioning_adapter.py

from __future__ import annotations

from dataclasses import dataclass
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
    PromptParserError,
    PromptParserRegistry,
    default_prompt_parser_registry,
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

@dataclass
class StepConditioningResolver:
    positive_multicond: MulticondLearnedConditioning
    negative_multicond: MulticondLearnedConditioning
    total_steps: int

    def _select_schedule_cond(
        self,
        schedules: list[ScheduledPromptConditioning],
        step_index: int,
    ) -> torch.Tensor:
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

                cond_tensor = self._select_schedule_cond(schedules, step_index)

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
        cond = self._resolve_multicond_for_step(self.positive_multicond, step_index)
        uncond = self._resolve_multicond_for_step(self.negative_multicond, step_index)
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
            fallback = "legacy_default" if parser_id == "legacy" else ("parser21_native" if parser_id == "parser21" else "canonical")
            profile_name = str(getattr(request, "prompt_shortcut_profile_name", "") or fallback)
            if (parser_id == "parser21" and profile_name == "legacy_default") or (parser_id == "legacy" and profile_name == "parser21_native"):
                profile_name = fallback
            profile = self.shortcut_registry.get(profile_name)
        compatible = parser_id in profile.compatible_parsers or (
            parser_id == "combined" and any(item in profile.compatible_parsers for item in ("legacy", "parser21"))
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
        positive_translation = self.shortcut_translator.translate(
            request.positive_prompt,
            profile=shortcut_profile,
            parser_id=descriptor.parser_id,
            prompt_role="positive",
        )
        negative_translation = self.shortcut_translator.translate(
            request.negative_prompt or "",
            profile=shortcut_profile,
            parser_id=descriptor.parser_id,
            prompt_role="negative",
        )
        if state is not None and hasattr(state, "p"):
            state.p.steps = request.steps
            state.p.batch_size = request.batch_size
            state.p.cfg_scale = request.cfg_scale
            state.p.width = int(getattr(request, "generation_width", request.width))
            state.p.height = int(getattr(request, "generation_height", request.height))
            state.p.positive_prompt = positive_translation.parser_input
            state.p.negative_prompt = negative_translation.parser_input
            if request.seed is not None:
                state.p.seed = request.seed
        generation_width = int(getattr(request, "generation_width", request.width))
        generation_height = int(getattr(request, "generation_height", request.height))
        recorded_route_plans = dict(getattr(request, "prompt_route_plan", {}) or {})
        try:
            pos_result = self._parse_prompt(
                parser=parser,
                state=state,
                prompt=positive_translation.parser_input,
                prompt_role="positive",
                steps=request.steps,
                hires_steps=hires_steps,
                model=cond_model,
                parser_options=parser_options,
                width=generation_width,
                height=generation_height,
                seed=request.seed,
                recorded_route_plan=dict(recorded_route_plans.get("positive") or {}),
            )

            neg_result = self._parse_prompt(
                parser=parser,
                state=state,
                prompt=negative_translation.parser_input,
                prompt_role="negative",
                steps=request.steps,
                hires_steps=hires_steps,
                model=cond_model,
                parser_options=parser_options,
                width=generation_width,
                height=generation_height,
                seed=request.seed,
                recorded_route_plan=dict(recorded_route_plans.get("negative") or {}),
            )
        except PromptParserError as exc:
            exc.diagnostics.setdefault("shortcut_profile", shortcut_profile.snapshot())
            exc.diagnostics.setdefault("positive_translation", positive_translation.metadata())
            exc.diagnostics.setdefault("negative_translation", negative_translation.metadata())
            raise

        resolver = StepConditioningResolver(
            positive_multicond=pos_result.conditioning_source.multicond,
            negative_multicond=neg_result.conditioning_source.multicond,
            total_steps=request.steps,
        )

        # Initial tensors for step 0 so the pipeline has direct cond/uncond fields.
        cond0, uncond0 = resolver.resolve(step_index=0)

        prompt_schedules = {
            "positive": list(pos_result.conditioning_source.flat_list),
            "negative": list(neg_result.conditioning_source.flat_list),
        }
        parser_warnings = [
            *positive_translation.warnings,
            *negative_translation.warnings,
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
            "substitution_count": len(positive_translation.substitutions) + len(negative_translation.substitutions),
            "warnings": [*positive_translation.warnings, *negative_translation.warnings],
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
            "translated_positive": positive_translation.parser_input,
            "translated_negative": negative_translation.parser_input,
            "shortcut_canonical_positive": positive_translation.canonical_prompt,
            "shortcut_canonical_negative": negative_translation.canonical_prompt,
            "shortcut_canonical_positive_structure": positive_translation.canonical_structure,
            "shortcut_canonical_negative_structure": negative_translation.canonical_structure,
            "canonical_positive": pos_result.canonical_prompt,
            "canonical_negative": neg_result.canonical_prompt,
            "canonical_positive_structure": pos_result.canonical_structure,
            "canonical_negative_structure": neg_result.canonical_structure,
            "shortcut_profile": shortcut_profile_metadata,
            "route_plan": parser_metadata.get("route_plan") or {},
            "shadow_comparison": parser_metadata.get("shadow_comparison") or {},
            "hires": hires_routing,
        }
        prompt_processing = {
            "contract_version": "image-gen-prompt-processing-v1",
            "base": {
                "parser": parser_metadata,
                "shortcut_profile": shortcut_profile_metadata,
                "raw": {"positive": request.positive_prompt, "negative": request.negative_prompt or ""},
                "canonical": {"positive": pos_result.canonical_prompt, "negative": neg_result.canonical_prompt},
                "route_plan": parser_metadata.get("route_plan") or {},
                "shadow_comparison": parser_metadata.get("shadow_comparison") or {},
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
            pooled_cond=None,
            pooled_uncond=None,
            extra={
                "positive_parsed": pos_result.conditioning_source,
                "negative_parsed": neg_result.conditioning_source,
                "positive_parse_result": pos_result,
                "negative_parse_result": neg_result,
                "resolver": resolver,
                "prompt_parser": parser_metadata,
                "prompt_shortcut_profile": shortcut_profile_metadata,
                "prompt_translation": prompt_translation,
                "prompt_contract": prompt_contract,
                "prompt_processing": prompt_processing,
                "prompt_preflight": prompt_preflight,
            },
        )
