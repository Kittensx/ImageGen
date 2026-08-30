from __future__ import annotations
from modules.parser.learned_conditioning import LearnedConditioning
from modules.parser.mixin import ScheduledPromptConditioning


import re
from collections import namedtuple
from lark import Lark
import random
import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from lark import Lark


def _restore_literal_scope_markers(text, replacements):
    value = str(text or "")
    for marker, literal in tuple(replacements or ()):
        value = value.replace(str(marker), str(literal))
    return value


@dataclass
class ParsedPromptResult:
    multicond: MulticondLearnedConditioning
    schedules: List[List[ScheduledPromptConditioning]]
    flat_list: list[str]

class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)
        
class ComposableScheduledPromptConditioning:
    def __init__(
        self,
        schedules,
        weight=1.0,
        *,
        active_until_step=None,
        hold_after_step=False,
        semantic_role="text",
        group_operation_id=None,
        group_local_weight=1.0,
        group_member_path=(),
        average_operation_id=None,
        average_local_weight=1.0,
        average_branch_index=None,
        composition_operation_id=None,
        composition_mode="",
        composition_algorithm="",
        composition_branch_index=None,
        sequence_operation_id=None,
        sequence_local_weight=1.0,
        sequence_item_index=None,
        relation_operation_id=None,
        relation_parent="",
        relation_child="",
        owner_text="",
        syntax_origin="",
        source_span=(None, None),
        terminator_consumed="",
        temporal_active_by_step=(),
        temporal_source="",
        chunk_break_segments=(),
        chunk_break_count=0,
    ):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = float(weight)
        self.active_until_step: int | None = active_until_step
        self.hold_after_step: bool = bool(hold_after_step)
        self.semantic_role: str = str(semantic_role or "text")
        self.group_operation_id: str | None = (
            str(group_operation_id) if group_operation_id is not None else None
        )
        self.group_local_weight: float = float(group_local_weight)
        self.group_member_path: tuple[int, ...] = tuple(group_member_path or ())
        self.average_operation_id: str | None = (
            str(average_operation_id) if average_operation_id is not None else None
        )
        self.average_local_weight: float = float(average_local_weight)
        self.average_branch_index: int | None = (
            int(average_branch_index) if average_branch_index is not None else None
        )
        self.composition_operation_id: str | None = (
            str(composition_operation_id) if composition_operation_id is not None else None
        )
        self.composition_mode: str = str(composition_mode or "")
        self.composition_algorithm: str = str(composition_algorithm or "")
        self.composition_branch_index: int | None = (
            int(composition_branch_index) if composition_branch_index is not None else None
        )
        self.sequence_operation_id: str | None = (
            str(sequence_operation_id) if sequence_operation_id is not None else None
        )
        self.sequence_local_weight: float = float(sequence_local_weight)
        self.sequence_item_index: int | None = (
            int(sequence_item_index) if sequence_item_index is not None else None
        )
        self.relation_operation_id: str | None = (
            str(relation_operation_id) if relation_operation_id is not None else None
        )
        self.relation_parent: str = str(relation_parent or "")
        self.relation_child: str = str(relation_child or "")
        self.owner_text: str = str(owner_text or "")
        self.syntax_origin: str = str(syntax_origin or "")
        self.source_span: tuple[int | None, int | None] = tuple(source_span or (None, None))
        self.terminator_consumed: str = str(terminator_consumed or "")
        self.temporal_active_by_step: tuple[bool, ...] = tuple(bool(x) for x in (temporal_active_by_step or ()))
        self.temporal_source: str = str(temporal_source or "")
        self.chunk_break_segments: tuple[str, ...] = tuple(str(x) for x in (chunk_break_segments or ()))
        self.chunk_break_count: int = int(chunk_break_count or 0)

    def is_active_at_step(self, step_1_based: int) -> bool:
        step = int(step_1_based)
        if self.temporal_active_by_step and 1 <= step <= len(self.temporal_active_by_step):
            if not self.temporal_active_by_step[step - 1]:
                return False
        if self.active_until_step is None:
            return True
        if step <= int(self.active_until_step):
            return True
        return bool(self.hold_after_step)

    def weight_at_step(self, step_1_based: int) -> float:
        return float(self.weight) if self.is_active_at_step(step_1_based) else 0.0
        
class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch
        



class PromptParserClass:
    def __init__(
        self,
        shared_state,
        prompts: SdConditioning | list[str],
        steps: int,
        model: nn.Module,
        positive_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        hires_steps: Optional[int] = None,
        conditioning_plans: Optional[List[ConditioningPlan]] = None,
        semantic_modes: Optional[dict[str, str]] = None,
    ):
        self.state = shared_state
        self.cfgm = self.state.cfgm
        self.p = self.state.p
        self.conditioning = self.state.conditioning
        self.sched = self.state.sched
        self.model = model if model is not None else self.cfgm.model        
        self.prompts = prompts
        self.steps = steps if steps is not None else self.p.steps
        self.hires_steps = hires_steps if hires_steps else None
        self.learned_conditioning = LearnedConditioning(shared_state = self.state, steps = self.steps, hires_steps = self.hires_steps, prompts = self.prompts)
        self.conditioning_plans = list(conditioning_plans or [])
        self.semantic_modes = {str(key): str(value) for key, value in dict(semantic_modes or {}).items()}
        self._last_branch_metadata = []
        #self.learned_conditioning.get_learned_cond()

    def _conditioning_plan_for(self, index: int, prompt: str):
        # Local imports avoid a package bootstrap cycle: prompt_parsers.__init__
        # registers the Legacy adapter, which itself imports PromptParserClass.
        from modules.prompt_parsers.compiler import compile_conditioning_plan
        from modules.prompt_parsers.ir import parse_prompt_ir

        prompt_ir = parse_prompt_ir(
            prompt, parser_namespace="legacy", semantic_modes=self.semantic_modes
        )
        if index < len(self.conditioning_plans):
            plan = self.conditioning_plans[index]
            if str(getattr(plan, "source", "")) == str(prompt_ir.normalized_source):
                return plan
        return compile_conditioning_plan(prompt_ir)

    def _has_structured_runtime_lowering(self) -> bool:
        return any(
            self._conditioning_plan_for(index, str(prompt)).lowering_required
            for index, prompt in enumerate(self.prompts)
        )

    def __call__(self) -> ParsedPromptResult:
        if self._has_structured_runtime_lowering():
            # Structural syntax must be lowered before *any* encoder call.
            # The composable path already owns the per-branch schedule lists, so
            # derive the diagnostic schedule surface from those encoded branches
            # rather than encoding the raw structural source first.
            multicond = self.get_multicond_learned_conditioning()
            schedules = [
                item.schedules
                for batch_items in multicond.batch
                for item in batch_items
            ]
        else:
            schedules = self.get_learned_conditioning()
            multicond = self.get_multicond_learned_conditioning()
        res_indexes, flat_list, _ = self.get_multicond_prompt_list()

        # get_multicond_prompt_list() intentionally returns scheduler-protected
        # marker text for runtime consumption. ParsedPromptResult.flat_list is a
        # diagnostic/user-facing surface, so never expose those opaque markers.
        restored_flat_list = list(flat_list)
        metadata_batches = list(getattr(self, "_last_branch_metadata", ()) or ())
        for prompt_indexes, branch_metadata in zip(res_indexes, metadata_batches):
            for (flat_index, _weight), branch in zip(prompt_indexes, branch_metadata):
                if branch is None:
                    continue
                replacements = tuple(
                    getattr(branch, "literal_scope_replacements", ()) or ()
                )
                if replacements:
                    restored_flat_list[flat_index] = _restore_literal_scope_markers(
                        restored_flat_list[flat_index], replacements
                    )

        return ParsedPromptResult(
            multicond=multicond,
            schedules=schedules,
            flat_list=restored_flat_list
        )

    def _uses_a1111_temporal(self) -> bool:
        return (
            self.semantic_modes.get("schedule_algorithm") == "a1111_schedule_v1"
            or self.semantic_modes.get("alternate_algorithm") == "a1111_alternate_v1"
        )

    def _uses_a1111_clip_conditioning(self) -> bool:
        return (
            self.semantic_modes.get("attention_algorithm") == "a1111_attention_v1"
            or self.semantic_modes.get("clip_chunking") == "a1111_clip_chunk_v1"
        )

    def _a1111_use_old_scheduling(self) -> bool:
        return bool(getattr(self.conditioning, "use_old_scheduling", False))

    def _compile_prompt_schedule(self, prompt: str, *, steps: int, hires_steps: int | None):
        if self._uses_a1111_temporal():
            from modules.prompt_parsers.a1111_semantics import compile_a1111_temporal_text

            result = compile_a1111_temporal_text(
                str(prompt),
                base_steps=int(steps),
                hires_steps=hires_steps,
                use_old_scheduling=self._a1111_use_old_scheduling(),
            )
            return [(segment.end_step, segment.text) for segment in result.segments]
        return self.learned_conditioning.get_learned_cond([prompt], steps, hires_steps)[0]

    def _encode_conditioning_batch(self, model, texts, *, forced_segments_by_prompt=None):
        values = list(texts)
        if self._uses_a1111_clip_conditioning():
            encoder = getattr(model, "encode_a1111_conditioning", None)
            capabilities = getattr(model, "a1111_prompt_capabilities", None)
            if not callable(encoder) or not callable(capabilities):
                raise RuntimeError(
                    f"A1111 Compatible requires an explicit runtime capability declaration and "
                    f"encode_a1111_conditioning() hook; runtime={type(model).__name__}."
                )
            declared = capabilities()
            if not bool(getattr(declared, "attention", False)) or not bool(getattr(declared, "long_clip_chunking", False)):
                raise RuntimeError(
                    f"Runtime {type(model).__name__} does not declare A1111 attention/long-CLIP support."
                )
            return encoder(values, forced_segments_by_prompt=forced_segments_by_prompt)
        if hasattr(model, "get_learned_conditioning"):
            return model.get_learned_conditioning(texts)
        if hasattr(model, "encode"):
            return model.encode(texts)
        raise AttributeError("Model does not support conditioning")

    def get_learned_conditioning(
        self,
        model=None,
        prompts=None,
        steps=None,
        hires_steps=None,
        literal_replacements_by_prompt=None,
    ):
        

        """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
        and the sampling step at which this condition is to be replaced by the next one.

        """
        #res: List[List[ScheduledPromptConditioning]]
        res = []
        model = model if model is not None else self.model
        prompts = prompts if prompts is not None else self.prompts
        steps = steps if steps is not None else self.steps
        hires_steps = hires_steps if hires_steps is not None else self.hires_steps
        replacements_by_prompt = list(literal_replacements_by_prompt or [])

        prompt_schedules = [
            self._compile_prompt_schedule(str(prompt), steps=int(steps), hires_steps=hires_steps)
            for prompt in prompts
        ]
        cache = {}

        for prompt_index, (prompt, prompt_schedule) in enumerate(zip(prompts, prompt_schedules)):
            replacements = (
                tuple(replacements_by_prompt[prompt_index])
                if prompt_index < len(replacements_by_prompt)
                else ()
            )
            cache_key = (str(prompt), replacements)
            cached = cache.get(cache_key, None)
            if cached is not None:
                res.append(cached)
                continue

            texts = SdConditioning(
                prompts=[_restore_literal_scope_markers(x[1], replacements) for x in prompt_schedule],
                width=self.p.width,
                height=self.p.height,
                copy_from=self.prompts
            )

            conds = self._encode_conditioning_batch(model, texts)


            cond_schedule = []
            for i, (end_at_step, _) in enumerate(prompt_schedule):
                if isinstance(conds, dict):
                    cond = {k: v[i] for k, v in conds.items()}
                else:
                    cond = conds[i]

                cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

            cache[cache_key] = cond_schedule
            res.append(cond_schedule)

        return res



    def get_multicond_learned_conditioning(self) -> MulticondLearnedConditioning:
        """Build composable conditioning from compiled branch intent.

        PPSR-06 compiles standard Classic schedules/alternates before encode.
        Non-temporal branches retain the historical LearnedConditioning path;
        temporal branches encode only their resolved unique texts, so raw
        bracket control syntax never reaches CLIP/T5.
        """
        from modules.prompt_parsers.temporal_semantics import compile_temporal_text
        from modules.prompt_parsers.a1111_semantics import compile_a1111_temporal_text

        res_indexes, prompt_flat_list, _ = self.get_multicond_prompt_list()
        metadata_batches = list(self._last_branch_metadata or [])

        metadata_by_flat_index = {}
        for batch_index, indexes in enumerate(res_indexes):
            batch_meta = metadata_batches[batch_index] if batch_index < len(metadata_batches) else []
            for branch_index, (flat_index, _weight) in enumerate(indexes):
                branch = batch_meta[branch_index] if branch_index < len(batch_meta) else None
                metadata_by_flat_index.setdefault(flat_index, branch)

        temporal_results = {}
        chunk_break_results = {}
        non_temporal_indices = []
        active_steps = int(self.hires_steps or self.steps)
        for index, text in enumerate(prompt_flat_list):
            branch = metadata_by_flat_index.get(index)
            chunk_segments = tuple(getattr(branch, "chunk_break_segments", ()) or ()) if branch is not None else ()
            if chunk_segments:
                if bool(getattr(branch, "temporal_compiled", False)):
                    if not self._uses_a1111_temporal():
                        raise ValueError(
                            "BREAK cannot currently share one branch with temporal scheduling outside the A1111 profile."
                        )
                    temporal_results[index] = compile_a1111_temporal_text(
                        " BREAK ".join(chunk_segments),
                        base_steps=int(self.steps),
                        hires_steps=self.hires_steps,
                        use_old_scheduling=self._a1111_use_old_scheduling(),
                    )
                else:
                    chunk_break_results[index] = chunk_segments
            elif branch is not None and bool(getattr(branch, "temporal_compiled", False)):
                if self._uses_a1111_temporal():
                    temporal_results[index] = compile_a1111_temporal_text(
                        str(text),
                        base_steps=int(self.steps),
                        hires_steps=self.hires_steps,
                        use_old_scheduling=self._a1111_use_old_scheduling(),
                    )
                else:
                    temporal_results[index] = compile_temporal_text(str(text), active_steps)
            else:
                non_temporal_indices.append(index)

        schedules_by_index = {}
        if non_temporal_indices:
            non_texts = SdConditioning(
                prompts=[prompt_flat_list[i] for i in non_temporal_indices],
                width=self.p.width,
                height=self.p.height, copy_from=self.prompts,
            )
            non_schedules = self.get_learned_conditioning(
                model=self.model,
                prompts=non_texts,
                steps=self.steps,
                hires_steps=self.hires_steps,
                literal_replacements_by_prompt=[
                    tuple(getattr(metadata_by_flat_index.get(i), "literal_scope_replacements", ()) or ())
                    for i in non_temporal_indices
                ],
            )
            for flat_index, schedule in zip(non_temporal_indices, non_schedules):
                schedules_by_index[flat_index] = schedule

        if temporal_results:
            unique_texts = []
            unique_index = {}
            for flat_index, result in temporal_results.items():
                branch = metadata_by_flat_index.get(flat_index)
                replacements = tuple(getattr(branch, "literal_scope_replacements", ()) or ())
                for segment in result.segments:
                    encoder_text = _restore_literal_scope_markers(segment.text, replacements)
                    if encoder_text not in unique_index:
                        unique_index[encoder_text] = len(unique_texts)
                        unique_texts.append(encoder_text)
            texts = SdConditioning(
                prompts=unique_texts, width=self.p.width, height=self.p.height, copy_from=self.prompts
            )
            encoded = self._encode_conditioning_batch(self.model, texts)

            def encoded_at(idx):
                if isinstance(encoded, dict):
                    return {key: value[idx] for key, value in encoded.items()}
                return encoded[idx]

            encoded_map = {text: encoded_at(idx) for text, idx in unique_index.items()}
            for flat_index, result in temporal_results.items():
                branch = metadata_by_flat_index.get(flat_index)
                replacements = tuple(getattr(branch, "literal_scope_replacements", ()) or ())
                schedules_by_index[flat_index] = [
                    ScheduledPromptConditioning(
                        segment.end_step,
                        encoded_map[_restore_literal_scope_markers(segment.text, replacements)],
                    )
                    for segment in result.segments
                ]

        if chunk_break_results:
            for flat_index, segments in chunk_break_results.items():
                branch = metadata_by_flat_index.get(flat_index)
                replacements = tuple(getattr(branch, "literal_scope_replacements", ()) or ())
                full_prompt = _restore_literal_scope_markers(
                    str(getattr(branch, "text", "") or prompt_flat_list[flat_index]),
                    replacements,
                )

                if self._uses_a1111_clip_conditioning():
                    encoded = self._encode_conditioning_batch(
                        self.model,
                        [full_prompt],
                        forced_segments_by_prompt=[list(segments)],
                    )
                    if isinstance(encoded, dict):
                        combined = {key: value[0] for key, value in encoded.items()}
                    else:
                        combined = encoded[0]
                    schedules_by_index[flat_index] = [
                        ScheduledPromptConditioning(active_steps, combined)
                    ]
                    continue

                # Structured runtimes must own their BREAK policy explicitly.
                # SDXL and SD3 have multiple encoder channels plus a pooled/global
                # channel, so the parser must not invent a generic pooled average.
                # IMAGE_GEN-owned structured runtimes implement this hook per
                # encoder capability. Tensor-only CLIP/OpenCLIP runtimes retain
                # the historical generic segment-concatenation path below.
                structured_break_encoder = getattr(
                    self.model, "encode_chunk_break_conditioning", None
                )
                if callable(structured_break_encoder):
                    combined = structured_break_encoder(
                        list(segments),
                        full_prompt=full_prompt,
                    )
                    if not isinstance(combined, dict):
                        raise TypeError(
                            "Structured BREAK encoder must return a conditioning dict."
                        )
                    required = {"cross_attention", "pooled"}
                    missing = sorted(required - set(combined))
                    if missing:
                        raise KeyError(
                            "Structured BREAK conditioning is missing required field(s): "
                            f"{missing}."
                        )
                    undeclared = sorted(set(combined) - required)
                    if undeclared:
                        raise KeyError(
                            "Structured BREAK conditioning returned unsupported field(s): "
                            f"{undeclared}."
                        )
                    cross = combined["cross_attention"]
                    pooled = combined["pooled"]
                    if not isinstance(cross, torch.Tensor) or cross.ndim < 2:
                        raise TypeError(
                            "Structured BREAK cross_attention must be a sequence tensor."
                        )
                    if not isinstance(pooled, torch.Tensor) or pooled.ndim != 1:
                        raise TypeError(
                            "Structured BREAK pooled conditioning must be a rank-1 vector "
                            "before resolver batching."
                        )
                else:
                    chunk_texts = SdConditioning(
                        prompts=list(segments),
                        width=self.p.width,
                        height=self.p.height,
                        copy_from=self.prompts,
                    )
                    if hasattr(self.model, "get_learned_conditioning"):
                        encoded = self.model.get_learned_conditioning(chunk_texts)
                    elif hasattr(self.model, "encode"):
                        encoded = self.model.encode(chunk_texts)
                    else:
                        raise AttributeError("Model does not support conditioning")

                    if isinstance(encoded, dict):
                        raise ValueError(
                            "Structured BREAK requires an explicit model-family "
                            "encode_chunk_break_conditioning() contract; generic pooled "
                            "or multi-encoder reduction is intentionally not inferred."
                        )
                    if not isinstance(encoded, torch.Tensor):
                        raise TypeError(
                            "BREAK encoding must be a tensor or use the structured BREAK contract."
                        )
                    combined = torch.cat(
                        [encoded[index] for index in range(len(segments))], dim=0
                    )
                schedules_by_index[flat_index] = [
                    ScheduledPromptConditioning(active_steps, combined)
                ]

        res = []
        for batch_index, indexes in enumerate(res_indexes):
            batch_items = []
            metadata = metadata_batches[batch_index] if batch_index < len(metadata_batches) else []
            for branch_index, (i, weight) in enumerate(indexes):
                cond_schedule = schedules_by_index[i]
                branch = metadata[branch_index] if branch_index < len(metadata) else None
                temporal = temporal_results.get(i)
                active_by_step = tuple(bool(text.strip()) for text in temporal.per_step_text) if temporal else ()
                item = ComposableScheduledPromptConditioning(
                    cond_schedule, weight,
                    active_until_step=getattr(branch, "active_until_step", None),
                    hold_after_step=getattr(branch, "hold_after_step", False),
                    semantic_role=getattr(branch, "semantic_role", "text"),
                    group_operation_id=getattr(branch, "group_operation_id", None),
                    group_local_weight=getattr(branch, "group_local_weight", 1.0),
                    group_member_path=getattr(branch, "group_member_path", ()),
                    average_operation_id=getattr(branch, "average_operation_id", None),
                    average_local_weight=getattr(branch, "average_local_weight", 1.0),
                    average_branch_index=getattr(branch, "average_branch_index", None),
                    composition_operation_id=getattr(branch, "composition_operation_id", None),
                    composition_mode=getattr(branch, "composition_mode", ""),
                    composition_algorithm=getattr(branch, "composition_algorithm", ""),
                    composition_branch_index=getattr(branch, "composition_branch_index", None),
                    sequence_operation_id=getattr(branch, "sequence_operation_id", None),
                    sequence_local_weight=getattr(branch, "sequence_local_weight", 1.0),
                    sequence_item_index=getattr(branch, "sequence_item_index", None),
                    relation_operation_id=getattr(branch, "relation_operation_id", None),
                    relation_parent=getattr(branch, "relation_parent", ""),
                    relation_child=getattr(branch, "relation_child", ""),
                    owner_text=getattr(branch, "owner_text", ""),
                    syntax_origin=getattr(branch, "syntax_origin", ""),
                    source_span=getattr(branch, "source_span", (None, None)),
                    terminator_consumed=getattr(branch, "terminator_consumed", ""),
                    temporal_active_by_step=active_by_step,
                    temporal_source=getattr(branch, "temporal_source", ""),
                    chunk_break_segments=getattr(branch, "chunk_break_segments", ()),
                    chunk_break_count=getattr(branch, "chunk_break_count", 0),
                )
                batch_items.append(item)
            res.append(batch_items)

        return MulticondLearnedConditioning(shape=(len(self.prompts),), batch=res)



    def get_multicond_prompt_list(self):
        """Return branch indexes and encoder-visible text.

        Plain/A1111-only prompts retain the historical implementation exactly.
        Classic structured prompts and escaped structural literals route through
        the PPSR-02 Prompt IR compiler so punctuation is consumed before encode.
        """
        res_indexes = []
        metadata_batches = []

        prompt_indexes = {}
        prompt_flat_list = SdConditioning(
            prompts=self.prompts,
            width=self.p.width,
            height=self.p.height,
            copy_from=self.prompts,
        )
        prompt_flat_list.clear()
        re_AND = re.compile(r"\bAND\b")
        re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")

        for prompt_index, prompt in enumerate(self.prompts):
            prompt_text = str(prompt)
            plan = self._conditioning_plan_for(prompt_index, prompt_text)
            indexes = []
            branch_metadata = []

            if plan.lowering_required:
                compiled_branches = list(plan.branches)
                for branch in compiled_branches:
                    text = str(getattr(branch, "protected_text", "") or branch.text)
                    weight = float(branch.weight)
                    index = prompt_indexes.get(text, None)
                    if index is None:
                        index = len(prompt_flat_list)
                        prompt_flat_list.append(text)
                        prompt_indexes[text] = index
                    indexes.append((index, weight))
                    branch_metadata.append(branch)
            else:
                # Compatibility path: preserve historical whitespace and branch
                # weight behavior for ordinary A1111 AND/schedule prompts.
                subprompts = re_AND.split(prompt_text)
                for subprompt in subprompts:
                    match = re_weight.search(subprompt)
                    text, weight = match.groups() if match is not None else (subprompt, 1.0)
                    weight = float(weight) if weight is not None else 1.0
                    index = prompt_indexes.get(text, None)
                    if index is None:
                        index = len(prompt_flat_list)
                        prompt_flat_list.append(text)
                        prompt_indexes[text] = index
                    indexes.append((index, weight))
                    branch_metadata.append(None)

            res_indexes.append(indexes)
            metadata_batches.append(branch_metadata)

        self._last_branch_metadata = metadata_batches
        return res_indexes, prompt_flat_list, prompt_indexes

