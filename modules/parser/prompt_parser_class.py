from __future__ import annotations
from modules.parser.learned_conditioning import LearnedConditioning
from modules.parser.mixin import ScheduledPromptConditioning


import re
from collections import namedtuple
from lark import Lark
import random
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from lark import Lark


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
        self._last_branch_metadata = []
        #self.learned_conditioning.get_learned_cond()

    def _conditioning_plan_for(self, index: int, prompt: str):
        # Local imports avoid a package bootstrap cycle: prompt_parsers.__init__
        # registers the Legacy adapter, which itself imports PromptParserClass.
        from modules.prompt_parsers.compiler import compile_conditioning_plan
        from modules.prompt_parsers.ir import parse_prompt_ir

        prompt_ir = parse_prompt_ir(prompt, parser_namespace="legacy")
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
        _, flat_list, _ = self.get_multicond_prompt_list()

        return ParsedPromptResult(
            multicond=multicond,
            schedules=schedules,
            flat_list=flat_list
        )

    def get_learned_conditioning(self, model = None, prompts=None, steps = None, hires_steps=None, ):
        

        """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
        and the sampling step at which this condition is to be replaced by the next one.

        """
        #res: List[List[ScheduledPromptConditioning]]
        res = []
        model = model if model is not None else self.model
        prompts = prompts if prompts is not None else self.prompts
        steps = steps if steps is not None else self.steps
        hires_steps = hires_steps if hires_steps is not None else self.hires_steps

        prompt_schedules = self.learned_conditioning.get_learned_cond(prompts, steps, hires_steps)
        cache = {}

        for prompt, prompt_schedule in zip(prompts, prompt_schedules):

            cached = cache.get(prompt, None)
            if cached is not None:
                res.append(cached)
                continue

            texts = SdConditioning(
                prompts=[x[1] for x in prompt_schedule],
                width=self.p.width,
                height=self.p.height,
                copy_from=self.prompts
            )

            if hasattr(model, "get_learned_conditioning"):
                conds = model.get_learned_conditioning(texts)
            elif hasattr(model, "encode"):
                conds = model.encode(texts)
            else:
                raise AttributeError("Model does not support conditioning")


            cond_schedule = []
            for i, (end_at_step, _) in enumerate(prompt_schedule):
                if isinstance(conds, dict):
                    cond = {k: v[i] for k, v in conds.items()}
                else:
                    cond = conds[i]

                cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

            cache[prompt] = cond_schedule
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

        res_indexes, prompt_flat_list, _ = self.get_multicond_prompt_list()
        metadata_batches = list(self._last_branch_metadata or [])

        metadata_by_flat_index = {}
        for batch_index, indexes in enumerate(res_indexes):
            batch_meta = metadata_batches[batch_index] if batch_index < len(metadata_batches) else []
            for branch_index, (flat_index, _weight) in enumerate(indexes):
                branch = batch_meta[branch_index] if branch_index < len(batch_meta) else None
                metadata_by_flat_index.setdefault(flat_index, branch)

        temporal_results = {}
        non_temporal_indices = []
        active_steps = int(self.hires_steps or self.steps)
        for index, text in enumerate(prompt_flat_list):
            branch = metadata_by_flat_index.get(index)
            if branch is not None and bool(getattr(branch, "temporal_compiled", False)):
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
                model=self.model, prompts=non_texts, steps=self.steps, hires_steps=self.hires_steps
            )
            for flat_index, schedule in zip(non_temporal_indices, non_schedules):
                schedules_by_index[flat_index] = schedule

        if temporal_results:
            unique_texts = []
            unique_index = {}
            for result in temporal_results.values():
                for segment in result.segments:
                    if segment.text not in unique_index:
                        unique_index[segment.text] = len(unique_texts)
                        unique_texts.append(segment.text)
            texts = SdConditioning(
                prompts=unique_texts, width=self.p.width, height=self.p.height, copy_from=self.prompts
            )
            if hasattr(self.model, "get_learned_conditioning"):
                encoded = self.model.get_learned_conditioning(texts)
            elif hasattr(self.model, "encode"):
                encoded = self.model.encode(texts)
            else:
                raise AttributeError("Model does not support conditioning")

            def encoded_at(idx):
                if isinstance(encoded, dict):
                    return {key: value[idx] for key, value in encoded.items()}
                return encoded[idx]

            encoded_map = {text: encoded_at(idx) for text, idx in unique_index.items()}
            for flat_index, result in temporal_results.items():
                schedules_by_index[flat_index] = [
                    ScheduledPromptConditioning(segment.end_step, encoded_map[segment.text])
                    for segment in result.segments
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
                    text = str(branch.text)
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

