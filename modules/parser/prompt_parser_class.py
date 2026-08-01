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
    def __init__(self, schedules, weight=1.0):
        
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight
        
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
        hires_steps: Optional[int] = None
        
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
        #self.learned_conditioning.get_learned_cond()

    def __call__(self) -> ParsedPromptResult:
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
        """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
        For each prompt, the list is obtained by splitting the prompt using the AND separator.
        Each prompt may be split by AND into multiple weighted subprompts.
        Each subprompt keeps its own full schedule list so the active conditioning
        can be resolved later per sampler step.

        https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
        """

        res_indexes, prompt_flat_list, _ = self.get_multicond_prompt_list()

        learned_conditioning = self.get_learned_conditioning(model=self.model, prompts=prompt_flat_list, steps=self.steps, hires_steps=self.hires_steps)

        

        res = []
        for indexes in res_indexes:
            #res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
            batch_items = []
            for i, weight in indexes:
                cond_schedule = learned_conditioning[i]   # already a list[ScheduledPromptConditioning]
                item = ComposableScheduledPromptConditioning(cond_schedule, weight)
                batch_items.append(item)
            res.append(batch_items)

       
        return MulticondLearnedConditioning(shape=(len(self.prompts),), batch=res)



    def get_multicond_prompt_list(self):
        res_indexes = []

        prompt_indexes = {}
        prompt_flat_list = SdConditioning(prompts = self.prompts, width=self.p.width, height=self.p.height, copy_from=self.prompts)
        prompt_flat_list.clear()
        re_AND = re.compile(r"\bAND\b")
        re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")

        for prompt in self.prompts:
            subprompts = re_AND.split(prompt)

            indexes = []
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

            res_indexes.append(indexes)

        return res_indexes, prompt_flat_list, prompt_indexes
