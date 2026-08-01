from imagecore.parser.prompt_parser_class import PromptParserClass, MulticondLearnedConditioning, ScheduledPromptConditioning
#from prompt_parser_class import PromptParserClass, MulticondLearnedConditioning, ScheduledPromptConditioning
from imagecore.utils.model_check_pass import is_model #state, model
from typing import Dict, List, Tuple, Optional
import inspect

import torch

#for uses see model_prompt_generator
	
class ConditioningManager:
    def __init__(self, shared_state=None):
        self.state = shared_state
        self.conditioning = self.state.conditioning
        self.cfgm = self.state.cfgm
        self.p = self.state.p
        self._cond = None
        self._uncond = None
        if not self.state:
            # Get the caller function or class
            stack = inspect.stack()
            caller = stack[1]
            caller_name = caller.function
            caller_file = caller.filename
            caller_line = caller.lineno

            print(f"[ConditioningManager] ❌ No shared_state passed.")
            print(f"  ↳ Called by: {caller_name} in {caller_file}:{caller_line}")

            raise ValueError("SharedState was required but not provided.")
            
        self.model = self.state.cfgm.model   
        is_model(self.state, self.model)
        
        self.positive_prompt = self.state.p.positive_prompt
        self.negative_prompt = self.state.p.negative_prompt       
        self.steps = self.state.p.steps
        self.hires_steps = self.state.p.hires_steps
        
        self.prompts = self.positive_prompt, self.negative_prompt
        parsed_pos = PromptParserClass(
            shared_state=self.state,
            prompts=[self.positive_prompt],
            steps=self.steps,
            model=self.model,
            hires_steps=self.hires_steps
           
        )()

        parsed_neg = PromptParserClass(
            shared_state=self.state,
            prompts=[self.negative_prompt or ""],  # fallback for empty prompt
            steps=self.steps,
            model=self.model,
            hires_steps=self.hires_steps
            
        )()

             
        self.state.p.prompt_schedules = {
            "positive": parsed_pos.flat_list,
            "negative": parsed_neg.flat_list
        }

    

    def __call__(self, return_schedules: bool = False):
        return self.get_conditioning(return_schedules=return_schedules)
    
    def parse_and_get_conditioning(self, prompt: str) -> Tuple[MulticondLearnedConditioning, torch.Tensor]:
        parsed = PromptParserClass(shared_state=self.state, prompts=[prompt],steps=self.steps, model=self.model, hires_steps=self.hires_steps)()
        tensor = self.get_tensor(parsed.multicond)
        return parsed, tensor


    def get_all_tensor(self, schedule: MulticondLearnedConditioning) -> torch.Tensor:
        """
        Computes a weighted conditioning tensor of shape [B, N, D]
        from a MulticondLearnedConditioning object.
        """
        all_batch_tensors = []

        for b_idx, composable_group in enumerate(schedule.batch):
            embeddings = []
            weights = []
           

            for i, item in enumerate(composable_group):
                if not hasattr(item, "schedules") or not item.schedules:
                    raise AttributeError(f"[Error] item {i} in batch[{b_idx}] has no valid ScheduledPromptConditioning")

                sched = item.schedules[0]

                embedding = None  # Default

                if isinstance(sched.cond, torch.Tensor):
                    embedding = sched.cond

                elif isinstance(sched.cond, list):
                    inner_conds = []
                    for j, inner in enumerate(sched.cond):
                        if isinstance(inner, ScheduledPromptConditioning):
                            inner_conds.append(inner.cond)
                        elif isinstance(inner, torch.Tensor):
                            inner_conds.append(inner)
                        else:
                            raise TypeError(f"[Error] schedule[{b_idx}][{i}].cond[{j}] is not valid. Got: {type(inner)}")

                    if not inner_conds:
                        raise ValueError(f"[Error] schedule[{b_idx}][{i}] has empty inner conditioning list.")

                    embedding = torch.stack(inner_conds, dim=0).mean(dim=0)

                if embedding is not None:
                    embeddings.append(embedding)
                    weights.append(getattr(item, "weight", 1.0))
                else:
                    raise TypeError(f"[Error] schedule[{b_idx}][{i}] sched.cond is not tensor or list of tensors. Got: {type(sched.cond)}")

            # Convert and normalize
            weights_tensor = torch.tensor(weights, dtype=embeddings[0].dtype, device=embeddings[0].device)
            weight_sum = weights_tensor.sum()
            weights_tensor /= max(weight_sum, 1e-8)

            stacked = torch.stack(embeddings, dim=0)                  # [N, D]
            weighted = stacked * weights_tensor.view(-1, 1)           # [N, D]
            combined = weighted.sum(dim=0).unsqueeze(0)               # [1, D]
            all_batch_tensors.append(combined)

        # Final shape: [B, 1, D]
        out = torch.stack(all_batch_tensors, dim=0)
        #print(f"[Debug] Raw stacked tensor shape before squeeze: {out.shape}")
        if out.ndim == 4 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out        # [B, 1, D] 

    def get_conditioning(self, return_schedules: bool = False):
        parser_pos = PromptParserClass(
            shared_state=self.state,
            prompts=[self.positive_prompt],
            steps=self.steps,
            model=self.model,
            hires_steps=self.hires_steps
            
        )()

        parser_neg = PromptParserClass(
            shared_state=self.state,
            prompts=[self.negative_prompt or ""],
            steps=self.steps,
            model=self.model,
            hires_steps=self.hires_steps
            
        )()

        self._cond = self.get_all_tensor(parser_pos.multicond)
        self._uncond = self.get_all_tensor(parser_neg.multicond)

        # Update state
        self.state.p.cond = self._cond
        self.state.p.uncond = self._uncond
        self.state.p.prompt_schedules = {
            "positive": parser_pos.flat_list,
            "negative": parser_neg.flat_list
        }

        if return_schedules:
            return self._cond, self._uncond, self.state.p.prompt_schedules
        return self._cond, self._uncond

  
    @property
    def cond(self) -> torch.Tensor:
        if self._cond is None:
            print("[ConditioningManager] ⏳ cond not initialized, calling get_conditioning()...")
            self.get_conditioning()
        return self._cond


    @cond.setter
    def cond(self, value: torch.Tensor):
        self._cond = value
        self.state.p.cond = value  # keep state in sync


    @property
    def uncond(self) -> torch.Tensor:
        if self._uncond is None:
            print("[ConditioningManager] ⏳ uncond not initialized, calling get_conditioning()...")
            self.get_conditioning()
        return self._uncond

    @uncond.setter
    def uncond(self, value: torch.Tensor):
        self._uncond = value
        self.state.p.uncond = value  # keep state in sync

    @property
    def prompt_schedules(self) -> dict:
        return self.state.p.prompt_schedules or {"positive": [], "negative": []}
