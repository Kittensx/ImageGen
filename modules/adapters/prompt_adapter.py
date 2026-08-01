from __future__ import annotations

from modules.contracts import ConditioningOutput


class ExistingPromptParserAdapter:
    def __init__(self, parser_cls):
        self.parser_cls = parser_cls

    def encode(self, components, request, state=None) -> ConditioningOutput:
        if state is None:
            raise ValueError("ExistingPromptParserAdapter requires a state object.")

        state.cfgm.model = components.text_encoder
        state.p.positive_prompt = request.positive_prompt
        state.p.negative_prompt = request.negative_prompt
        state.p.steps = request.steps
        state.p.hires_steps = getattr(state.p, "hires_steps", request.steps)

        parser_pos = self.parser_cls(
            shared_state=state,
            prompts=[request.positive_prompt],
            steps=request.steps,
            model=components.text_encoder,
            hires_steps=getattr(state.p, "hires_steps", request.steps),
            **request.parser_kwargs,
        )()

        parser_neg = self.parser_cls(
            shared_state=state,
            prompts=[request.negative_prompt or ""],
            steps=request.steps,
            model=components.text_encoder,
            hires_steps=getattr(state.p, "hires_steps", request.steps),
            **request.parser_kwargs,
        )()

        # You may replace this with your own “flatten/merge multicond” logic
        cond = parser_pos.get_all_tensor(parser_pos.multicond) if hasattr(parser_pos, "get_all_tensor") else parser_pos.cond
        uncond = parser_neg.get_all_tensor(parser_neg.multicond) if hasattr(parser_neg, "get_all_tensor") else parser_neg.cond

        schedules = {
            "positive": getattr(parser_pos, "flat_list", []),
            "negative": getattr(parser_neg, "flat_list", []),
        }

        return ConditioningOutput(
            cond=cond,
            uncond=uncond,
            prompt_schedules=schedules,
        )