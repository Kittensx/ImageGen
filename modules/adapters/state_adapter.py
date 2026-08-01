from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class PipelineStateAdapter:
    aliases: Any
    cfgm: Any
    p: Any
    sched: Any
    samp: Any
    conditioning: Any
    d: Any
    extra: dict = field(default_factory=dict)

    @classmethod
    def create_empty(cls, device, dtype):
        return cls(
            aliases=SimpleNamespace(),
            cfgm=SimpleNamespace(model=None, tokenizer_path=None),
            p=SimpleNamespace(
                positive_prompt="",
                negative_prompt="",
                steps=20,
                hires_steps=20,
                batch_size=1,
                cfg_scale=7.0,
                width=512,
                height=512,
            ),
            sched=SimpleNamespace(sigmas=None, scheduler_fn=None),
            samp=SimpleNamespace(sampler_fn=None, sampler_name=None),
            conditioning=SimpleNamespace(cond=None, uncond=None, shape=None),
            d=SimpleNamespace(device=device, dtype=dtype),
        )