from dataclasses import dataclass, asdict

@dataclass
class ArgSchema:
    type_: Any
    default: Any = None
    desc: str = ""
    choices: Optional[list] = None
    range_: Optional[list] = None
    required: bool = False

# Usage
'''
"steps": asdict(ArgSchema(
    type_=int,
    default=25,
    desc="Number of denoising steps",
    range_=[1, 150],
    required=True
)),
"sigma_min": asdict(ArgSchema(
    type_=Optional[float],
    default=0.2,
    desc="Minimum sigma value for noise schedule",
    range_=[0.0, 50.0]
)),
'''