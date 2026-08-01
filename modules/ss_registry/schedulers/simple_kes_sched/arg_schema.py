from typing import Any, Optional, List
from dataclasses import dataclass, asdict

ARG_SCHEMA_ORDER = ["type", "default", "short_desc", "long_desc","choices", "range", "required"]

@dataclass
class ArgSchema:
    type_: Any
    default: Any = None
    short_desc: str = ""
    long_desc: str = ""
    choices: Optional[list] = None
    range_: Optional[list] = None
    required: bool = False