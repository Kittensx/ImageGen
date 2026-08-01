"""Compatibility imports for the canonical adapter protocols.

New code should import from ``modules.contracts``. This module remains so older
adapter files and external experiments do not break during the phased migration.
"""

from modules.contracts import (
    PromptAdapterProtocol,
    SamplerAdapterProtocol,
    SchedulerAdapterProtocol,
)

# Historical name retained for callers that imported SamplerAdapter.
SamplerAdapter = SamplerAdapterProtocol

__all__ = [
    "PromptAdapterProtocol",
    "SchedulerAdapterProtocol",
    "SamplerAdapterProtocol",
    "SamplerAdapter",
]
