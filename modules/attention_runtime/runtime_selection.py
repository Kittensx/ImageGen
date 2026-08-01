from __future__ import annotations


EXPLICIT_BACKENDS = frozenset({"eager", "sdpa", "xformers"})
AUTOMATIC_BACKEND_ORDER = ("xformers", "sdpa", "eager")


def backend_candidates(requested: str) -> list[str]:
    if requested in EXPLICIT_BACKENDS:
        return [requested]
    return list(AUTOMATIC_BACKEND_ORDER)
