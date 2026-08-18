from __future__ import annotations

from typing import Any, Mapping


VP_SIGMA_DOMAIN = "vp_sigma"
FLOW_MATCH_DOMAIN = "flow_match"
KNOWN_SCHEDULE_DOMAINS = frozenset({VP_SIGMA_DOMAIN, FLOW_MATCH_DOMAIN})


def normalize_schedule_domain(value: Any, *, default: str = VP_SIGMA_DOMAIN) -> str:
    token = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "vp": VP_SIGMA_DOMAIN,
        "sigma": VP_SIGMA_DOMAIN,
        "vp_sigma": VP_SIGMA_DOMAIN,
        "flow": FLOW_MATCH_DOMAIN,
        "flowmatch": FLOW_MATCH_DOMAIN,
        "flow_match": FLOW_MATCH_DOMAIN,
    }
    normalized = aliases.get(token, token)
    if normalized not in KNOWN_SCHEDULE_DOMAINS:
        raise ValueError(
            f"Unknown scheduler domain {value!r}; expected one of {sorted(KNOWN_SCHEDULE_DOMAINS)}."
        )
    return normalized


def scheduler_domain_from_capabilities(capabilities: Mapping[str, Any] | None) -> str:
    """Return a scheduler's mathematical domain.

    Legacy scheduler descriptors predate this field and therefore default to
    ``vp_sigma``. New flow schedulers must opt in explicitly.
    """
    caps = dict(capabilities or {})
    return normalize_schedule_domain(caps.get("schedule_domain"), default=VP_SIGMA_DOMAIN)


def require_schedule_domain(capabilities: Mapping[str, Any] | None, required_domain: str) -> str:
    actual = scheduler_domain_from_capabilities(capabilities)
    required = normalize_schedule_domain(required_domain)
    if actual != required:
        raise ValueError(
            "Scheduler mathematical-domain mismatch: "
            f"selected scheduler provides {actual!r}, runtime requires {required!r}."
        )
    return actual
