from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class HiresPairQualification:
    sampler_name: str
    scheduler_name: str
    qualification_id: str
    qualified: bool
    required_matrix_pair: bool
    compatibility_mode: str
    notes: tuple[str, ...] = ()

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "sampler_name": self.sampler_name,
            "scheduler_name": self.scheduler_name,
            "qualification_id": self.qualification_id,
            "qualified": bool(self.qualified),
            "required_matrix_pair": bool(self.required_matrix_pair),
            "compatibility_mode": self.compatibility_mode,
            "notes": list(self.notes),
        }


# Historical Phase 14M-4 reference matrix. This is retained for diagnostics and
# regression coverage only; it is not an execution allowlist. Runtime-compatible
# sampler/scheduler pairs do not need to be enumerated here before hires use.
_REFERENCE_PAIRS: dict[tuple[str, str], HiresPairQualification] = {
    ("dpmpp_2m", "standard_karras"): HiresPairQualification(
        "dpmpp_2m",
        "standard_karras",
        "phase14m4-dpmpp2m-standard-karras-v1",
        True,
        True,
        "fixed_steps",
        ("Standard Karras reference pair for A1111-compatible fixed-step hires.",),
    ),
    ("kes", "simple_kes"): HiresPairQualification(
        "kes",
        "simple_kes",
        "phase14m4-kes-simple-kes-v1",
        True,
        True,
        "extended_steps",
        ("KES is qualified independently; schedule validity does not imply A1111 parity.",),
    ),
    ("dpmpp_2m", "simple_kes"): HiresPairQualification(
        "dpmpp_2m",
        "simple_kes",
        "phase14m4-dpmpp2m-simple-kes-v1",
        True,
        True,
        "fixed_steps",
        ("Simple KES schedule consumed through the DPM++ 2M fixed-step contract.",),
    ),
    ("simple_euler", "standard_karras"): HiresPairQualification(
        "simple_euler",
        "standard_karras",
        "phase14m4-simple-euler-standard-karras-v1",
        True,
        False,
        "fixed_steps",
    ),
    ("simple_euler", "simple_kes"): HiresPairQualification(
        "simple_euler",
        "simple_kes",
        "phase14m4-simple-euler-simple-kes-v1",
        True,
        False,
        "fixed_steps",
    ),
    ("kes", "standard_karras"): HiresPairQualification(
        "kes",
        "standard_karras",
        "phase14m4-kes-standard-karras-clamped-v1",
        True,
        False,
        "fixed_steps",
        (
            "KES step expansion and tail metadata are intentionally clamped by the standard Karras scheduler.",
        ),
    ),
}


REQUIRED_PHASE14M4_MATRIX = frozenset(
    {
        ("dpmpp_2m", "standard_karras"),
        ("kes", "simple_kes"),
        ("dpmpp_2m", "simple_kes"),
    }
)


def qualified_hires_pairs() -> tuple[HiresPairQualification, ...]:
    """Return historically exercised/reference pairs, not an execution allowlist."""

    return tuple(_REFERENCE_PAIRS[key] for key in sorted(_REFERENCE_PAIRS))


def require_qualified_hires_pair(
    sampler_name: Any,
    scheduler_name: Any,
    *,
    compatibility: Mapping[str, Any] | None = None,
) -> HiresPairQualification:
    """Describe a hires pair without imposing a static qualification allowlist.

    Registry/plugin compatibility remains the runtime contract. Historical
    Phase 14M-4 pairs keep their original diagnostic IDs, while any other
    compatible pair is admitted and marked as open runtime selection.
    """

    key = (_normalize(sampler_name), _normalize(scheduler_name))
    if not key[0] or not key[1]:
        raise ValueError(
            "Hires refinement requires both a sampler and scheduler selection."
        )

    compatibility_report = dict(compatibility or {})
    if compatibility_report and not bool(compatibility_report.get("is_compatible", True)):
        reasons = "; ".join(str(item) for item in compatibility_report.get("reasons") or [])
        raise ValueError(
            "The requested hires sampler/scheduler pair is registry-incompatible: "
            + (reasons or "no compatibility reason was reported")
        )

    reference = _REFERENCE_PAIRS.get(key)
    if reference is not None:
        return reference

    compatibility_mode = str(
        compatibility_report.get("negotiated_pipeline_mode")
        or compatibility_report.get("compatibility_mode")
        or "runtime_compatible"
    ).strip() or "runtime_compatible"
    return HiresPairQualification(
        sampler_name=key[0],
        scheduler_name=key[1],
        qualification_id="open-runtime-selection-v1",
        qualified=True,
        required_matrix_pair=False,
        compatibility_mode=compatibility_mode,
        notes=(
            "No static hires sampler/scheduler allowlist is applied; runtime plugin compatibility governs execution.",
        ),
    )

