from __future__ import annotations

"""Declarative ImageGen-native prompt semantic policy.

Phase 04 freezes *intent* without cutting over production grammar.  The active
parser remains PPSR-09 compatible while later qualification phases implement
and promote the target surfaces recorded here.

Nothing in this module performs parsing, lowering, conditioning, or denoising.
It exists so tests, diagnostics, replay planning, and later implementation
phases share one stable vocabulary instead of duplicating semantic strings.
"""

from dataclasses import asdict, dataclass
from typing import Any


IMAGEGEN_UNIQUE_PROMPT_POLICY_VERSION = "image-gen-unique-prompt-semantics-v1"

# Stable algorithm identities.  Some are active today, some are target
# algorithms whose production activation is deliberately deferred.
BRANCH_AVERAGE_ALGORITHM = "branch_average_v1"
COHESIVE_GROUP_ALGORITHM = "shared_context_focus_v1"
COMPOSABLE_AND_ALGORITHM = "a1111_composable_guidance_v1"
CHUNK_BREAK_ALGORITHM = "encoder_chunk_break_v1"
BINDING_LOWERING_ALGORITHM = "bidirectional_pair_reinforcement_v1"
TARGET_BIND_SCOPE = "target_only"
SUBTREE_BIND_SCOPE = "subtree"
RELATION_ALGORITHM = "classic_structured_v1"


@dataclass(frozen=True)
class OperatorPolicy:
    semantic_id: str
    operation: str
    target_surface: str
    target_algorithm: str
    phase04_status: str
    current_surface: str = ""
    current_algorithm: str = ""
    current_semantic_id: str = ""
    production_enabled: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Phase-04 status vocabulary:
#   active_existing        active semantic retained from PPSR-09 / Classic
#   experimental_active    executable experiment, not production cutover
#   planned_surface_move   algorithm exists but target punctuation is deferred
#   planned_runtime        target semantics are specified but runtime is deferred
IMAGEGEN_NATIVE_OPERATOR_POLICY: tuple[OperatorPolicy, ...] = (
    OperatorPolicy(
        semantic_id="COHESIVE_GROUP",
        operation="Cohesive Groups",
        target_surface="{A, B}",
        target_algorithm=COHESIVE_GROUP_ALGORITHM,
        phase04_status="planned_surface_move",
        current_surface="⦃A, B⦄",
        current_algorithm=COHESIVE_GROUP_ALGORITHM,
        current_semantic_id="EXPERIMENTAL_GROUP",
        production_enabled=False,
        notes=(
            "Keep historical ordinary braces on branch_average_v1 until final qualification.",
            "White braces remain the PPSR-09 A/B delimiter before cutover.",
        ),
    ),
    OperatorPolicy(
        semantic_id="AVERAGE_SET",
        operation="Averaged Independent Conditions",
        target_surface="A || B || C",
        target_algorithm=BRANCH_AVERAGE_ALGORITHM,
        phase04_status="planned_surface_move",
        current_surface="{A, B, C}",
        current_algorithm=BRANCH_AVERAGE_ALGORITHM,
        current_semantic_id="GROUP",
        production_enabled=False,
        notes=(
            "Target operator is N-ary.",
            "Explicit average-branch weight syntax is intentionally deferred.",
            "Historical replay may continue to invoke branch_average_v1 directly.",
        ),
    ),
    OperatorPolicy(
        semantic_id="COMPOSABLE_AND",
        operation="Composable AND Conditions",
        target_surface="A AND B",
        target_algorithm=COMPOSABLE_AND_ALGORITHM,
        phase04_status="planned_runtime",
        current_surface="A AND B",
        current_algorithm="legacy_normalized_average_v1",
        current_semantic_id="LEGACY_CONJUNCTION",
        production_enabled=False,
        notes=(
            "Target composable AND must not route through branch_average_v1.",
            "The current textual AND split is retained until A1111 guidance composition is implemented.",
        ),
    ),
    OperatorPolicy(
        semantic_id="CHUNK_BREAK",
        operation="Chunk BREAK",
        target_surface="A BREAK B",
        target_algorithm=CHUNK_BREAK_ALGORITHM,
        phase04_status="planned_runtime",
        current_surface="A BREAK B",
        current_algorithm="literal_encoder_text",
        current_semantic_id="TEXT",
        production_enabled=False,
        notes=(
            "BREAK belongs inside the branch in which it appears.",
            "It must not split top-level AND or average composition.",
        ),
    ),
    OperatorPolicy(
        semantic_id="TARGET_BIND",
        operation="Local Binding",
        target_surface="modifier^target",
        target_algorithm=BINDING_LOWERING_ALGORITHM,
        phase04_status="active_existing",
        current_surface="modifier^target",
        current_algorithm=BINDING_LOWERING_ALGORITHM,
        current_semantic_id="BOUND_CONCEPT_TARGET_ONLY",
        production_enabled=True,
        notes=(
            "Scope is target_only.",
            "An explicit local bind is an inheritance barrier.",
        ),
    ),
    OperatorPolicy(
        semantic_id="SUBTREE_BIND",
        operation="Inherited Binding",
        target_surface="modifier*target",
        target_algorithm=BINDING_LOWERING_ALGORITHM,
        phase04_status="active_existing",
        current_surface="modifier*target",
        current_algorithm=BINDING_LOWERING_ALGORITHM,
        current_semantic_id="BOUND_CONCEPT_SUBTREE",
        production_enabled=True,
        notes=(
            "Scope is subtree.",
            "A child ^ or * bind starts a new local scope/barrier.",
        ),
    ),
    OperatorPolicy(
        semantic_id="RELATION",
        operation="Parent/Child Relationships",
        target_surface="property::value!",
        target_algorithm=RELATION_ALGORITHM,
        phase04_status="active_existing",
        current_surface="property::value!",
        current_algorithm=RELATION_ALGORITHM,
        current_semantic_id="RELATION",
        production_enabled=True,
        notes=("Close punctuation must not reach encoder text unless escaped.",),
    ),
    OperatorPolicy(
        semantic_id="OWNER_RELATION",
        operation="Owner Relationships",
        target_surface="owner:::property::value!, other::value!!",
        target_algorithm=RELATION_ALGORITHM,
        phase04_status="active_existing",
        current_surface="owner:::property::value!, other::value!!",
        current_algorithm=RELATION_ALGORITHM,
        current_semantic_id="OWNER_RELATION",
        production_enabled=True,
        notes=("Owner/deep relation semantics remain Classic-compatible.",),
    ),
)


# Tightest to loosest semantic interaction.  This is a policy contract for
# future parsers, not a Phase-04 parser implementation.
IMAGEGEN_INTERACTION_ORDER: tuple[str, ...] = (
    "ESCAPES",
    "LOCAL_BINDINGS",
    "STRUCTURAL_RELATIONS",
    "COHESIVE_GROUP_FOCUS",
    "CHUNK_BREAK_WITHIN_BRANCH",
    "BRANCH_COMPOSITION",
)

# The only unavoidable native Comfy collision identified by the Phase-04 policy
# is ordinary brace syntax.  Other aliases remain candidates subject to profile
# validation; this declaration does not register a Comfy profile.
COMFYUI_COHESIVE_GROUP_SAFE_ALIAS = "COHERE{A, B}"
COMFYUI_DYNAMIC_CHOICE_RESERVED_SURFACE = "{A|B}"
CROSS_PROFILE_IMAGEGEN_EXTENSION_CANDIDATES: tuple[str, ...] = (
    "^",
    "*",
    "||",
    "AND",
    "BREAK",
    "::",
    ":::",
)


HELP_OPERATION_SECTIONS: tuple[str, ...] = (
    "Normal Context",
    "Cohesive Groups",
    "Averaged Independent Conditions",
    "Composable AND Conditions",
    "Chunk BREAK",
    "Local Binding",
    "Inherited Binding",
    "Parent/Child Relationships",
    "Weights",
    "Schedules and Alternates",
    "Dynamic Prompts",
    "Escaping",
    "Profile Compatibility",
    "Replay Compatibility",
)

HELP_OPERATION_FIELDS: tuple[str, ...] = (
    "what it means",
    "what it does not mean",
    "example syntax per built-in profile",
    "rough conditioning description",
    "known model-family limitations",
    "replay behavior",
)


def operator_policy_by_semantic_id(semantic_id: str) -> OperatorPolicy:
    key = str(semantic_id or "").strip().upper()
    for item in IMAGEGEN_NATIVE_OPERATOR_POLICY:
        if item.semantic_id == key:
            return item
    raise KeyError(key)


def policy_snapshot() -> dict[str, Any]:
    return {
        "contract": IMAGEGEN_UNIQUE_PROMPT_POLICY_VERSION,
        "production_cutover_performed": False,
        "operators": [item.to_dict() for item in IMAGEGEN_NATIVE_OPERATOR_POLICY],
        "interaction_order": list(IMAGEGEN_INTERACTION_ORDER),
        "comfyui_collision_policy": {
            "reserved_dynamic_choice": COMFYUI_DYNAMIC_CHOICE_RESERVED_SURFACE,
            "cohesive_group_safe_alias": COMFYUI_COHESIVE_GROUP_SAFE_ALIAS,
            "cross_profile_candidates": list(CROSS_PROFILE_IMAGEGEN_EXTENSION_CANDIDATES),
        },
        "help": {
            "sections": list(HELP_OPERATION_SECTIONS),
            "required_fields": list(HELP_OPERATION_FIELDS),
        },
    }
