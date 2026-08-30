from __future__ import annotations

"""Compile Prompt IR into tensor-free conditioning intent.

PPSR-03 made ``{...}`` a real local composition scope. PPSR-04 extends the
same hierarchy to Classic relationship syntax: ``::`` relations and ``:::``
owner sequences become typed sequence operations. Relation members are
resolved locally before the resulting sequence participates in unrelated
``AND`` composition, so relation count never changes the outer branch weight.
"""

from dataclasses import dataclass, field, replace
import re
from typing import Any

from modules.prompt_parsers.numeric_semantics import NumericSemantic
from modules.prompt_parsers.temporal_semantics import contains_temporal_syntax

from modules.prompt_parsers.group_conditioning import (
    GroupDiagnostic,
    contains_group,
    expand_group_operation,
)
from modules.prompt_parsers.experimental_group_conditioning import (
    ExperimentalGroupDiagnostic,
    contains_experimental_group,
    expand_experimental_group_operation,
)
from modules.prompt_parsers.binding_semantics import (
    BindingDiagnostic,
    apply_inherited_bindings,
    binding_diagnostics,
    binding_phrase,
    child_inheritance,
    contains_binding,
    inherited_text,
)
from modules.prompt_parsers.ir import (
    AverageSet,
    Alternate,
    BoundConcept,
    ChunkBreak,
    Conjunction,
    ConjunctionBranch,
    ExperimentalGroup,
    Group,
    IRNode,
    Literal,
    LiteralTextScope,
    SemanticScope,
    OwnerSequence,
    Prompt,
    Quantity,
    PromptIR,
    Relation,
    Scheduled,
    Sequence,
    SequenceItemIR,
    Text,
    Weighted,
)

LEGACY_CONDITIONING_PLAN_CONTRACT_VERSION = "image-gen-conditioning-plan-v6"
CONDITIONING_PLAN_CONTRACT_VERSION = "image-gen-conditioning-plan-v7"
_ESCAPED_STRUCTURAL_RE = re.compile(r"\\[{}⦃⦄^*:!|\\]")
_CHUNK_BREAK_SENTINEL = "\x1eIMAGEGEN_CHUNK_BREAK\x1e"


@dataclass(frozen=True)
class RelationshipDiagnostic:
    operation_id: str
    syntax_origin: str
    owner: str = ""
    item_count: int = 0
    relation_count: int = 0
    source_items: tuple[str, ...] = field(default_factory=tuple)
    source_spans: tuple[tuple[int | None, int | None], ...] = field(default_factory=tuple)
    compiled_branch_texts: tuple[str, ...] = field(default_factory=tuple)
    raw_item_weights: tuple[float, ...] = field(default_factory=tuple)
    normalized_local_weights: tuple[float, ...] = field(default_factory=tuple)
    activity_windows: tuple[int | None, ...] = field(default_factory=tuple)
    terminators_consumed: tuple[str, ...] = field(default_factory=tuple)
    top_terminator_consumed: str = ""
    compatibility_aliases: tuple[str, ...] = field(default_factory=tuple)
    parent_scope: str = ""
    owner_composition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "syntax_origin": self.syntax_origin,
            "owner": self.owner,
            "item_count": int(self.item_count),
            "relation_count": int(self.relation_count),
            "source_items": list(self.source_items),
            "source_spans": [[start, end] for start, end in self.source_spans],
            "compiled_branch_texts": list(self.compiled_branch_texts),
            "raw_item_weights": [float(item) for item in self.raw_item_weights],
            "normalized_local_weights": [float(item) for item in self.normalized_local_weights],
            "activity_windows": list(self.activity_windows),
            "terminators_consumed": list(self.terminators_consumed),
            "top_terminator_consumed": self.top_terminator_consumed,
            "compatibility_aliases": list(self.compatibility_aliases),
            "parent_scope": self.parent_scope,
            "owner_composition": self.owner_composition,
        }


@dataclass(frozen=True)
class ConditioningBranch:
    text: str
    weight: float = 1.0
    active_until_step: int | None = None
    hold_after_step: bool = False
    semantic_role: str = "text"
    source_node_type: str = "text"
    group_operation_id: str | None = None
    group_local_weight: float = 1.0
    group_member_path: tuple[int, ...] = field(default_factory=tuple)
    average_operation_id: str | None = None
    average_local_weight: float = 1.0
    average_branch_index: int | None = None
    composition_operation_id: str | None = None
    composition_mode: str = ""
    composition_algorithm: str = ""
    composition_branch_index: int | None = None
    sequence_operation_id: str | None = None
    sequence_local_weight: float = 1.0
    sequence_item_index: int | None = None
    relation_operation_id: str | None = None
    relation_parent: str = ""
    relation_child: str = ""
    owner_text: str = ""
    syntax_origin: str = ""
    source_span: tuple[int | None, int | None] = (None, None)
    terminator_consumed: str = ""
    temporal_source: str = ""
    temporal_compiled: bool = False
    parent_scope: str = ""
    owner_composition: str = ""
    chunk_break_segments: tuple[str, ...] = field(default_factory=tuple)
    chunk_break_count: int = 0
    protected_text: str = ""
    literal_scope_replacements: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "weight": float(self.weight),
            "active_until_step": self.active_until_step,
            "hold_after_step": bool(self.hold_after_step),
            "semantic_role": self.semantic_role,
            "source_node_type": self.source_node_type,
            "group_operation_id": self.group_operation_id,
            "group_local_weight": float(self.group_local_weight),
            "group_member_path": list(self.group_member_path),
            "sequence_operation_id": self.sequence_operation_id,
            "sequence_local_weight": float(self.sequence_local_weight),
            "sequence_item_index": self.sequence_item_index,
            "relation_operation_id": self.relation_operation_id,
            "relation_parent": self.relation_parent,
            "relation_child": self.relation_child,
            "owner_text": self.owner_text,
            "syntax_origin": self.syntax_origin,
            "source_span": list(self.source_span),
            "terminator_consumed": self.terminator_consumed,
            "temporal_source": self.temporal_source,
            "temporal_compiled": bool(self.temporal_compiled),
            "parent_scope": self.parent_scope,
            "owner_composition": self.owner_composition,
        }
        # Preserve Phase-04/earlier exact replay payloads byte-for-byte at the
        # semantic level.  PPSR-09E metadata is serialized only when that
        # operator actually participates in the branch.
        if self.average_operation_id is not None:
            payload.update(
                {
                    "average_operation_id": self.average_operation_id,
                    "average_local_weight": float(self.average_local_weight),
                    "average_branch_index": self.average_branch_index,
                }
            )
        if self.composition_operation_id is not None:
            payload.update(
                {
                    "composition_operation_id": self.composition_operation_id,
                    "composition_mode": self.composition_mode,
                    "composition_algorithm": self.composition_algorithm,
                    "composition_branch_index": self.composition_branch_index,
                }
            )
        if self.chunk_break_count or self.chunk_break_segments:
            payload.update(
                {
                    "chunk_break_segments": list(self.chunk_break_segments),
                    "chunk_break_count": int(self.chunk_break_count),
                }
            )
        if self.protected_text or self.literal_scope_replacements:
            payload.update(
                {
                    "protected_text": self.protected_text,
                    "literal_scope_replacements": [list(item) for item in self.literal_scope_replacements],
                }
            )
        return payload


@dataclass(frozen=True)
class ConditioningPlan:
    source: str
    branches: tuple[ConditioningBranch, ...]
    lowering_required: bool
    fallbacks: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    group_diagnostics: tuple[GroupDiagnostic, ...] = field(default_factory=tuple)
    experimental_group_diagnostics: tuple[ExperimentalGroupDiagnostic, ...] = field(default_factory=tuple)
    binding_diagnostics: tuple[BindingDiagnostic, ...] = field(default_factory=tuple)
    relationship_diagnostics: tuple[RelationshipDiagnostic, ...] = field(default_factory=tuple)
    numeric_semantics: tuple[NumericSemantic, ...] = field(default_factory=tuple)
    composition_mode: str = "standard"
    composition_algorithm: str = ""
    composition_operation_id: str | None = None
    contract: str = CONDITIONING_PLAN_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": self.contract,
            "source": self.source,
            "lowering_required": bool(self.lowering_required),
            "branches": [item.to_dict() for item in self.branches],
            "fallbacks": list(self.fallbacks),
            "warnings": list(self.warnings),
            "group_diagnostics": [item.to_dict() for item in self.group_diagnostics],
            "relationship_diagnostics": [item.to_dict() for item in self.relationship_diagnostics],
            "numeric_semantics": [item.to_dict() for item in self.numeric_semantics],
        }
        if self.composition_mode not in {"", "standard", "legacy_normalized_average"}:
            payload.update(
                {
                    "composition_mode": self.composition_mode,
                    "composition_algorithm": self.composition_algorithm,
                    "composition_operation_id": self.composition_operation_id,
                }
            )
        # Preserve PPSR-08 exact-replay digests for prompts that do not use the
        # PPSR-09 experiment. New fields serialize only when the new semantics
        # are actually present.
        if self.experimental_group_diagnostics:
            payload["experimental_group_diagnostics"] = [
                item.to_dict() for item in self.experimental_group_diagnostics
            ]
        if self.binding_diagnostics:
            payload["binding_diagnostics"] = [item.to_dict() for item in self.binding_diagnostics]
        return payload


@dataclass
class _CompileContext:
    group_operation_counter: int = 0
    average_operation_counter: int = 0
    composition_operation_counter: int = 0
    sequence_operation_counter: int = 0
    fallbacks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    group_diagnostics: list[GroupDiagnostic] = field(default_factory=list)
    experimental_group_diagnostics: list[ExperimentalGroupDiagnostic] = field(default_factory=list)
    binding_diagnostics: list[BindingDiagnostic] = field(default_factory=list)
    relationship_diagnostics: list[RelationshipDiagnostic] = field(default_factory=list)

    def next_group_operation_id(self) -> str:
        value = f"groupop-{self.group_operation_counter}"
        self.group_operation_counter += 1
        return value

    def next_average_operation_id(self) -> str:
        value = f"avgop-{self.average_operation_counter}"
        self.average_operation_counter += 1
        return value

    def next_composition_operation_id(self) -> str:
        value = f"compop-{self.composition_operation_counter}"
        self.composition_operation_counter += 1
        return value

    def next_sequence_operation_id(self) -> str:
        value = f"seqop-{self.sequence_operation_counter}"
        self.sequence_operation_counter += 1
        return value


@dataclass(frozen=True)
class _SequenceMetadata:
    operation_id: str
    local_weight: float
    item_index: int
    relation_operation_id: str | None = None
    relation_parent: str = ""
    relation_child: str = ""
    owner_text: str = ""
    syntax_origin: str = ""
    source_span: tuple[int | None, int | None] = (None, None)
    terminator: str = ""
    parent_scope: str = ""
    owner_composition: str = ""



def _node_type(node: IRNode) -> str:
    if isinstance(node, OwnerSequence):
        return "owner_sequence"
    if isinstance(node, Relation):
        return "relation"
    return type(node).__name__.lower()



def _render_ir_node(node: IRNode, inherited_modifiers: tuple[str, ...] = ()) -> str:
    if isinstance(node, ChunkBreak):
        return _CHUNK_BREAK_SENTINEL
    if isinstance(node, LiteralTextScope):
        return str(node.value)
    if isinstance(node, SemanticScope):
        return _render_ir_node(node.node, inherited_modifiers)
    if isinstance(node, (Text, Literal)):
        value = str(node.value).replace(r"\!", "!")
        return inherited_text(value, inherited_modifiers).strip()
    if isinstance(node, BoundConcept):
        # Explicit local bindings are barriers: inherited subtree modifiers are
        # deliberately not injected into this target.
        return binding_phrase(node).strip()
    if isinstance(node, (Group, ExperimentalGroup)):
        return ", ".join(
            filter(None, (_render_ir_node(item, inherited_modifiers) for item in node.items))
        ).strip()
    if isinstance(node, Prompt):
        fragments: list[str] = []
        for item in node.parts:
            if isinstance(item, (Text, Literal)):
                value = str(item.value).replace(r"\!", "!")
                fragments.append(inherited_text(value, inherited_modifiers))
            else:
                fragments.append(_render_ir_node(item, inherited_modifiers))
        return "".join(fragments).strip()
    if isinstance(node, Relation):
        parent = _render_ir_node(node.parent, inherited_modifiers).strip(" ,")
        child_raw = getattr(node.child, "value", None)
        descendant_modifiers = child_inheritance(node.parent, inherited_modifiers)
        child = _render_ir_node(node.child, descendant_modifiers).strip(" ,")
        # Closed-sequence terminators belong to the relationship, not encoder
        # text. A backslash-protected bang remains literal user text.
        if not isinstance(node.child, Literal) and not (
            isinstance(child_raw, str) and child_raw.rstrip().endswith(r"\!")
        ):
            removed = 0
            while child.endswith("!") and removed < 2:
                child = child[:-1].rstrip()
                removed += 1
        return ", ".join(item for item in (parent, child) if item)
    if isinstance(node, Weighted):
        return _render_ir_node(node.node, inherited_modifiers)
    if isinstance(node, OwnerSequence):
        owner = _render_ir_node(node.owner, inherited_modifiers).strip(" ,")
        descendant_modifiers = child_inheritance(node.owner, inherited_modifiers)
        return "; ".join(
            ", ".join(
                part
                for part in (owner, _render_ir_node(item.node, descendant_modifiers).strip(" ,"))
                if part
            )
            for item in node.items
        ).strip()
    if isinstance(node, Sequence):
        return ", ".join(
            filter(None, (_render_ir_node(item.node, inherited_modifiers) for item in node.items))
        ).strip()
    if isinstance(node, AverageSet):
        return " || ".join(_render_ir_node(item, inherited_modifiers) for item in node.branches)
    if isinstance(node, Conjunction):
        return " AND ".join(_render_ir_node(item.node, inherited_modifiers) for item in node.branches)
    if isinstance(node, (Scheduled, Alternate)):
        return node.value
    return str(node)


def render_ir_node(node: IRNode) -> str:
    return _render_ir_node(node, ())


_LITERAL_SCOPE_MARKER_PREFIX = "__IG_LITERAL_RUNTIME_"
_AVERAGE_COMBINATION_LIMIT = 256


def _protect_literal_scopes_for_runtime(
    node: IRNode,
    *,
    counter: list[int] | None = None,
) -> tuple[IRNode, tuple[tuple[str, str], ...]]:
    """Replace literal quote scopes with opaque scheduler-safe markers.

    The final encoder text is restored before model.encode(), but the opaque
    form is what the legacy scheduling/alternate parser sees. This guarantees
    that syntax written inside double quotes remains text all the way through
    runtime lowering instead of being reinterpreted after PromptIR compilation.
    """

    state = counter if counter is not None else [0]
    replacements: list[tuple[str, str]] = []

    def visit(current: IRNode) -> IRNode:
        if isinstance(current, LiteralTextScope):
            marker = f"{_LITERAL_SCOPE_MARKER_PREFIX}{state[0]:04d}__"
            state[0] += 1
            replacements.append((marker, str(current.value)))
            return Text(source_text=current.source_text, value=marker)
        if isinstance(current, SemanticScope):
            return replace(current, node=visit(current.node))
        if isinstance(current, Group):
            return replace(current, items=tuple(visit(item) for item in current.items))
        if isinstance(current, ExperimentalGroup):
            return replace(current, items=tuple(visit(item) for item in current.items))
        if isinstance(current, AverageSet):
            return replace(current, branches=tuple(visit(item) for item in current.branches))
        if isinstance(current, Prompt):
            return replace(current, parts=tuple(visit(item) for item in current.parts))
        if isinstance(current, Relation):
            return replace(current, parent=visit(current.parent), child=visit(current.child))
        if isinstance(current, Weighted):
            return replace(current, node=visit(current.node))
        if isinstance(current, Quantity):
            return replace(current, node=visit(current.node))
        if isinstance(current, Sequence):
            return replace(
                current,
                items=tuple(replace(item, node=visit(item.node)) for item in current.items),
            )
        if isinstance(current, OwnerSequence):
            return replace(
                current,
                owner=visit(current.owner),
                items=tuple(replace(item, node=visit(item.node)) for item in current.items),
            )
        if isinstance(current, Conjunction):
            return replace(
                current,
                branches=tuple(replace(branch, node=visit(branch.node)) for branch in current.branches),
            )
        return current

    protected = visit(node)
    return protected, tuple(replacements)


def _restore_literal_markers(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    text = str(value or "")
    for marker, literal in replacements:
        text = text.replace(str(marker), str(literal))
    return text


def _restore_branch_literal_scopes(
    branch: ConditioningBranch,
    replacements: tuple[tuple[str, str], ...],
) -> ConditioningBranch:
    if not replacements:
        return branch
    protected_text = str(branch.text or "")
    return replace(
        branch,
        text=_restore_literal_markers(protected_text, replacements),
        protected_text=protected_text,
        literal_scope_replacements=tuple(replacements),
        chunk_break_segments=tuple(
            _restore_literal_markers(item, replacements)
            for item in tuple(branch.chunk_break_segments or ())
        ),
    )


def _contains_average_set(node: IRNode) -> bool:
    if isinstance(node, AverageSet):
        return True
    if isinstance(node, SemanticScope):
        return _contains_average_set(node.node)
    if isinstance(node, (Group, ExperimentalGroup)):
        return any(_contains_average_set(item) for item in node.items)
    if isinstance(node, Prompt):
        return any(_contains_average_set(item) for item in node.parts)
    if isinstance(node, Relation):
        return _contains_average_set(node.parent) or _contains_average_set(node.child)
    if isinstance(node, Weighted):
        return _contains_average_set(node.node)
    if isinstance(node, Quantity):
        return _contains_average_set(node.node)
    if isinstance(node, Sequence):
        return any(_contains_average_set(item.node) for item in node.items)
    if isinstance(node, OwnerSequence):
        return _contains_average_set(node.owner) or any(
            _contains_average_set(item.node) for item in node.items
        )
    if isinstance(node, Conjunction):
        return any(_contains_average_set(branch.node) for branch in node.branches)
    return False


def _count_composable_conjunctions(node: IRNode) -> int:
    count = 0
    if isinstance(node, Conjunction) and node.composition_mode.startswith("a1111_composable"):
        count += 1
    if isinstance(node, SemanticScope):
        return count + _count_composable_conjunctions(node.node)
    if isinstance(node, (Group, ExperimentalGroup)):
        return count + sum(_count_composable_conjunctions(item) for item in node.items)
    if isinstance(node, Prompt):
        return count + sum(_count_composable_conjunctions(item) for item in node.parts)
    if isinstance(node, Relation):
        return count + _count_composable_conjunctions(node.parent) + _count_composable_conjunctions(node.child)
    if isinstance(node, Weighted):
        return count + _count_composable_conjunctions(node.node)
    if isinstance(node, Quantity):
        return count + _count_composable_conjunctions(node.node)
    if isinstance(node, Sequence):
        return count + sum(_count_composable_conjunctions(item.node) for item in node.items)
    if isinstance(node, OwnerSequence):
        return count + _count_composable_conjunctions(node.owner) + sum(
            _count_composable_conjunctions(item.node) for item in node.items
        )
    if isinstance(node, AverageSet):
        return count + sum(_count_composable_conjunctions(item) for item in node.branches)
    return count


def _cross_node_variants(
    left: list[tuple[list[IRNode], float]],
    right: list[tuple[IRNode, float]],
) -> list[tuple[list[IRNode], float]]:
    if len(left) * len(right) > _AVERAGE_COMBINATION_LIMIT:
        raise ValueError(
            f"AverageSet expansion exceeds {_AVERAGE_COMBINATION_LIMIT} complete prompt combinations."
        )
    output: list[tuple[list[IRNode], float]] = []
    for left_nodes, left_weight in left:
        for right_node, right_weight in right:
            output.append(([*left_nodes, right_node], float(left_weight) * float(right_weight)))
    return output


def _expand_average_variants(node: IRNode) -> list[tuple[IRNode, float]]:
    """Cartesian-expand local AverageSet axes into complete semantic branches."""

    if isinstance(node, AverageSet):
        raw = tuple(node.local_weights or tuple(1.0 for _ in node.branches))
        if len(raw) != len(node.branches):
            raise ValueError("AverageSet local_weights must match the branch count.")
        total = float(sum(float(value) for value in raw))
        if abs(total) <= 1e-12:
            raise ValueError("AverageSet local weights must not sum to zero.")
        output: list[tuple[IRNode, float]] = []
        for branch, raw_weight in zip(node.branches, raw):
            branch_weight = float(raw_weight) / total
            for expanded, nested_weight in _expand_average_variants(branch):
                output.append((expanded, branch_weight * float(nested_weight)))
        return output
    if isinstance(node, SemanticScope):
        return [
            (replace(node, node=child), weight)
            for child, weight in _expand_average_variants(node.node)
        ]
    if isinstance(node, Prompt):
        combinations: list[tuple[list[IRNode], float]] = [([], 1.0)]
        for part in node.parts:
            combinations = _cross_node_variants(combinations, _expand_average_variants(part))
        return [
            (replace(node, parts=tuple(parts)), weight)
            for parts, weight in combinations
        ]
    if isinstance(node, Group):
        combinations: list[tuple[list[IRNode], float]] = [([], 1.0)]
        for item in node.items:
            combinations = _cross_node_variants(combinations, _expand_average_variants(item))
        return [(replace(node, items=tuple(items)), weight) for items, weight in combinations]
    if isinstance(node, ExperimentalGroup):
        combinations: list[tuple[list[IRNode], float]] = [([], 1.0)]
        for item in node.items:
            combinations = _cross_node_variants(combinations, _expand_average_variants(item))
        return [(replace(node, items=tuple(items)), weight) for items, weight in combinations]
    if isinstance(node, Relation):
        output: list[tuple[IRNode, float]] = []
        for parent, parent_weight in _expand_average_variants(node.parent):
            for child, child_weight in _expand_average_variants(node.child):
                output.append((replace(node, parent=parent, child=child), parent_weight * child_weight))
        return output
    if isinstance(node, Weighted):
        return [(replace(node, node=child), weight) for child, weight in _expand_average_variants(node.node)]
    if isinstance(node, Quantity):
        return [(replace(node, node=child), weight) for child, weight in _expand_average_variants(node.node)]
    if isinstance(node, Sequence):
        combinations: list[tuple[list[IRNode], float]] = [([], 1.0)]
        for item in node.items:
            variants = [(child, weight) for child, weight in _expand_average_variants(item.node)]
            expanded_items: list[tuple[IRNode, float]] = [
                (replace(item, node=child), weight) for child, weight in variants
            ]
            combinations = _cross_node_variants(combinations, expanded_items)
        return [
            (replace(node, items=tuple(items)), weight)
            for items, weight in combinations
        ]
    if isinstance(node, OwnerSequence):
        output: list[tuple[IRNode, float]] = []
        for owner, owner_weight in _expand_average_variants(node.owner):
            combinations: list[tuple[list[IRNode], float]] = [([], owner_weight)]
            for item in node.items:
                variants = [
                    (replace(item, node=child), weight)
                    for child, weight in _expand_average_variants(item.node)
                ]
                combinations = _cross_node_variants(combinations, variants)
            for items, weight in combinations:
                output.append((replace(node, owner=owner, items=tuple(items)), weight))
        return output
    if isinstance(node, Conjunction):
        # Mixed average/composable guidance is rejected before this helper.
        return [(node, 1.0)]
    return [(node, 1.0)]


def _compile_average_composite(node: IRNode, ctx: _CompileContext) -> list[ConditioningBranch]:
    if _count_composable_conjunctions(node):
        raise ValueError(
            "PPSR-09E does not define mixed top-level || and composable AND semantics in one semantic scope. "
            "Use one branch-composition operation at a time during qualification."
        )
    variants = _expand_average_variants(node)
    if len(variants) > _AVERAGE_COMBINATION_LIMIT:
        raise ValueError(
            f"AverageSet expansion exceeds {_AVERAGE_COMBINATION_LIMIT} complete prompt combinations."
        )
    total = float(sum(weight for _, weight in variants))
    if abs(total) <= 1e-12:
        raise ValueError("AverageSet expanded branch weights sum to zero.")
    operation_id = ctx.next_average_operation_id()
    output: list[ConditioningBranch] = []
    for branch_index, (branch_node, weight) in enumerate(variants):
        local_weight = float(weight) / total
        nested = _compile_node(branch_node, ctx)
        for item in nested:
            output.append(
                replace(
                    item,
                    average_operation_id=operation_id,
                    average_local_weight=local_weight,
                    average_branch_index=branch_index,
                    semantic_role=(
                        item.semantic_role
                        if item.semantic_role not in {"text", "literal"}
                        else "average_set_member"
                    ),
                )
            )
    return output


def _expand_single_composable_context(node: IRNode) -> Conjunction | None:
    """Lift one nested single-quoted composable AND into complete prompt branches."""

    if isinstance(node, Conjunction) and node.composition_mode.startswith("a1111_composable"):
        return node
    count = _count_composable_conjunctions(node)
    if count == 0:
        return None
    if count > 1:
        raise ValueError(
            "Multiple independent local composable AND scopes in one prompt are not yet qualified."
        )

    def variants(current: IRNode) -> list[tuple[IRNode, float]]:
        if isinstance(current, Conjunction) and current.composition_mode.startswith("a1111_composable"):
            return [(branch.node, float(branch.weight)) for branch in current.branches]
        if isinstance(current, SemanticScope):
            return [(replace(current, node=child), weight) for child, weight in variants(current.node)]
        if isinstance(current, Prompt):
            combos: list[tuple[list[IRNode], float]] = [([], 1.0)]
            for part in current.parts:
                combos = _cross_node_variants(combos, variants(part))
            return [(replace(current, parts=tuple(parts)), weight) for parts, weight in combos]
        return [(current, 1.0)]

    expanded = variants(node)
    if len(expanded) <= 1:
        return None
    return Conjunction(
        source_text=node.source_text,
        branches=tuple(
            ConjunctionBranch(node=child, weight=weight, source_text=render_ir_node(child))
            for child, weight in expanded
        ),
        composition_mode="a1111_composable_guidance_v1",
        algorithm="a1111_composable_guidance_v1",
    )


def _consume_chunk_break_sentinels(branch: ConditioningBranch) -> ConditioningBranch:
    text = str(branch.text or "")
    if _CHUNK_BREAK_SENTINEL not in text:
        return branch
    raw_segments = text.split(_CHUNK_BREAK_SENTINEL)
    segments = tuple(segment.strip(" ,") for segment in raw_segments)
    if not all(segments):
        raise ValueError("BREAK requires non-empty encoder text on both sides of each boundary.")
    clean_text = " ".join(segments).strip()
    return replace(
        branch,
        text=clean_text,
        chunk_break_segments=segments,
        chunk_break_count=max(0, len(segments) - 1),
    )


def _apply_sequence_metadata(branch: ConditioningBranch, metadata: _SequenceMetadata) -> ConditioningBranch:
    return replace(
        branch,
        sequence_operation_id=metadata.operation_id,
        sequence_local_weight=float(metadata.local_weight),
        sequence_item_index=int(metadata.item_index),
        relation_operation_id=metadata.relation_operation_id,
        relation_parent=metadata.relation_parent,
        relation_child=metadata.relation_child,
        owner_text=metadata.owner_text,
        syntax_origin=metadata.syntax_origin,
        source_span=metadata.source_span,
        terminator_consumed=metadata.terminator,
        parent_scope=metadata.parent_scope,
        owner_composition=metadata.owner_composition,
    )



def _compile_group_operation(
    node: IRNode,
    ctx: _CompileContext,
    *,
    semantic_role: str,
    source_node_type: str,
    outer_weight: float = 1.0,
    active_until_step: int | None = None,
    hold_after_step: bool = False,
    sequence_metadata: _SequenceMetadata | None = None,
) -> list[ConditioningBranch]:
    operation_id = ctx.next_group_operation_id()
    prepared_node = apply_inherited_bindings(node) if contains_binding(node) else node
    expansion = expand_group_operation(
        prepared_node,
        operation_id=operation_id,
        render_node=render_ir_node,
    )
    ctx.group_diagnostics.extend(expansion.diagnostics)

    if expansion.fallback_used:
        ctx.fallbacks.append("group_deterministic_safe_flatten")
        if expansion.warning:
            ctx.warnings.append(expansion.warning)
        branch = ConditioningBranch(
            text=render_ir_node(node),
            weight=float(outer_weight),
            active_until_step=active_until_step,
            hold_after_step=hold_after_step,
            semantic_role="group_safe_flatten",
            source_node_type=source_node_type,
        )
        return [_apply_sequence_metadata(branch, sequence_metadata)] if sequence_metadata else [branch]

    # A singleton group must be conditioning-equivalent to plain text. Keep
    # diagnostics for observability but do not force a one-member averaging
    # operation into the runtime.
    singleton_only = bool(expansion.diagnostics) and all(
        item.member_count == 1 for item in expansion.diagnostics
    ) and len(expansion.variants) == 1

    output: list[ConditioningBranch] = []
    for variant in expansion.variants:
        text = str(variant.text).strip()
        if not text:
            continue
        branch = ConditioningBranch(
            text=text,
            weight=float(outer_weight),
            active_until_step=active_until_step,
            hold_after_step=hold_after_step,
            semantic_role="text" if singleton_only else semantic_role,
            source_node_type="text" if singleton_only else source_node_type,
            group_operation_id=None if singleton_only else operation_id,
            group_local_weight=1.0 if singleton_only else float(variant.local_weight),
            group_member_path=() if singleton_only else tuple(variant.member_path),
        )
        output.append(_apply_sequence_metadata(branch, sequence_metadata) if sequence_metadata else branch)
    return output



def _compile_experimental_group_operation(
    node: IRNode,
    ctx: _CompileContext,
    *,
    semantic_role: str,
    source_node_type: str,
    outer_weight: float = 1.0,
    active_until_step: int | None = None,
    hold_after_step: bool = False,
    sequence_metadata: _SequenceMetadata | None = None,
) -> list[ConditioningBranch]:
    operation_id = ctx.next_group_operation_id()
    prepared_node = apply_inherited_bindings(node) if contains_binding(node) else node
    expansion = expand_experimental_group_operation(
        prepared_node,
        operation_id=operation_id,
        render_node=render_ir_node,
    )
    ctx.experimental_group_diagnostics.extend(expansion.diagnostics)

    if expansion.fallback_used:
        ctx.fallbacks.append("experimental_group_deterministic_safe_flatten")
        if expansion.warning:
            ctx.warnings.append(expansion.warning)
        branch = ConditioningBranch(
            text=render_ir_node(node),
            weight=float(outer_weight),
            active_until_step=active_until_step,
            hold_after_step=hold_after_step,
            semantic_role="experimental_group_safe_flatten",
            source_node_type=source_node_type,
        )
        return [_apply_sequence_metadata(branch, sequence_metadata)] if sequence_metadata else [branch]

    singleton_only = bool(expansion.diagnostics) and all(
        item.member_count == 1 for item in expansion.diagnostics
    ) and len(expansion.variants) == 1

    output: list[ConditioningBranch] = []
    for variant in expansion.variants:
        text = str(variant.text).strip()
        if not text:
            continue
        branch = ConditioningBranch(
            text=text,
            weight=float(outer_weight),
            active_until_step=active_until_step,
            hold_after_step=hold_after_step,
            semantic_role="text" if singleton_only else semantic_role,
            source_node_type="text" if singleton_only else source_node_type,
            group_operation_id=None if singleton_only else operation_id,
            group_local_weight=1.0 if singleton_only else float(variant.local_weight),
            group_member_path=() if singleton_only else tuple(variant.member_path),
        )
        output.append(_apply_sequence_metadata(branch, sequence_metadata) if sequence_metadata else branch)
    return output


def _normalized_weights(items: tuple[SequenceItemIR, ...]) -> tuple[float, ...]:
    raw = [float(item.weight) for item in items]
    total = float(sum(raw))
    if not raw:
        return ()
    if abs(total) <= 1e-12:
        # Preserve typed raw weights in runtime metadata; diagnostic normalization
        # cannot be meaningful for a zero-sum sequence.
        return tuple(raw)
    return tuple(float(item) / total for item in raw)



def _relationship_parts(item_node: IRNode) -> tuple[str, str]:
    if not isinstance(item_node, Relation):
        return "", ""
    return render_ir_node(item_node.parent).strip(" ,"), render_ir_node(item_node.child).strip(" ,")




def _compile_terminal_attachment_owner_sequence(
    node: OwnerSequence,
    owner_sequence: Sequence,
    ctx: _CompileContext,
    *,
    operation_id: str,
) -> list[ConditioningBranch]:
    """Compile an ungrouped ``a:b:c:::...`` owner with terminal attachment.

    The legacy ``:`` chain remains one equal/local-weight sequence composition,
    but only its terminal item is the owner for the ``:::`` body.  Conceptually:

        a : b : c ::: X
        [a] [b] [c -> X]

    Wrapping the owner chain in ``{...}`` bypasses this helper; group expansion
    then attaches X to every locally weighted owner variant so the *entire*
    composition acts as parent.
    """
    if not owner_sequence.items:
        return []

    syntax_origin = "classic_owner_sequence_terminal_attachment"
    terminal_index = len(owner_sequence.items) - 1
    terminal_item = owner_sequence.items[terminal_index]
    terminal_node = terminal_item.node
    terminal_text = render_ir_node(terminal_node).strip(" ,")
    owner_composition = render_ir_node(owner_sequence).strip(" ,")
    branches: list[ConditioningBranch] = []

    # Prefix members retain the legacy sequence's local weights unchanged.
    for index, owner_item in enumerate(owner_sequence.items[:-1]):
        metadata = _SequenceMetadata(
            operation_id=operation_id,
            local_weight=float(owner_item.weight),
            item_index=index,
            owner_text=terminal_text,
            syntax_origin=syntax_origin,
            source_span=(owner_item.source_start, owner_item.source_end),
            terminator=str(owner_item.terminator or ""),
            parent_scope="terminal_attachment",
            owner_composition=owner_composition,
        )
        if contains_experimental_group(owner_item.node):
            branches.extend(
                _compile_experimental_group_operation(
                    owner_item.node,
                    ctx,
                    semantic_role="owner_sequence_prefix_experimental_group",
                    source_node_type="owner_sequence",
                    active_until_step=owner_item.active_until_step,
                    sequence_metadata=metadata,
                )
            )
        elif contains_group(owner_item.node):
            branches.extend(
                _compile_group_operation(
                    owner_item.node,
                    ctx,
                    semantic_role="owner_sequence_prefix_group",
                    source_node_type="owner_sequence",
                    active_until_step=owner_item.active_until_step,
                    sequence_metadata=metadata,
                )
            )
        else:
            text = render_ir_node(owner_item.node).strip()
            if text:
                branches.append(
                    _apply_sequence_metadata(
                        ConditioningBranch(
                            text=text,
                            active_until_step=owner_item.active_until_step,
                            semantic_role="owner_sequence_prefix",
                            source_node_type="owner_sequence",
                        ),
                        metadata,
                    )
                )

    # The ::: body is a relation sequence owned by the *terminal* chain item.
    # Split the terminal item's local influence across body items so the whole
    # terminal subtree still owns exactly one sequence-member share.
    body_weights = _normalized_weights(node.items)
    terminal_weight = float(terminal_item.weight)
    for body_index, item in enumerate(node.items):
        body_share = float(body_weights[body_index]) if body_weights else 1.0
        local_weight = terminal_weight * body_share
        contextual = Relation(parent=terminal_node, child=item.node)
        relation_parent, relation_child = _relationship_parts(item.node)
        metadata = _SequenceMetadata(
            operation_id=operation_id,
            local_weight=local_weight,
            item_index=terminal_index,
            relation_operation_id=f"{operation_id}.terminal-rel-{body_index}",
            relation_parent=relation_parent,
            relation_child=relation_child,
            owner_text=terminal_text,
            syntax_origin=syntax_origin,
            source_span=(item.source_start, item.source_end),
            terminator=str(item.terminator or ""),
            parent_scope="terminal_attachment",
            owner_composition=owner_composition,
        )
        active_until = item.active_until_step
        if terminal_item.active_until_step is not None:
            active_until = terminal_item.active_until_step
        if contains_experimental_group(contextual):
            branches.extend(
                _compile_experimental_group_operation(
                    contextual,
                    ctx,
                    semantic_role="terminal_owner_relation_experimental_group_variant",
                    source_node_type="owner_sequence",
                    active_until_step=active_until,
                    sequence_metadata=metadata,
                )
            )
        elif contains_group(contextual):
            branches.extend(
                _compile_group_operation(
                    contextual,
                    ctx,
                    semantic_role="terminal_owner_relation_group_variant",
                    source_node_type="owner_sequence",
                    active_until_step=active_until,
                    sequence_metadata=metadata,
                )
            )
        else:
            text = render_ir_node(contextual).strip()
            if text:
                branches.append(
                    _apply_sequence_metadata(
                        ConditioningBranch(
                            text=text,
                            active_until_step=active_until,
                            semantic_role="terminal_owner_relation_item",
                            source_node_type="owner_sequence",
                        ),
                        metadata,
                    )
                )

    owner_raw_weights = tuple(float(item.weight) for item in owner_sequence.items)
    owner_total = float(sum(owner_raw_weights))
    owner_normalized = tuple(
        weight / owner_total for weight in owner_raw_weights
    ) if owner_raw_weights and abs(owner_total) > 1e-12 else owner_raw_weights
    ctx.relationship_diagnostics.append(
        RelationshipDiagnostic(
            operation_id=operation_id,
            syntax_origin=syntax_origin,
            owner=terminal_text,
            item_count=len(owner_sequence.items),
            relation_count=len(node.items),
            source_items=tuple(str(item.source_text or "") for item in owner_sequence.items),
            source_spans=tuple((item.source_start, item.source_end) for item in owner_sequence.items),
            compiled_branch_texts=tuple(item.text for item in branches),
            raw_item_weights=owner_raw_weights,
            normalized_local_weights=owner_normalized,
            activity_windows=tuple(item.active_until_step for item in owner_sequence.items),
            terminators_consumed=tuple(str(item.terminator or "") for item in node.items),
            top_terminator_consumed=str(node.top_terminator or ""),
            parent_scope="terminal_attachment",
            owner_composition=owner_composition,
        )
    )
    return branches


def _compile_sequence(
    node: Sequence | OwnerSequence,
    ctx: _CompileContext,
) -> list[ConditioningBranch]:
    operation_id = ctx.next_sequence_operation_id()
    if isinstance(node, OwnerSequence) and isinstance(node.owner, Sequence):
        return _compile_terminal_attachment_owner_sequence(
            node, node.owner, ctx, operation_id=operation_id
        )
    owner = render_ir_node(node.owner).strip(" ,") if isinstance(node, OwnerSequence) else ""
    syntax_origin = str(getattr(node, "syntax_origin", "") or "classic_closed_sequence")
    top_terminator = str(getattr(node, "top_terminator", "") or "")
    outer_sequence_weight = float(node.weight) if isinstance(node, Sequence) else 1.0
    last_index = len(node.items) - 1
    branches: list[ConditioningBranch] = []

    for index, item in enumerate(node.items):
        active_until = item.active_until_step
        hold_after = False
        if isinstance(node, Sequence) and node.active_until_step is not None:
            active_until = node.active_until_step
            hold_after = index == last_index

        relation_parent, relation_child = _relationship_parts(item.node)
        relation_operation_id = f"{operation_id}.rel-{index}" if isinstance(item.node, Relation) else None
        source_span = (item.source_start, item.source_end)
        metadata = _SequenceMetadata(
            operation_id=operation_id,
            local_weight=float(item.weight),
            item_index=index,
            relation_operation_id=relation_operation_id,
            relation_parent=relation_parent,
            relation_child=relation_child,
            owner_text=owner,
            syntax_origin=syntax_origin,
            source_span=source_span,
            terminator=str(item.terminator or ""),
            parent_scope=("whole_composition" if isinstance(node, OwnerSequence) and isinstance(node.owner, (Group, ExperimentalGroup)) else ""),
            owner_composition=(render_ir_node(node.owner).strip(" ,") if isinstance(node, OwnerSequence) else ""),
        )

        contextual: IRNode
        if isinstance(node, OwnerSequence):
            contextual = Relation(parent=node.owner, child=item.node)
            semantic_role = "owner_relation_item"
            source_node_type = "owner_sequence"
        else:
            contextual = item.node
            semantic_role = "relation_item" if isinstance(item.node, Relation) else "sequence_item"
            source_node_type = "sequence"

        if contains_experimental_group(contextual):
            experimental_role = (
                "owner_relation_experimental_group_variant"
                if isinstance(node, OwnerSequence)
                else "relation_experimental_group_variant" if isinstance(item.node, Relation) else "sequence_experimental_group_item"
            )
            branches.extend(
                _compile_experimental_group_operation(
                    contextual,
                    ctx,
                    semantic_role=experimental_role,
                    source_node_type=source_node_type,
                    outer_weight=outer_sequence_weight,
                    active_until_step=active_until,
                    hold_after_step=hold_after,
                    sequence_metadata=metadata,
                )
            )
            continue

        if contains_group(contextual):
            group_role = (
                "owner_relation_group_variant"
                if isinstance(node, OwnerSequence)
                else "relation_group_variant" if isinstance(item.node, Relation) else "sequence_group_item"
            )
            branches.extend(
                _compile_group_operation(
                    contextual,
                    ctx,
                    semantic_role=group_role,
                    source_node_type=source_node_type,
                    outer_weight=outer_sequence_weight,
                    active_until_step=active_until,
                    hold_after_step=hold_after,
                    sequence_metadata=metadata,
                )
            )
            continue

        text = render_ir_node(contextual).strip()
        if not text:
            continue
        branches.append(
            _apply_sequence_metadata(
                ConditioningBranch(
                    text=text,
                    weight=outer_sequence_weight,
                    active_until_step=active_until,
                    hold_after_step=hold_after,
                    semantic_role=semantic_role,
                    source_node_type=source_node_type,
                ),
                metadata,
            )
        )

    normalized = _normalized_weights(node.items)
    aliases = tuple(
        warning for warning in ctx.warnings if "compatibility" in warning.lower() or "alias" in warning.lower()
    )
    ctx.relationship_diagnostics.append(
        RelationshipDiagnostic(
            operation_id=operation_id,
            syntax_origin=syntax_origin,
            owner=owner,
            item_count=len(node.items),
            relation_count=sum(isinstance(item.node, Relation) for item in node.items),
            source_items=tuple(str(item.source_text or "") for item in node.items),
            source_spans=tuple((item.source_start, item.source_end) for item in node.items),
            compiled_branch_texts=tuple(item.text for item in branches),
            raw_item_weights=tuple(float(item.weight) for item in node.items),
            normalized_local_weights=normalized,
            activity_windows=tuple(
                node.active_until_step if isinstance(node, Sequence) and node.active_until_step is not None else item.active_until_step
                for item in node.items
            ),
            terminators_consumed=tuple(str(item.terminator or "") for item in node.items),
            top_terminator_consumed=top_terminator,
            compatibility_aliases=aliases,
            parent_scope=("whole_composition" if isinstance(node, OwnerSequence) and isinstance(node.owner, (Group, ExperimentalGroup)) else ""),
            owner_composition=(render_ir_node(node.owner).strip(" ,") if isinstance(node, OwnerSequence) else ""),
        )
    )
    return branches



def _compile_node(node: IRNode, ctx: _CompileContext) -> list[ConditioningBranch]:
    if _contains_average_set(node):
        return _compile_average_composite(node, ctx)

    if isinstance(node, SemanticScope):
        return _compile_node(node.node, ctx)

    if not isinstance(node, Conjunction):
        lifted_conjunction = _expand_single_composable_context(node)
        if lifted_conjunction is not None:
            return _compile_node(lifted_conjunction, ctx)

    if isinstance(node, LiteralTextScope):
        return [
            ConditioningBranch(
                text=str(node.value),
                semantic_role="literal_text_scope",
                source_node_type="literal_text_scope",
            )
        ]

    if isinstance(node, (OwnerSequence, Sequence)):
        return _compile_sequence(node, ctx)

    if isinstance(node, AverageSet):
        if any(isinstance(item, Conjunction) and item.composition_mode.startswith("a1111_composable") for item in node.branches):
            raise ValueError(
                "PPSR-09E does not define mixed top-level || and composable AND semantics. "
                "Use one branch-composition operator at a time during qualification."
            )
        operation_id = ctx.next_average_operation_id()
        raw_weights = tuple(node.local_weights or tuple(1.0 for _ in node.branches))
        if len(raw_weights) != len(node.branches):
            raise ValueError("AverageSet local_weights must match the branch count.")
        weight_total = sum(float(value) for value in raw_weights)
        if abs(weight_total) <= 1e-8:
            raise ValueError("AverageSet local weights must not sum to zero.")
        normalized = tuple(float(value) / weight_total for value in raw_weights)
        output: list[ConditioningBranch] = []
        for branch_index, (branch_node, local_weight) in enumerate(zip(node.branches, normalized)):
            nested = _compile_node(branch_node, ctx)
            for item in nested:
                output.append(
                    replace(
                        item,
                        average_operation_id=operation_id,
                        average_local_weight=float(local_weight),
                        average_branch_index=int(branch_index),
                        semantic_role=(
                            item.semantic_role
                            if item.semantic_role not in {"text", "literal"}
                            else "average_set_member"
                        ),
                    )
                )
        return output

    if isinstance(node, ExperimentalGroup):
        return _compile_experimental_group_operation(
            node,
            ctx,
            semantic_role="experimental_group_member",
            source_node_type="experimental_group",
        )

    if isinstance(node, Group):
        return _compile_group_operation(
            node,
            ctx,
            semantic_role="group_member",
            source_node_type="group",
        )

    if isinstance(node, Prompt):
        if contains_experimental_group(node):
            return _compile_experimental_group_operation(
                node,
                ctx,
                semantic_role="experimental_group_context_variant",
                source_node_type="prompt",
            )
        if contains_group(node):
            return _compile_group_operation(
                node,
                ctx,
                semantic_role="group_context_variant",
                source_node_type="prompt",
            )
        return [
            ConditioningBranch(
                text=render_ir_node(node),
                semantic_role="prompt_with_structured_parts",
                source_node_type="prompt",
            )
        ]

    if isinstance(node, Relation):
        if contains_experimental_group(node):
            return _compile_experimental_group_operation(
                node,
                ctx,
                semantic_role="relation_experimental_group_variant",
                source_node_type="relation",
            )
        if contains_group(node):
            return _compile_group_operation(
                node,
                ctx,
                semantic_role="relation_group_variant",
                source_node_type="relation",
            )
        return [
            ConditioningBranch(
                text=render_ir_node(node),
                semantic_role="relation",
                source_node_type="relation",
                relation_parent=render_ir_node(node.parent).strip(" ,"),
                relation_child=render_ir_node(node.child).strip(" ,"),
            )
        ]

    if isinstance(node, Weighted):
        nested = _compile_node(node.node, ctx)
        return [replace(item, weight=float(item.weight) * float(node.weight)) for item in nested]

    if isinstance(node, Conjunction):
        if node.composition_mode.startswith("a1111_composable"):
            operation_id = ctx.next_composition_operation_id()
            output: list[ConditioningBranch] = []
            for branch_index, branch in enumerate(node.branches):
                nested = _compile_node(branch.node, ctx)
                for item in nested:
                    output.append(
                        replace(
                            item,
                            weight=float(item.weight) * float(branch.weight),
                            composition_operation_id=operation_id,
                            composition_mode=node.composition_mode,
                            composition_algorithm=node.algorithm,
                            composition_branch_index=int(branch_index),
                            semantic_role=(
                                item.semantic_role
                                if item.semantic_role not in {"text", "literal"}
                                else "composable_and_member"
                            ),
                        )
                    )
            return output
        output: list[ConditioningBranch] = []
        for branch in node.branches:
            nested = _compile_node(branch.node, ctx)
            for item in nested:
                output.append(replace(item, weight=float(item.weight) * float(branch.weight)))
        return output

    return [
        ConditioningBranch(
            text=render_ir_node(node),
            semantic_role="literal" if isinstance(node, Literal) else "text",
            source_node_type=_node_type(node),
        )
    ]



def _root_is_structured(root: IRNode) -> bool:
    if isinstance(root, Conjunction):
        return any(not isinstance(item.node, Text) for item in root.branches)
    if isinstance(root, Prompt):
        return any(not isinstance(item, Text) for item in root.parts)
    return not isinstance(root, Text)



def compile_conditioning_plan(prompt_ir: PromptIR) -> ConditioningPlan:
    ctx = _CompileContext(warnings=list(prompt_ir.warnings))
    ctx.binding_diagnostics.extend(binding_diagnostics(prompt_ir.root))

    protected_root, literal_replacements = _protect_literal_scopes_for_runtime(prompt_ir.root)
    branches = _compile_node(protected_root, ctx)
    branches = [_consume_chunk_break_sentinels(item) for item in branches]
    branches = [
        _restore_branch_literal_scopes(item, literal_replacements)
        for item in branches
    ]
    branches = [
        replace(
            item,
            temporal_source=(
                (item.protected_text or item.text)
                if contains_temporal_syntax(item.protected_text or item.text)
                else ""
            ),
            temporal_compiled=bool(contains_temporal_syntax(item.protected_text or item.text)),
        )
        for item in branches
    ]

    composition_mode = "standard"
    composition_algorithm = ""
    composition_operation_id = None
    if _contains_average_set(prompt_ir.root):
        composition_mode = "normalized_average"
        composition_algorithm = "branch_average_v1"
        composition_operation_id = next(
            (item.average_operation_id for item in branches if item.average_operation_id),
            None,
        )
    elif _count_composable_conjunctions(prompt_ir.root):
        composition_mode = "a1111_composable_guidance"
        composition_algorithm = "a1111_composable_guidance_v1"
        composition_operation_id = next(
            (item.composition_operation_id for item in branches if item.composition_operation_id),
            None,
        )
    elif isinstance(prompt_ir.root, Conjunction):
        composition_mode = "legacy_normalized_average"
        composition_algorithm = prompt_ir.root.algorithm

    # Runtime lowering is required not only for syntactically structured IR,
    # but also whenever the compiled plan carries semantics that the historical
    # raw-text compatibility path cannot preserve.  In particular, a plain
    # text-only A1111 composable AND still needs its branch metadata to survive
    # into StepConditioningResolver; otherwise get_multicond_prompt_list()
    # would re-split the raw text and discard composition_mode/branch indexes.
    semantic_runtime_route_required = composition_mode == "a1111_composable_guidance"
    lowering_required = semantic_runtime_route_required or _root_is_structured(prompt_ir.root) or bool(
        _ESCAPED_STRUCTURAL_RE.search(prompt_ir.raw_source)
    ) or any(item.temporal_compiled for item in branches)
    return ConditioningPlan(
        source=prompt_ir.normalized_source,
        branches=tuple(branches),
        lowering_required=lowering_required,
        fallbacks=tuple(dict.fromkeys(ctx.fallbacks)),
        warnings=tuple(dict.fromkeys(ctx.warnings)),
        group_diagnostics=tuple(ctx.group_diagnostics),
        experimental_group_diagnostics=tuple(ctx.experimental_group_diagnostics),
        binding_diagnostics=tuple(ctx.binding_diagnostics),
        relationship_diagnostics=tuple(ctx.relationship_diagnostics),
        numeric_semantics=tuple(prompt_ir.numeric_semantics),
        composition_mode=composition_mode,
        composition_algorithm=composition_algorithm,
        composition_operation_id=composition_operation_id,
        contract=(
            CONDITIONING_PLAN_CONTRACT_VERSION
            if composition_mode in {"normalized_average", "a1111_composable_guidance"}
            or any(item.chunk_break_count or item.chunk_break_segments for item in branches)
            or any(item.literal_scope_replacements for item in branches)
            else LEGACY_CONDITIONING_PLAN_CONTRACT_VERSION
        ),
    )
