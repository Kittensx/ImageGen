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
    Alternate,
    BoundConcept,
    Conjunction,
    ExperimentalGroup,
    Group,
    IRNode,
    Literal,
    OwnerSequence,
    Prompt,
    PromptIR,
    Relation,
    Scheduled,
    Sequence,
    SequenceItemIR,
    Text,
    Weighted,
)

CONDITIONING_PLAN_CONTRACT_VERSION = "image-gen-conditioning-plan-v6"
_ESCAPED_STRUCTURAL_RE = re.compile(r"\\[{}⦃⦄^*:!|\\]")


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    if isinstance(node, Conjunction):
        return " AND ".join(_render_ir_node(item.node, inherited_modifiers) for item in node.branches)
    if isinstance(node, (Scheduled, Alternate)):
        return node.value
    return str(node)


def render_ir_node(node: IRNode) -> str:
    return _render_ir_node(node, ())


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
    if isinstance(node, (OwnerSequence, Sequence)):
        return _compile_sequence(node, ctx)

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
    branches = _compile_node(prompt_ir.root, ctx)
    branches = [
        replace(
            item,
            temporal_source=item.text if contains_temporal_syntax(item.text) else "",
            temporal_compiled=bool(contains_temporal_syntax(item.text)),
        )
        for item in branches
    ]
    lowering_required = _root_is_structured(prompt_ir.root) or bool(
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
    )
