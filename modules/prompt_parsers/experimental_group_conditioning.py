from __future__ import annotations

"""PPSR-09 experimental cohesive-group conditioning.

This module intentionally does *not* replace PPSR-03 ``{...}`` grouping.
``ExperimentalGroup`` / ``⦃...⦄`` is an A/B-only implementation whose first
algorithm keeps every member in shared encoder context and then repeats one
member as the local focus branch:

    ⦃red hair, green eyes⦄
      -> 0.5 * encode("red hair, green eyes, red hair")
       + 0.5 * encode("red hair, green eyes, green eyes")

The established runtime group resolver still performs the local weighted
average, but the source text seen by each encoder branch never loses the other
group members. Singleton groups collapse to plain text for parity.
"""

from dataclasses import dataclass, field, replace
import os
from typing import Callable

from modules.prompt_parsers.ir import (
    ExperimentalGroup,
    IRNode,
    Literal,
    LiteralTextScope,
    SemanticScope,
    Prompt,
    Relation,
    Text,
    Weighted,
)

DEFAULT_EXPERIMENTAL_GROUP_COMBO_LIMIT = 100
EXPERIMENTAL_GROUP_ALGORITHM = "shared_context_focus_v1"


@dataclass(frozen=True)
class ExperimentalGroupVariant:
    text: str
    local_weight: float = 1.0
    member_path: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentalGroupDiagnostic:
    operation_id: str
    group_id: str
    source_text: str
    algorithm: str
    member_count: int
    source_members: tuple[str, ...]
    focus_branch_texts: tuple[str, ...]
    explicit_weight_flags: tuple[bool, ...]
    raw_member_weights: tuple[float, ...]
    normalized_local_weights: tuple[float, ...]
    combination_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "group_id": self.group_id,
            "source_text": self.source_text,
            "algorithm": self.algorithm,
            "member_count": int(self.member_count),
            "source_members": list(self.source_members),
            "focus_branch_texts": list(self.focus_branch_texts),
            "explicit_weight_flags": list(self.explicit_weight_flags),
            "raw_member_weights": [float(item) for item in self.raw_member_weights],
            "normalized_local_weights": [float(item) for item in self.normalized_local_weights],
            "combination_count": int(self.combination_count),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ExperimentalGroupExpansion:
    operation_id: str
    variants: tuple[ExperimentalGroupVariant, ...]
    diagnostics: tuple[ExperimentalGroupDiagnostic, ...]
    fallback_used: bool = False
    fallback_reason: str = ""
    warning: str = ""


class ExperimentalGroupCompilationError(ValueError):
    pass


class ExperimentalGroupCombinationLimitExceeded(ExperimentalGroupCompilationError):
    pass


def experimental_group_combo_limit() -> int:
    raw = str(
        os.environ.get(
            "PPSR09_EXPERIMENTAL_GROUP_COMBO_LIMIT",
            DEFAULT_EXPERIMENTAL_GROUP_COMBO_LIMIT,
        )
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_EXPERIMENTAL_GROUP_COMBO_LIMIT
    return max(1, value)


def contains_experimental_group(node: IRNode) -> bool:
    from modules.prompt_parsers.ir import Conjunction, OwnerSequence, Sequence

    if isinstance(node, ExperimentalGroup):
        return True
    if isinstance(node, Prompt):
        return any(contains_experimental_group(item) for item in node.parts)
    if isinstance(node, Relation):
        return contains_experimental_group(node.parent) or contains_experimental_group(node.child)
    if isinstance(node, Weighted):
        return contains_experimental_group(node.node)
    if isinstance(node, Sequence):
        return any(contains_experimental_group(item.node) for item in node.items)
    if isinstance(node, OwnerSequence):
        return contains_experimental_group(node.owner) or any(
            contains_experimental_group(item.node) for item in node.items
        )
    if isinstance(node, Conjunction):
        return any(contains_experimental_group(item.node) for item in node.branches)
    return False


def _join_relation(left: str, right: str) -> str:
    return ", ".join(part for part in (left.strip(" ,"), right.strip(" ,")) if part)


class _ExperimentalExpansionBuilder:
    def __init__(
        self,
        *,
        operation_id: str,
        render_node: Callable[[IRNode], str],
        combo_limit: int,
    ) -> None:
        self.operation_id = str(operation_id)
        self.render_node = render_node
        self.combo_limit = max(1, int(combo_limit))
        self.group_counter = 0
        self.diagnostics: list[ExperimentalGroupDiagnostic] = []

    def _guard(self, count: int) -> None:
        if int(count) > self.combo_limit:
            raise ExperimentalGroupCombinationLimitExceeded(
                f"experimental group combination count {count} exceeds deterministic limit {self.combo_limit}"
            )

    def _cross(
        self,
        left: list[ExperimentalGroupVariant],
        right: list[ExperimentalGroupVariant],
        *,
        join: str,
    ) -> list[ExperimentalGroupVariant]:
        self._guard(len(left) * len(right))
        output: list[ExperimentalGroupVariant] = []
        for a in left:
            for b in right:
                if join == "concat":
                    text = f"{a.text}{b.text}"
                elif join == "relation":
                    text = _join_relation(a.text, b.text)
                else:
                    text = f"{a.text}{join}{b.text}"
                output.append(
                    ExperimentalGroupVariant(
                        text=text,
                        local_weight=float(a.local_weight) * float(b.local_weight),
                        member_path=tuple(a.member_path) + tuple(b.member_path),
                    )
                )
        return output

    def _experimental_group(
        self,
        node: ExperimentalGroup,
        path: tuple[int, ...],
    ) -> list[ExperimentalGroupVariant]:
        group_id = f"{self.operation_id}.xg{self.group_counter}"
        self.group_counter += 1
        if not node.items:
            return [ExperimentalGroupVariant(text="", local_weight=1.0, member_path=path)]

        raw_weights: list[float] = []
        explicit_flags: list[bool] = []
        child_nodes: list[IRNode] = []
        source_members: list[str] = []
        for item in node.items:
            explicit = isinstance(item, Weighted)
            child = item.node if explicit else item
            weight = float(item.weight) if explicit else 1.0
            raw_weights.append(weight)
            explicit_flags.append(explicit)
            child_nodes.append(child)
            source_members.append(self.render_node(child).strip(" ,"))

        if any(weight < 0 for weight in raw_weights):
            raise ExperimentalGroupCompilationError(
                f"negative experimental group-local weight is unsupported in {group_id}"
            )
        total = float(sum(raw_weights))
        if total <= 0:
            raise ExperimentalGroupCompilationError(
                f"experimental group-local weights sum to zero in {group_id}"
            )
        normalized = [float(weight) / total for weight in raw_weights]

        shared_context = ", ".join(member for member in source_members if member).strip(" ,")
        if len(child_nodes) == 1:
            focus_texts = tuple(source_members)
            variants = [
                ExperimentalGroupVariant(
                    text=source_members[0] if source_members else "",
                    local_weight=1.0,
                    member_path=path + (0,),
                )
            ]
        else:
            variants = []
            focus_texts_list: list[str] = []
            for member_index, (member_text, member_weight) in enumerate(
                zip(source_members, normalized)
            ):
                focus_text = _join_relation(shared_context, member_text)
                focus_texts_list.append(focus_text)
                variants.append(
                    ExperimentalGroupVariant(
                        text=focus_text,
                        local_weight=float(member_weight),
                        member_path=path + (member_index,),
                    )
                )
            focus_texts = tuple(focus_texts_list)

        self._guard(len(variants))
        self.diagnostics.append(
            ExperimentalGroupDiagnostic(
                operation_id=self.operation_id,
                group_id=group_id,
                source_text=str(node.source_text or "⦃" + ", ".join(source_members) + "⦄"),
                algorithm=str(node.algorithm or EXPERIMENTAL_GROUP_ALGORITHM),
                member_count=len(child_nodes),
                source_members=tuple(source_members),
                focus_branch_texts=focus_texts,
                explicit_weight_flags=tuple(explicit_flags),
                raw_member_weights=tuple(raw_weights),
                normalized_local_weights=tuple(normalized),
            )
        )
        return variants

    def expand(
        self,
        node: IRNode,
        *,
        path: tuple[int, ...] = (),
    ) -> list[ExperimentalGroupVariant]:
        if isinstance(node, ExperimentalGroup):
            return self._experimental_group(node, path)
        if isinstance(node, LiteralTextScope):
            return [ExperimentalGroupVariant(text=str(node.value), member_path=path)]
        if isinstance(node, SemanticScope):
            return self.expand(node.node, path=path)
        if isinstance(node, (Text, Literal)):
            value = str(node.value).replace(r"\!", "!")
            return [ExperimentalGroupVariant(text=value, member_path=path)]
        if isinstance(node, Prompt):
            variants = [ExperimentalGroupVariant(text="", member_path=path)]
            for part in node.parts:
                variants = self._cross(variants, self.expand(part, path=path), join="concat")
            return variants
        if isinstance(node, Relation):
            return self._cross(
                self.expand(node.parent, path=path),
                self.expand(node.child, path=path),
                join="relation",
            )
        if isinstance(node, Weighted):
            output = self.expand(node.node, path=path)
            return [
                ExperimentalGroupVariant(
                    text=item.text,
                    local_weight=float(item.local_weight) * float(node.weight),
                    member_path=item.member_path,
                )
                for item in output
            ]
        # Bindings, sequences, standard groups, and other established nodes are
        # rendered as one shared-context phrase here. Their own compiler remains
        # authoritative when they are not nested inside ⦃...⦄.
        return [ExperimentalGroupVariant(text=self.render_node(node), member_path=path)]


def expand_experimental_group_operation(
    node: IRNode,
    *,
    operation_id: str,
    render_node: Callable[[IRNode], str],
    combo_limit: int | None = None,
) -> ExperimentalGroupExpansion:
    limit = (
        experimental_group_combo_limit()
        if combo_limit is None
        else max(1, int(combo_limit))
    )
    builder = _ExperimentalExpansionBuilder(
        operation_id=operation_id,
        render_node=render_node,
        combo_limit=limit,
    )
    try:
        variants = builder.expand(node)
        builder._guard(len(variants))
    except (
        ExperimentalGroupCompilationError,
        ExperimentalGroupCombinationLimitExceeded,
    ) as exc:
        reason = str(exc)
        diagnostics = tuple(
            replace(item, fallback_used=True, fallback_reason=reason)
            for item in builder.diagnostics
        )
        return ExperimentalGroupExpansion(
            operation_id=operation_id,
            variants=(),
            diagnostics=diagnostics,
            fallback_used=True,
            fallback_reason=reason,
            warning=(
                f"PPSR-09 experimental group fallback: {reason}; "
                "experimental group flattened deterministically."
            ),
        )

    count = len(variants)
    diagnostics = tuple(replace(item, combination_count=count) for item in builder.diagnostics)
    return ExperimentalGroupExpansion(
        operation_id=operation_id,
        variants=tuple(variants),
        diagnostics=diagnostics,
    )
