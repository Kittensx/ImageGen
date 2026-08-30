from __future__ import annotations

"""Deterministic group expansion for PPSR-03.

This module is deliberately tensor-free.  It turns PromptIR group structure
into context-preserving encoder-text variants with locally normalized weights.
The runtime resolver later averages the encoded variants *inside* their group
operation before applying unrelated top-level/AND weights.
"""

from dataclasses import dataclass, field, replace
import os
from typing import Callable

from modules.prompt_parsers.ir import (
    Conjunction,
    Group,
    IRNode,
    Literal,
    LiteralTextScope,
    SemanticScope,
    OwnerSequence,
    Prompt,
    Relation,
    Sequence,
    Text,
    Weighted,
)

DEFAULT_GROUP_COMBO_LIMIT = 100


@dataclass(frozen=True)
class GroupVariant:
    text: str
    local_weight: float = 1.0
    member_path: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GroupDiagnostic:
    operation_id: str
    group_id: str
    source_text: str
    member_count: int
    source_members: tuple[str, ...]
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
            "member_count": int(self.member_count),
            "source_members": list(self.source_members),
            "explicit_weight_flags": list(self.explicit_weight_flags),
            "raw_member_weights": [float(item) for item in self.raw_member_weights],
            "normalized_local_weights": [float(item) for item in self.normalized_local_weights],
            "combination_count": int(self.combination_count),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class GroupExpansion:
    operation_id: str
    variants: tuple[GroupVariant, ...]
    diagnostics: tuple[GroupDiagnostic, ...]
    fallback_used: bool = False
    fallback_reason: str = ""
    warning: str = ""


class GroupCompilationError(ValueError):
    pass


class GroupCombinationLimitExceeded(GroupCompilationError):
    pass


def group_combo_limit() -> int:
    raw = str(os.environ.get("GROUP_COMBO_LIMIT", DEFAULT_GROUP_COMBO_LIMIT)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_GROUP_COMBO_LIMIT
    return max(1, value)


def contains_group(node: IRNode) -> bool:
    if isinstance(node, Group):
        return True
    if isinstance(node, SemanticScope):
        return contains_group(node.node)
    if isinstance(node, LiteralTextScope):
        return False
    if isinstance(node, Prompt):
        return any(contains_group(item) for item in node.parts)
    if isinstance(node, Relation):
        return contains_group(node.parent) or contains_group(node.child)
    if isinstance(node, Weighted):
        return contains_group(node.node)
    if isinstance(node, Sequence):
        return any(contains_group(item.node) for item in node.items)
    if isinstance(node, OwnerSequence):
        return contains_group(node.owner) or any(contains_group(item.node) for item in node.items)
    if isinstance(node, Conjunction):
        return any(contains_group(item.node) for item in node.branches)
    return False


def _join_relation(left: str, right: str) -> str:
    return ", ".join(part for part in (left.strip(" ,"), right.strip(" ,")) if part)


class _ExpansionBuilder:
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
        self.diagnostics: list[GroupDiagnostic] = []

    def _guard(self, count: int) -> None:
        if int(count) > self.combo_limit:
            raise GroupCombinationLimitExceeded(
                f"group combination count {count} exceeds deterministic limit {self.combo_limit}"
            )

    def _cross(self, left: list[GroupVariant], right: list[GroupVariant], *, join: str) -> list[GroupVariant]:
        self._guard(len(left) * len(right))
        output: list[GroupVariant] = []
        for a in left:
            for b in right:
                if join == "concat":
                    text = f"{a.text}{b.text}"
                elif join == "relation":
                    text = _join_relation(a.text, b.text)
                else:
                    text = f"{a.text}{join}{b.text}"
                output.append(
                    GroupVariant(
                        text=text,
                        local_weight=float(a.local_weight) * float(b.local_weight),
                        member_path=tuple(a.member_path) + tuple(b.member_path),
                    )
                )
        return output

    def _group(self, node: Group, path: tuple[int, ...]) -> list[GroupVariant]:
        group_id = f"{self.operation_id}.g{self.group_counter}"
        self.group_counter += 1
        if not node.items:
            return [GroupVariant(text="", local_weight=1.0, member_path=path)]

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
            source_members.append(self.render_node(child))

        invalid_reason = ""
        if any(weight < 0 for weight in raw_weights):
            invalid_reason = f"negative group-local weight is unsupported in {group_id}"
        total = float(sum(raw_weights))
        if not invalid_reason and total <= 0:
            invalid_reason = f"group-local weights sum to zero in {group_id}"
        normalized = (
            [float(weight) / total for weight in raw_weights]
            if not invalid_reason
            else []
        )

        self.diagnostics.append(
            GroupDiagnostic(
                operation_id=self.operation_id,
                group_id=group_id,
                source_text=str(node.source_text or ("{" + ", ".join(source_members) + "}")),
                member_count=len(child_nodes),
                source_members=tuple(source_members),
                explicit_weight_flags=tuple(explicit_flags),
                raw_member_weights=tuple(raw_weights),
                normalized_local_weights=tuple(normalized),
            )
        )
        if invalid_reason:
            raise GroupCompilationError(invalid_reason)

        output: list[GroupVariant] = []
        for member_index, (child, member_weight) in enumerate(zip(child_nodes, normalized)):
            child_variants = self.expand(child, path=path + (member_index,))
            self._guard(len(output) + len(child_variants))
            for variant in child_variants:
                output.append(
                    GroupVariant(
                        text=variant.text,
                        local_weight=float(member_weight) * float(variant.local_weight),
                        member_path=variant.member_path or path + (member_index,),
                    )
                )
        return output

    def expand(self, node: IRNode, *, path: tuple[int, ...] = ()) -> list[GroupVariant]:
        if isinstance(node, Group):
            return self._group(node, path)
        if isinstance(node, LiteralTextScope):
            return [GroupVariant(text=str(node.value), member_path=path)]
        if isinstance(node, SemanticScope):
            return self.expand(node.node, path=path)
        if isinstance(node, (Text, Literal)):
            # Preserve Prompt-part whitespace so outer context survives exactly;
            # compiler.py trims only the final complete encoder text.
            value = str(node.value).replace(r"\!", "!")
            return [GroupVariant(text=value, member_path=path)]
        if isinstance(node, Prompt):
            variants = [GroupVariant(text="", member_path=path)]
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
                GroupVariant(
                    text=item.text,
                    local_weight=float(item.local_weight) * float(node.weight),
                    member_path=item.member_path,
                )
                for item in output
            ]
        if isinstance(node, Sequence):
            # PPSR-06A: a legacy ``a:b:c`` sequence may be wrapped in a Group
            # specifically so the *entire equal-weight composition* can act as
            # a parent/owner.  Preserve the sequence members as locally weighted
            # variants instead of flattening them to ``a, b, c`` text.
            #
            # Legacy active-until windows are intentionally not guessed here;
            # temporal composition belongs to explicit PPSR-06 ``[...]`` syntax.
            if node.active_until_step is not None or any(
                item.active_until_step is not None for item in node.items
            ):
                raise GroupCompilationError(
                    "nested legacy sequence activity windows cannot be safely collapsed inside a group"
                )
            if not node.items:
                return [GroupVariant(text="", local_weight=1.0, member_path=path)]
            raw_weights = [float(item.weight) for item in node.items]
            if any(weight < 0 for weight in raw_weights):
                raise GroupCompilationError("negative nested sequence-local weight is unsupported")
            total = float(sum(raw_weights))
            if total <= 0:
                raise GroupCompilationError("nested sequence-local weights sum to zero")
            output: list[GroupVariant] = []
            for item_index, item in enumerate(node.items):
                item_weight = float(item.weight) / total
                child_variants = self.expand(item.node, path=path + (item_index,))
                self._guard(len(output) + len(child_variants))
                for variant in child_variants:
                    output.append(
                        GroupVariant(
                            text=variant.text,
                            local_weight=item_weight * float(variant.local_weight),
                            member_path=variant.member_path or path + (item_index,),
                        )
                    )
            return output
        # Conjunctions/owner-sequences own outer branch semantics in compiler.py.
        return [GroupVariant(text=self.render_node(node), member_path=path)]


def expand_group_operation(
    node: IRNode,
    *,
    operation_id: str,
    render_node: Callable[[IRNode], str],
    combo_limit: int | None = None,
) -> GroupExpansion:
    limit = group_combo_limit() if combo_limit is None else max(1, int(combo_limit))
    builder = _ExpansionBuilder(
        operation_id=operation_id,
        render_node=render_node,
        combo_limit=limit,
    )
    try:
        variants = builder.expand(node)
        builder._guard(len(variants))
    except (GroupCompilationError, GroupCombinationLimitExceeded) as exc:
        reason = str(exc)
        diagnostics = tuple(
            replace(
                item,
                fallback_used=True,
                fallback_reason=reason,
            )
            for item in builder.diagnostics
        )
        return GroupExpansion(
            operation_id=operation_id,
            variants=(),
            diagnostics=diagnostics,
            fallback_used=True,
            fallback_reason=reason,
            warning=f"PPSR-03 group fallback: {reason}; group flattened deterministically.",
        )

    count = len(variants)
    diagnostics = tuple(replace(item, combination_count=count) for item in builder.diagnostics)
    return GroupExpansion(
        operation_id=operation_id,
        variants=tuple(variants),
        diagnostics=diagnostics,
    )
