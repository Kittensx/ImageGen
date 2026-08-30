from __future__ import annotations

"""PPSR-09 experimental modifier/target binding semantics.

``modifier^target``
    Target-only binding. The explicit binding is an inheritance barrier: an
    ancestor ``*`` modifier is not applied to that target/subtree, and the local
    modifier itself does not propagate to descendants.

``modifier*target``
    Subtree binding. The explicit binding is also an inheritance barrier for
    ancestor modifiers, then establishes its own modifier for descendants.

The compiler keeps these as structural semantics until encoder-text lowering.
The first experimental lowering reinforces the pair bidirectionally rather than
sending a naked modifier branch to the encoder:

    red^hair -> "red hair, hair is red"

This is intentionally an experiment, not a promise of hard symbolic control.
"""

from dataclasses import dataclass, replace
from typing import Iterable

from modules.prompt_parsers.ir import (
    BoundConcept,
    Alternate,
    Conjunction,
    ConjunctionBranch,
    ExperimentalGroup,
    Group,
    IRNode,
    OwnerSequence,
    Quantity,
    Prompt,
    Relation,
    Scheduled,
    Sequence,
    SequenceItemIR,
    Text,
    Literal,
    LiteralTextScope,
    SemanticScope,
    Weighted,
)

BINDING_ALGORITHM = "bidirectional_pair_reinforcement_v1"


@dataclass(frozen=True)
class BindingDiagnostic:
    source_text: str
    operator: str
    modifier: str
    target: str
    scope: str
    inheritance_barrier: bool = True
    algorithm: str = BINDING_ALGORITHM

    def to_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "operator": self.operator,
            "modifier": self.modifier,
            "target": self.target,
            "scope": self.scope,
            "inheritance_barrier": bool(self.inheritance_barrier),
            "algorithm": self.algorithm,
        }


def binding_phrase(node: BoundConcept) -> str:
    modifier = " ".join(str(node.modifier or "").split())
    target = " ".join(str(node.target or "").split())
    if not modifier:
        return target
    if not target:
        return modifier
    # Keep the modifier attached to the target in both clauses; never emit the
    # modifier as an isolated encoder branch.
    return f"{modifier} {target}, {target} is {modifier}"


def contains_binding(node: IRNode) -> bool:
    if isinstance(node, BoundConcept):
        return True
    if isinstance(node, SemanticScope):
        return contains_binding(node.node)
    if isinstance(node, LiteralTextScope):
        return False
    if isinstance(node, (Group, ExperimentalGroup)):
        return any(contains_binding(item) for item in node.items)
    if isinstance(node, Prompt):
        return any(contains_binding(item) for item in node.parts)
    if isinstance(node, Relation):
        return contains_binding(node.parent) or contains_binding(node.child)
    if isinstance(node, Weighted):
        return contains_binding(node.node)
    if isinstance(node, Sequence):
        return any(contains_binding(item.node) for item in node.items)
    if isinstance(node, OwnerSequence):
        return contains_binding(node.owner) or any(contains_binding(item.node) for item in node.items)
    if isinstance(node, Conjunction):
        return any(contains_binding(item.node) for item in node.branches)
    return False


def direct_binding(node: IRNode) -> BoundConcept | None:
    """Return a binding only when this exact semantic atom owns the node.

    A Prompt wrapper with only whitespace/punctuation plus one BoundConcept is
    accepted so parser formatting does not change inheritance behavior.
    """
    if isinstance(node, BoundConcept):
        return node
    if isinstance(node, Weighted):
        return direct_binding(node.node)
    if isinstance(node, Prompt):
        bound: list[BoundConcept] = []
        for part in node.parts:
            if isinstance(part, BoundConcept):
                bound.append(part)
                continue
            value = getattr(part, "value", None)
            if isinstance(value, str) and not value.strip(" ,;:!\t\r\n"):
                continue
            return None
        if len(bound) == 1:
            return bound[0]
    return None


def child_inheritance(parent: IRNode, inherited: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve modifiers inherited by structural descendants of ``parent``.

    Any explicit local binding is a barrier. ``^`` clears ancestor inheritance
    and starts no new scope; ``*`` clears ancestors then starts its own scope.
    """
    binding = direct_binding(parent)
    if binding is None:
        return tuple(inherited)
    if str(binding.scope or "") == "subtree" or binding.operator == "*":
        return (str(binding.modifier or "").strip(),) if str(binding.modifier or "").strip() else ()
    return ()


def inherited_text(text: str, modifiers: tuple[str, ...]) -> str:
    value = str(text or "")
    if not modifiers or not any(char.isalnum() for char in value):
        return value
    prefix = " ".join(item.strip() for item in modifiers if item.strip()).strip()
    if not prefix:
        return value
    # This intentionally allows conflicting unbound descriptors to compete:
    # red*car -> child text "white seats" becomes "red white seats". Adding
    # white^seats creates an explicit barrier and suppresses inherited red.
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    core = value.strip()
    return f"{leading}{prefix} {core}{trailing}"



def apply_inherited_bindings(node: IRNode, inherited: tuple[str, ...] = ()) -> IRNode:
    """Materialize ``*`` inheritance into a structural copy for branch expansion.

    The ordinary renderer can carry inheritance recursively, but the group
    expanders intentionally split nodes into independent branches.  Applying
    inheritance to a copy first ensures a descendant inside ``{...}`` or
    ``⦃...⦄`` does not accidentally escape its ancestor ``*`` scope.  Explicit
    ``^``/``*`` bindings remain barriers and therefore are never prefixed by an
    inherited modifier.
    """
    if isinstance(node, LiteralTextScope):
        return node
    if isinstance(node, SemanticScope):
        return replace(node, node=apply_inherited_bindings(node.node, inherited))
    if isinstance(node, (Text, Literal)):
        return replace(node, value=inherited_text(str(node.value or ""), inherited))
    if isinstance(node, BoundConcept):
        return node
    if isinstance(node, Group):
        return replace(node, items=tuple(apply_inherited_bindings(item, inherited) for item in node.items))
    if isinstance(node, ExperimentalGroup):
        return replace(node, items=tuple(apply_inherited_bindings(item, inherited) for item in node.items))
    if isinstance(node, Prompt):
        return replace(node, parts=tuple(apply_inherited_bindings(item, inherited) for item in node.parts))
    if isinstance(node, Relation):
        parent = apply_inherited_bindings(node.parent, inherited)
        descendant_scope = child_inheritance(node.parent, inherited)
        child = apply_inherited_bindings(node.child, descendant_scope)
        return replace(node, parent=parent, child=child)
    if isinstance(node, Weighted):
        return replace(node, node=apply_inherited_bindings(node.node, inherited))
    if isinstance(node, Quantity):
        return replace(node, node=apply_inherited_bindings(node.node, inherited))
    if isinstance(node, Sequence):
        return replace(
            node,
            items=tuple(
                replace(item, node=apply_inherited_bindings(item.node, inherited))
                for item in node.items
            ),
        )
    if isinstance(node, OwnerSequence):
        owner = apply_inherited_bindings(node.owner, inherited)
        descendant_scope = child_inheritance(node.owner, inherited)
        return replace(
            node,
            owner=owner,
            items=tuple(
                replace(item, node=apply_inherited_bindings(item.node, descendant_scope))
                for item in node.items
            ),
        )
    if isinstance(node, Conjunction):
        return replace(
            node,
            branches=tuple(
                replace(branch, node=apply_inherited_bindings(branch.node, inherited))
                for branch in node.branches
            ),
        )
    if isinstance(node, Scheduled):
        return replace(node, value=inherited_text(str(node.value or ""), inherited))
    if isinstance(node, Alternate):
        return replace(node, value=inherited_text(str(node.value or ""), inherited))
    return node


def iter_bindings(node: IRNode) -> Iterable[BoundConcept]:
    if isinstance(node, BoundConcept):
        yield node
        return
    if isinstance(node, SemanticScope):
        yield from iter_bindings(node.node)
        return
    if isinstance(node, LiteralTextScope):
        return
    if isinstance(node, (Group, ExperimentalGroup)):
        for item in node.items:
            yield from iter_bindings(item)
        return
    if isinstance(node, Prompt):
        for item in node.parts:
            yield from iter_bindings(item)
        return
    if isinstance(node, Relation):
        yield from iter_bindings(node.parent)
        yield from iter_bindings(node.child)
        return
    if isinstance(node, Weighted):
        yield from iter_bindings(node.node)
        return
    if isinstance(node, Sequence):
        for item in node.items:
            yield from iter_bindings(item.node)
        return
    if isinstance(node, OwnerSequence):
        yield from iter_bindings(node.owner)
        for item in node.items:
            yield from iter_bindings(item.node)
        return
    if isinstance(node, Conjunction):
        for item in node.branches:
            yield from iter_bindings(item.node)


def binding_diagnostics(node: IRNode) -> tuple[BindingDiagnostic, ...]:
    return tuple(
        BindingDiagnostic(
            source_text=item.source_text,
            operator=item.operator,
            modifier=item.modifier,
            target=item.target,
            scope=item.scope,
        )
        for item in iter_bindings(node)
    )
