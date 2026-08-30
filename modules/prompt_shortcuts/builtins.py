from __future__ import annotations

from typing import Any

from modules.prompt_shortcuts.contracts import (
    DEFAULT_LEGACY_SEMANTIC_MODES,
    PromptShortcutProfileDescriptor,
)

_COMMON_PALETTE: tuple[dict[str, Any], ...] = (
    {"id": "attention", "label": "( )", "operator": "ATTENTION", "kind": "wrap", "prefix": "(", "suffix": ")", "placeholder": "subject", "description": "Increase attention using parentheses.", "example": "(subject)"},
    {"id": "deemphasis", "label": "[ ]", "operator": "DEEMPHASIS", "kind": "wrap", "prefix": "[", "suffix": "]", "placeholder": "subject", "description": "Decrease attention using square brackets.", "example": "[subject]"},
    {"id": "weight", "label": ":1.2", "operator": "WEIGHT", "kind": "wrap", "prefix": "(", "suffix": ":1.2)", "placeholder": "subject", "description": "Apply a numeric attention weight.", "example": "(subject:1.2)"},
    {"id": "alternate", "label": "A|B", "operator": "ALTERNATE", "kind": "wrap", "prefix": "[", "suffix": "|option]", "placeholder": "subject", "description": "Alternate prompt text across sampling steps.", "example": "[subject|option]"},
    {"id": "schedule", "label": "A→B", "operator": "SCHEDULE", "kind": "wrap", "prefix": "[", "suffix": ":replacement:0.5]", "placeholder": "subject", "description": "Schedule a prompt transition.", "example": "[subject:replacement:0.5]"},
    {"id": "and", "operator": "AND", "kind": "insert", "insert": " AND ", "description": "Create a composable conditioning branch.", "example": "cat:1.2 AND dog:0.8"},
    {"id": "group", "label": "{ }", "operator": "GROUP_OPEN", "kind": "wrap", "prefix": "{", "suffix": "}", "placeholder": "items", "description": "Group prompt items.", "example": "{red, blue}"},
    {"id": "sequence", "operator": "SEQUENCE", "kind": "insert", "insert": "::", "description": "Insert a closed sequence item separator. IMAGE_GEN uses \"property::value!\" for legacy closed sequences.", "example": "outfit::red_dress!"},
    {"id": "parent-child", "operator": "PARENT_CHILD", "kind": "insert", "insert": "PARENT_CHILD", "description": "Bind a parent concept to a child concept in canonical form.", "example": "parent PARENT_CHILD child", "parsers": ["combined"]},
    {"id": "deep-sequence", "operator": "DEEP_SEQUENCE", "kind": "insert", "insert": ":::", "description": "Insert a top-level sequence owner separator.", "example": "character:::outfit::red_dress!, appearance::green_eyes!!"},
    {"id": "close", "operator": "CLOSE", "kind": "insert", "insert": "!", "description": "Close the current sequence item.", "example": "!"},
    {"id": "top-close", "operator": "TOP_CLOSE", "kind": "insert", "insert": "!!", "description": "Close the top-level/deep sequence.", "example": "!!"},
    {"id": "bind", "operator": "BIND", "kind": "template", "template": "{{selection}} BIND{subject => detail}", "placeholder": "base prompt", "description": "Bind details to a base prompt.", "example": "portrait BIND{girl => red eyes}", "parsers": ["parser21", "superhybrid"]},
    {"id": "chunk", "operator": "CHUNK", "kind": "template", "template": "CHUNK{ {{selection}} | option }", "placeholder": "subject", "description": "Create a Parser 21 chunk block.", "example": "CHUNK{wolf | moonlight}", "parsers": ["parser21", "superhybrid"]},
    {"id": "blend", "operator": "BLEND", "kind": "template", "template": "BLEND{ {{selection}} * 1.0 | option * 1.0 }", "placeholder": "subject", "description": "Blend weighted Parser 21 branches.", "example": "BLEND{wolf * 1.0 | moonlight * 0.6}", "parsers": ["parser21", "superhybrid"]},
    {"id": "pool", "operator": "POOL", "kind": "template", "template": "POOL{ {{selection}} | option }", "placeholder": "subject", "description": "Create a Parser 21 pool block.", "example": "POOL{red | green | blue}", "parsers": ["parser21", "superhybrid"]},
    {"id": "morph", "operator": "MORPH", "kind": "template", "template": "MORPH{ {{selection}} => target }", "placeholder": "subject", "description": "Create a Parser 21 morph operation.", "example": "MORPH{day => night}", "parsers": ["parser21", "superhybrid"]},
    {"id": "assemble", "operator": "ASSEMBLE", "kind": "template", "template": "ASSEMBLE{ base={{selection}}, detail=detail }", "placeholder": "subject", "description": "Create a Parser 21 assemble block.", "example": "ASSEMBLE{base=portrait, detail=red eyes}", "parsers": ["parser21", "superhybrid"]},
)


def _emitters() -> dict[str, dict[str, str]]:
    return {
        "legacy": {
            "AND": "AND",
            "GROUP_OPEN": "{",
            "GROUP_CLOSE": "}",
            "SEQUENCE": "::",
            "DEEP_SEQUENCE": ":::",
            "CLOSE": "!",
            "TOP_CLOSE": "!!",
        },
        "combined": {
            "AND": "AND",
            "GROUP_OPEN": "{",
            "GROUP_CLOSE": "}",
            "SEQUENCE": "::",
            "DEEP_SEQUENCE": ":::",
            "PARENT_CHILD": "PARENT_CHILD",
            "CLOSE": "!!",
            "TOP_CLOSE": "!!!",
            "CHUNK": "CHUNK",
            "BLEND": "BLEND",
            "BIND": "BIND",
            "POOL": "POOL",
            "MORPH": "MORPH",
            "ASSEMBLE": "ASSEMBLE",
            "COMPOUND": "COMPOUND",
        },
        "parser21": {
            "AND": "AND",
            "GROUP_OPEN": "{",
            "GROUP_CLOSE": "}",
            "SEQUENCE": "::",
            "DEEP_SEQUENCE": ":::",
            "CLOSE": "!!",
            "TOP_CLOSE": "!!!",
            "CHUNK": "CHUNK",
            "BLEND": "BLEND",
            "BIND": "BIND",
            "POOL": "POOL",
            "MORPH": "MORPH",
            "ASSEMBLE": "ASSEMBLE",
            "COMPOUND": "COMPOUND",
        },
        "superhybrid": {
            "AND": "AND",
            "GROUP_OPEN": "{",
            "GROUP_CLOSE": "}",
            "SEQUENCE": "::",
            "DEEP_SEQUENCE": ":::",
            "CLOSE": "!!",
            "TOP_CLOSE": "!!!",
            "CHUNK": "CHUNK",
            "BLEND": "BLEND",
            "BIND": "BIND",
            "POOL": "POOL",
            "MORPH": "MORPH",
            "ASSEMBLE": "ASSEMBLE",
            "COMPOUND": "COMPOUND",
        },
    }


def _legacy_break_emitter(emitters: dict[str, dict[str, str]]) -> dict[str, str]:
    """Extend the production Legacy profile with typed BREAK emission only."""

    return {
        **emitters["legacy"],
        "BREAK": "BREAK",
    }


def _ppsr09e_legacy_emitter(emitters: dict[str, dict[str, str]]) -> dict[str, str]:
    """Extend Legacy emission only for provisional PPSR-09E profiles.

    Existing built-in profiles must retain their Phase-04 mapping hashes for
    replay/profile-snapshot compatibility, so the experimental tokens are not
    added to the shared emitter table above.
    """

    return {
        **emitters["legacy"],
        "AVERAGE_SET": "||",
        "BREAK": "BREAK",
    }


def _semantic_modes(**overrides: str) -> dict[str, str]:
    modes = dict(DEFAULT_LEGACY_SEMANTIC_MODES)
    modes.update({str(key): str(value) for key, value in overrides.items()})
    return modes


def builtin_prompt_shortcut_profiles() -> tuple[PromptShortcutProfileDescriptor, ...]:
    emitters = _emitters()
    return (
        PromptShortcutProfileDescriptor(
            profile_id="canonical",
            label="Canonical Operators",
            version="1",
            aliases={
                "AND": ("AND",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "PARENT_CHILD": ("PARENT_CHILD",),
                "DEEP_SEQUENCE": (":::",),
                "CLOSE": ("!!",),
                "TOP_CLOSE": ("!!!",),
                "CHUNK": ("CHUNK",),
                "BLEND": ("BLEND",),
                "BIND": ("BIND",),
                "POOL": ("POOL",),
                "MORPH": ("MORPH",),
                "ASSEMBLE": ("ASSEMBLE",),
                "COMPOUND": ("COMPOUND",),
            },
            parser_emitters=emitters,
            compatible_parsers=("legacy", "parser21", "superhybrid", "combined"),
            description="Stable parser-independent operator names and symbols.",
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="legacy_default",
            label="Legacy Default",
            version="3",
            aliases={
                "AND": ("AND",),
                "BREAK": ("BREAK",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "DEEP_SEQUENCE": (":::",),
                "CLOSE": ("!",),
                "TOP_CLOSE": ("!!",),
            },
            parser_emitters={"legacy": _legacy_break_emitter(emitters)},
            compatible_parsers=("legacy",),
            semantic_modes=_semantic_modes(break_mode="encoder_chunk_break_v1"),
            description=(
                "Preserves IMAGE_GEN legacy prompt syntax while using the corrected typed "
                "BREAK chunk-boundary runtime across supported model families."
            ),
            palette=tuple(item for item in _COMMON_PALETTE if not item.get("parsers") or "legacy" in item.get("parsers", [])),
        ),
        PromptShortcutProfileDescriptor(
            profile_id="advanced_symbols",
            label="Advanced Symbols",
            version="1",
            aliases={
                "AND": ("AND",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "PARENT_CHILD": ("PARENT_CHILD",),
                "DEEP_SEQUENCE": (":::",),
                "CLOSE": ("!!",),
                "TOP_CLOSE": ("!!!",),
                "CHUNK": ("&&", "CHUNK"),
                "BLEND": ("<+>", "BLEND"),
                "BIND": ("BIND",),
                "POOL": ("$$", "POOL"),
                "MORPH": (">>", "MORPH"),
                "ASSEMBLE": ("@@", "ASSEMBLE"),
            },
            parser_emitters=emitters,
            compatible_parsers=("legacy", "parser21", "superhybrid", "combined"),
            description="IMAGE_GEN sequence symbols plus optional readable or symbolic Parser 21 operators.",
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="parser21_native",
            label="Parser 21 Native",
            version="21",
            aliases={
                "AND": ("AND",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "DEEP_SEQUENCE": (":::",),
                "CLOSE": ("!!",),
                "TOP_CLOSE": ("!!!",),
                "CHUNK": ("CHUNK",),
                "BLEND": ("BLEND",),
                "BIND": ("BIND",),
                "POOL": ("POOL",),
                "MORPH": ("MORPH",),
                "ASSEMBLE": ("ASSEMBLE",),
                "COMPOUND": ("COMPOUND",),
            },
            parser_emitters={"parser21": emitters["parser21"]},
            compatible_parsers=("parser21",),
            credit="Parser 21 syntax contributed by GitHub user Konpr",
            description="Native Prompt Parser 21 words and sequence symbols.",
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="superhybrid_native",
            label="SuperHybrid Native",
            version="1",
            aliases={
                "AND": ("AND",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "DEEP_SEQUENCE": (":::",),
                "CLOSE": ("!!",),
                "TOP_CLOSE": ("!!!",),
                "CHUNK": ("CHUNK",),
                "BLEND": ("BLEND",),
                "BIND": ("BIND",),
                "POOL": ("POOL",),
                "MORPH": ("MORPH",),
                "ASSEMBLE": ("ASSEMBLE",),
                "COMPOUND": ("COMPOUND",),
            },
            parser_emitters={"superhybrid": emitters["superhybrid"]},
            compatible_parsers=("superhybrid",),
            description="Native SuperHybrid operators.",
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="imagegen_next",
            label="ImageGen Next (PPSR-09E Experimental)",
            version="09e-1",
            aliases={
                "AND": ("AND",),
                "AVERAGE_SET": ("||",),
                "BREAK": ("BREAK",),
                "GROUP_OPEN": ("{",),
                "GROUP_CLOSE": ("}",),
                "SEQUENCE": ("::",),
                "DEEP_SEQUENCE": (":::" ,),
                "CLOSE": ("!",),
                "TOP_CLOSE": ("!!",),
            },
            parser_emitters={"legacy": _ppsr09e_legacy_emitter(emitters)},
            compatible_parsers=("legacy",),
            semantic_modes=_semantic_modes(
                average_surface="double_pipe_v1",
                average_composition="branch_average_v1",
                and_composition="a1111_composable_guidance_v1",
                break_mode="encoder_chunk_break_v1",
                double_quote_scope="literal_text_scope_v1",
                single_quote_scope="semantic_scope_v1",
            ),
            description=(
                "Experimental PPSR-09E profile. Keeps ordinary braces on historical "
                "branch averaging while qualifying ||, composable AND, and BREAK."
            ),
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="a1111_compatible",
            label="A1111 Compatible",
            version="10b-1",
            aliases={
                "AND": ("AND",),
                "BREAK": ("BREAK",),
            },
            parser_emitters={
                "legacy": {
                    "AND": "AND",
                    "BREAK": "BREAK",
                }
            },
            compatible_parsers=("legacy",),
            semantic_modes=_semantic_modes(
                attention_algorithm="a1111_attention_v1",
                and_composition="a1111_composable_guidance_v1",
                group_composition="literal",
                relation_mode="literal",
                break_mode="encoder_chunk_break_v1",
                schedule_algorithm="a1111_schedule_v1",
                alternate_algorithm="a1111_alternate_v1",
                clip_chunking="a1111_clip_chunk_v1",
            ),
            preprocessing={
                "pipeline": "a1111_compat_preprocess_v1",
                "style_template": "a1111_prompt_placeholder_v1",
                "extra_networks": "imagegen_runtime_asset_adapter",
                "prompt_matrix": "queue_expansion_followup",
                "comments": "preserve",
            },
            description=(
                "PPSR-10B A1111-compatible prompt style: A1111 attention, schedules, "
                "alternation, composable AND, BREAK and long-CLIP chunking. Prompt matrix "
                "remains a separate queue-expansion feature."
            ),
            palette=tuple(
                item for item in _COMMON_PALETTE
                if item.get("id") in {"attention", "deemphasis", "weight", "alternate", "schedule", "and"}
            ) + (
                {
                    "id": "break",
                    "operator": "BREAK",
                    "kind": "insert",
                    "insert": " BREAK ",
                    "description": "Force the current A1111 CLIP chunk to flush before continuing.",
                    "example": "foreground BREAK background",
                },
            ),
        ),
        PromptShortcutProfileDescriptor(
            profile_id="a1111_compatible_test",
            label="A1111 Compatible (PPSR-09E Test)",
            version="09e-1",
            aliases={
                "AND": ("AND",),
                "BREAK": ("BREAK",),
                "AVERAGE_SET": ("||",),
            },
            parser_emitters={"legacy": _ppsr09e_legacy_emitter(emitters)},
            compatible_parsers=("legacy",),
            semantic_modes=_semantic_modes(
                average_surface="double_pipe_v1",
                and_composition="a1111_composable_guidance_v1",
                break_mode="encoder_chunk_break_v1",
            ),
            description="Experimental A1111 parity snapshot used by PPSR-09E qualification.",
            palette=_COMMON_PALETTE,
        ),
        PromptShortcutProfileDescriptor(
            profile_id="comfyui_compatible_test",
            label="ComfyUI Compatible (PPSR-09E Test)",
            version="09e-1",
            aliases={
                "AVERAGE_SET": ("||",),
            },
            parser_emitters={"legacy": _ppsr09e_legacy_emitter(emitters)},
            compatible_parsers=("legacy",),
            semantic_modes=_semantic_modes(
                average_surface="double_pipe_v1",
                and_composition="literal",
                break_mode="literal",
                dynamic_choice="comfy_frontend_random_v1",
            ),
            preprocessing={
                "pipeline": "comfy_frontend_dynamic_prompt_v1",
                "dynamic_choice": "enabled",
                "comments": "strip",
            },
            reserved_syntax=("{", "}"),
            description="Experimental ComfyUI compatibility snapshot; not a production preset.",
            palette=_COMMON_PALETTE,
        ),
    )


BUILTIN_PARSER_PRESETS: tuple[dict[str, object], ...] = (
    {
        "preset_id": "legacy_default",
        "name": "Legacy Default",
        "prompt_parser_name": "legacy",
        "shortcut_profile_name": "legacy_default",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "combined_prefer_legacy",
        "name": "Combined (Prefer Legacy)",
        "prompt_parser_name": "combined",
        "shortcut_profile_name": "canonical",
        "prompt_parser_kwargs": {"fallback_order": ["legacy", "parser21"]},
        "fallback_policy": "prefer_legacy",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "combined_prefer_parser21",
        "name": "Combined (Prefer Parser 21)",
        "prompt_parser_name": "combined",
        "shortcut_profile_name": "canonical",
        "prompt_parser_kwargs": {"fallback_order": ["parser21", "legacy"]},
        "fallback_policy": "prefer_parser21",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "combined_strict",
        "name": "Combined (Strict)",
        "prompt_parser_name": "combined",
        "shortcut_profile_name": "canonical",
        "prompt_parser_kwargs": {"strict": True},
        "fallback_policy": "strict",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "parser21_native",
        "name": "Parser 21 Native",
        "prompt_parser_name": "parser21",
        "shortcut_profile_name": "parser21_native",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "superhybrid_native",
        "name": "SuperHybrid Native",
        "prompt_parser_name": "superhybrid",
        "shortcut_profile_name": "superhybrid_native",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "imagegen_next",
        "name": "ImageGen Next (PPSR-09E Experimental)",
        "prompt_parser_name": "legacy",
        "shortcut_profile_name": "imagegen_next",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "a1111_compatible",
        "name": "A1111 Compatible",
        "prompt_parser_name": "legacy",
        "shortcut_profile_name": "a1111_compatible",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "a1111_compatible_test",
        "name": "A1111 Compatible (PPSR-09E Test)",
        "prompt_parser_name": "legacy",
        "shortcut_profile_name": "a1111_compatible_test",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
    {
        "preset_id": "comfyui_compatible_test",
        "name": "ComfyUI Compatible (PPSR-09E Test)",
        "prompt_parser_name": "legacy",
        "shortcut_profile_name": "comfyui_compatible_test",
        "prompt_parser_kwargs": {},
        "fallback_policy": "fail",
        "hires_inheritance": "same_as_base",
    },
)
