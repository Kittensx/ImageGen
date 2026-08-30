# Prompt Tools Frontend

IMAGE_GEN's prompt controls are exposed through one stable WebUI entry point,
`prompt-tools.js`, while individual prompt features are maintained separately.
This is an internal architecture change; existing prompt controls and saved
prompt settings keep the same user-facing behavior.

## Prompt features

The WebUI supports:

- prompt parser selection and parser-specific advanced settings
- shortcut profile selection and user parser presets
- a caret-aware prompt symbol palette
- shortcut profile editing, validation, import, and export
- prompt translation/preflight inspection
- raw, parser-input, and canonical diagnostic views
- REGION Builder handoff for base and hires positive prompts
- independent or inherited hires prompt parser/profile routing
- hires dimension and second-pass planning status

## Prompt validation

The **Validate** action performs prompt preflight and opens the prompt inspection
view. The inspector groups blocking errors, warnings, and notices and can show
semantic/source changes separately from canonical structure.

Generation also performs preflight before a request is queued. That automatic
preflight refreshes the preview information without forcing the inspector open.

## Symbol palette targeting

The prompt symbol palette can target the base positive/negative prompts or hires
positive/negative prompts explicitly. In automatic mode it follows the active
prompt field/caret and falls back to the last prompt target used.

## REGION Builder

The REGION Builder bridge uses the dimensions for the pass being edited. Base
prompt REGION editing uses base generation dimensions. Hires REGION editing uses
the current hires dimension plan when hires is enabled.

## Hires prompt routing

The hires pass can inherit the base parser/profile or use explicit second-pass
settings where the selected parser/profile supports them. The WebUI also shows
second-pass sampler/scheduler and dimension-plan status.

## Troubleshooting after an upgrade

If prompt controls appear stale after updating IMAGE_GEN, perform a normal page
reload first. R10 changed the browser module layout and the public prompt module
uses a new cache revision so current browsers should request the new files
automatically.

If a prompt feature still fails to initialize, inspect the browser developer
console for a failed request under:

```text
/assets/js/features/prompt/
```

A missing file in that directory indicates an incomplete source update rather
than a parser or checkpoint problem.


## Structured Classic prompt semantics

The shared Classic syntax used by Legacy, Parser21, SuperHybrid, and Combined routing now preserves semantic structure before text encoding.

```text
{red hair, green eyes}
```

creates a cohesive local group. Members have equal local influence by default. Explicit member weights are relative inside the group:

```text
{red hair:2, green eyes:1}
```

normalizes to two-thirds red-hair influence and one-third green-eyes influence inside that group. Grouping is not the same as increasing attention weight and is not equivalent to ordinary comma text.

Relationship syntax is structural:

```text
owner:::property::value!!
```

`:::` establishes owner/parent scope, `::` establishes the relation, and `!` / `!!` terminate the respective structure. These control characters are consumed before CLIP/OpenCLIP/T5 unless explicitly escaped by the user.

Legacy single-colon chains remain supported for compatibility. Grouping a chain changes its parent scope:

```text
{lake:sky:clouds}:::X
```

makes the whole `lake -> sky -> clouds` composition the parent. Without the braces:

```text
lake:sky:clouds:::X
```

`clouds` is the terminal attachment owner while the sequence still contributes through its local sequence scope.

Standard schedule/alternate forms are also compiled before text encoding:

```text
[cat:dog:7]
[cat:dog:0.5]
[cat:dog:50%]
[cat|dog]
```

Integer, fractional, percentage, weight, and quantity values are typed by their grammar location rather than by an integer-versus-decimal guess.

### Legacy BREAK

The Legacy Default profile now treats uppercase `BREAK` as a real encoder chunk boundary rather than ordinary prompt text. The same typed BREAK runtime is used across supported SD1, SDXL, and SD3/SD3.5 conditioning families. Other Legacy semantics are unchanged: Legacy `AND` still uses its historical normalized composition behavior, and `||` plus the new quote scopes are not enabled by this change.

For example:

```text
portrait BREAK city
```

encodes the two text segments independently for the fixed-context CLIP stream and joins them according to the active model-family BREAK contract.

## Semantic Structure inspector

Prompt Inspector now includes a **Semantic Structure** card for base and hires positive/negative prompts. It shows:

- semantic and structure digests;
- group members, raw weights, and normalized local percentages;
- owner/relation scope and parent-scope mode;
- schedules and activity windows;
- categorized warnings and safe fallbacks;
- encoder-visible text; and
- effective-final branch contribution for static prompts.

For temporal prompts, effective contribution is labeled **dynamic by step** instead of displaying a misleading fixed percentage.

## Semantic replay

New generations record a parser-neutral PPSR semantic replay contract containing PromptIR, the compiled conditioning plan, semantic/structure digests, parser/compiler contract versions, model-family degradation information, and any fallbacks.

Exact replay prefers this recorded semantic structure and validates the current compiler result against the recorded digest. It does not re-guess whether an old number meant a weight or a schedule step. Canonical-v1 records remain loadable through compatibility migration.

If the semantic record has been altered or the current compiler produces a different semantic digest, exact replay fails closed rather than silently generating with changed prompt meaning.

## Parser qualification

The normal parser development gate remains model-free:

```bat
run.bat parser-test
```

It now prints/saves numeric, temporal, parent-scope, effective-weight, and replay/cutover reports.

Real-checkpoint PPSR-08 qualification is separate because it intentionally generates images:

```bat
testing\test_validations\qualification\generation\ppsr08_prompt_parser_image_qualification.bat --model sdxl="C:\path\model.safetensors"
```

The qualification runner creates a unique `images/`, `requests/`, and `logs/` evidence run, compares `standing` with `{standing}`, verifies grouped/weighted prompts are image-distinct where expected, performs exact semantic-manifest replay, and creates a contact sheet for visual review.

## PPSR-09 Semantic Structure experiments

Prompt Inspector displays PPSR-09 experimental cohesive groups and modifier/target bindings. White-brace groups show their algorithm, members, normalized focus weights, and focus encoder text. Bindings show the source operator (`^` or `*`), target-only/subtree scope, inheritance-barrier state, and lowering algorithm. These diagnostics are experimental and do not replace the established `{...}` control until PPSR-10 decides the cutover.
