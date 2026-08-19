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
