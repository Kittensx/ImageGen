# What's New in the Current ImageGen Alpha

This page highlights recent user-visible improvements present in the current source. It is intentionally shorter and more product-focused than the chronological changelog.

## Stable Diffusion 2.x Generation

Qualified **Stable Diffusion 2.x** checkpoints can now use ImageGen's normal generation runtime through a dedicated OpenCLIP conditioning path and SD 2.x runtime-profile contract.

SD 2.x no longer reuses SD 1.x text-conditioning assumptions. The runtime validates the 1024-wide OpenCLIP conditioning contract, resolves SD 2.x prediction/runtime profiles, and keeps architecture qualification explicit.

SDXL remains planned for base-model generation.

## Stronger LoRA Compatibility and Inspection

LoRA compatibility now distinguishes:

- Stable Diffusion family;
- adapter format;
- target components; and
- actual runtime-loader support.

The standard adapter loader covers conventional Kohya/Diffusers/PEFT/up-down LoRA representations when their targets map cleanly to supported model components.

ImageGen can also recognize formats such as LoHa and LoKr without pretending the standard loader can execute them. Unsupported formats are reported clearly instead of being silently routed through an unsafe generic fallback.

## Asset Hub

ImageGen now includes a provider-neutral Asset Hub, with **Civitai as the first provider**.

The current flow supports:

```text
discover
-> stage download
-> verify size/hash
-> classify
-> install or quarantine
-> record provenance
```

Downloads remain outside live model folders until verification and installation succeed. Installed assets retain local provenance so useful provider/model metadata can remain available offline.

## Help Center

The WebUI now includes a searchable Help Center backed by the public `help_documentation/` tree.

Topics can provide category navigation, related guides, local images/video, and explicit external links. The Home Changelog now uses the same shared Markdown viewer instead of maintaining a separate document renderer.

## Theme Manager

Theme Manager now exposes semantic appearance roles for application surfaces, component/card surfaces, borders, accents, primary text, and secondary text.

The editor reports contrast diagnostics without silently changing the user's colors. Low-contrast themes remain allowed, with a confirmation warning enabled by default.

Local theme-package import is also available with validation, explicit activation, and safety checks that reject executable/script content and unsafe package paths.

## Workspace Manager and Responsive Layouts

Workspace Manager now controls registered page components using portable base layouts.

Wide, Standard, Compact, and Narrow presentations derive from the same saved layout based on the actual workspace width rather than forcing users to maintain separate layouts for every display size.

## Better Seed and Parameter Randomization

Advanced Seed controls now live directly beneath Seed and can switch between:

- sequential;
- random; and
- random-within-range behavior.

The Seed expression and structured range controls synchronize. Range forms such as `[5000,15000]`, `-1 [5000,15000]`, and `-1, [5000,15000]` are accepted and normalized.

Eligible numeric settings also have a shared advanced range editor with random min/max values, optional integer resolution, and independent hard bounds. CFG Scale uses CFG-appropriate examples and runtime locks instead of inheriting Seed-specific values.

## CFG Lab Presets

CFG Lab can now save user presets inside ImageGen and import/export preset files. CFG-related random-range state can travel with those presets.

Effective CFG min/max locks are enforced by the guidance runtime rather than existing only as visual UI limits.

## More Capable Queue Control

Queued work can now be paused and resumed per item, and active multi-image work can pause at a safe image boundary.

Paused jobs can coexist while other non-paused work continues. Queued items can be moved higher or lower without silently taking over the active generation.

Finite and continuous generation expose clearer progress such as `2 of 20` and `2 of ∞`.

## Safer Replay Preferences

Replay now keeps a stronger boundary between generation state and user-owned operational preferences.

Historical runs no longer automatically re-enable TXT/diagnostic sidecars, low-resolution Hires artifacts, or overwrite the user's preferred Hires upscaler merely because an older run used different output settings.

Current defaults favor the compact replay record:

- TXT sidecar off;
- compact replay JSON on;
- diagnostics JSON off; and
- exact low-resolution Hires base artifact off.

## Faster Variation Matrix Workflow

Variation Matrix still validates expanded jobs internally, but users no longer need to manually create a separate `validated job` state before queueing the previewed Cartesian expansion.

## Improved Hires Recovery

When Hires has no valid upscaler selected, the UI now focuses the affected Hires control and provides recovery guidance instead of relying only on a detached corner error.

The preferred Hires upscaler is also preserved as a user preference across unrelated replay actions.

## Better REGION Selection

New REGION boxes prefer nearby free/low-overlap placement when possible, and overlapping/contained regions can be selected through a region-stack list so buried boxes are easier to edit.

## In-Program User Configuration Editing

`user_config/user-config.yml` can now be edited from inside ImageGen. YAML is validated before save, the write is atomic, and the previous file is retained as a backup.
