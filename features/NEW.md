# What's New in the Current ImageGen Alpha

This page highlights recent user-visible improvements present in the current source. It is intentionally shorter and more product-focused than the chronological changelog.

## SDXL, SD3 Medium, and SD3.5 Medium Generation

ImageGen's active txt2img architecture set now extends through **SDXL, SD3 Medium, and SD3.5 Medium** in addition to SD 1.x and qualified SD 2.x.

SDXL uses its architecture-specific dual-encoder/pooled-conditioning path and runtime profiles. SD3 Medium and SD3.5 Medium use the Flow Match transformer path with 16-channel latents, CLIP-L + CLIP-G conditioning, SD3-specific runtime profiles, and staged component residency.

The current source includes a dedicated `install_sd3_support.bat` setup helper for SD3-family runtime assets and shared text encoders. It does not download the user's main model checkpoint.

SD3 Medium and SD3.5 Medium normal txt2img generation have been verified with image-generation tests. Broader SD3 workflows such as Hires, SD3 LoRA application, Img2Img, and REGION/Canvas combinations remain separate qualification work.

## Advanced Models Component Composition

The WebUI now includes **Advanced Models** mode for building a generation composition from registry-fingerprinted components instead of requiring one monolithic checkpoint to define every active role.

The initial family contracts cover SD 1.x, SD 2.x, SDXL, and SD3/SD3.5 roles. Compatible model weights, VAEs, text encoders, and optional SD3 T5/T5XXL components can be selected by exact component identity.

The component registry distinguishes exact identity from source location. A component can be available as a standalone physical file, as a digital component inside one or more donor checkpoints, or through both source types without changing its fingerprint.

Auto selection is conservative: required roles are automatically resolved only when exactly one eligible compatible fingerprint exists. Ambiguous choices remain explicit rather than being guessed from filenames.

## Persistent Generation Queue

Recoverable generation queues now survive application restarts.

ImageGen persists queue ordering, individually paused queued jobs, whole-queue hold state, and recoverable multi-image progress. If the application closes while a generation is active, that job can be restored at the front of the queue, but the queue starts held so reopening ImageGen does not immediately restart GPU work without an explicit Resume.

Completed, failed, and normally cancelled jobs are not requeued.

## Runtime and Replay Architecture Hardening

Several large runtime areas have been decomposed without intentionally changing their generation semantics.

The KES scheduler now separates schedule construction, cache behavior, randomization, blending, stabilization, and diagnostics. The txt2img runtime separates model preflight, pipeline construction, residency, and request execution behind the existing `Txt2ImgRunner` surface. Catalog/replay work now uses domain-specific catalog services and a shared preflight-token store instead of duplicating token lifecycle code across replay, batch replay, batch import, and variation workflows.

These changes primarily improve maintainability and reduce the risk of future feature work crossing unrelated runtime responsibilities.

## Stable Diffusion 2.x Generation

Qualified **Stable Diffusion 2.x** checkpoints can now use ImageGen's normal generation runtime through a dedicated OpenCLIP conditioning path and SD 2.x runtime-profile contract.

SD 2.x no longer reuses SD 1.x text-conditioning assumptions. The runtime validates the 1024-wide OpenCLIP conditioning contract, resolves SD 2.x prediction/runtime profiles, and keeps architecture qualification explicit.

SDXL has since moved into the active generation runtime; this SD 2.x section is retained as the earlier architecture-expansion milestone.

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
