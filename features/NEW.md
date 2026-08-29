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

LoRA compatibility now distinguishes Stable Diffusion family, adapter format, target components, and actual runtime-loader support instead of assuming that one family label proves compatibility.

The standard adapter layer now has architecture-aware target contracts for SD1, SD2, SDXL, and SD3-family transformer/text-encoder layouts. For qualified paths, user weight `1.0` means the adapter's normal/native effect after loader-internal rank/alpha normalization.

**SD3 / SD3.5 LoRA application is not currently claimed as supported.** The architecture mapping groundwork exists, but a suitable real SD3-family LoRA has not yet been available for controlled end-to-end qualification.

ImageGen can also recognize formats such as LoHa and LoKr without pretending the standard loader can execute them. Unsupported or unsafe-to-inspect formats are reported/restricted rather than being silently routed through a generic fallback.

## Asset Browser / Asset Hub — Experimental

ImageGen now exposes a substantially expanded Asset Browser on top of the provider-neutral Asset Hub, with **Civitai as the first provider**.

The current workflow supports independent search tabs, pausing/resuming provider fetches without losing results/cursors, continuous or manual paging, a persistent local discovery index, local filtering, staged preview loading, detailed model/version/file inspection, saved-for-later assets, provider gallery caching, managed downloads, verification, classification, automatic safe installation, quarantine, and provenance.

The Download Manager adds bounded concurrency, queue limits, bandwidth limits, provider request spacing, retries, pause/resume/cancel controls, restart recovery, history/cleanup tools, and verified partial-transfer resume when safe HTTP Range continuation can be proven.

> **Experimental — active bug testing.** Checkpoint and LoRA discovery/download/install behavior is available for testing but is still receiving corrective work around search lifecycle, previews, file/version selection, interrupted transfers, automatic installation, and library reconciliation.

## Help Center

The WebUI now includes a searchable Help Center backed by the public `help_documentation/` tree.

Topics can provide category navigation, related guides, local images/video, and explicit external links. The Home Changelog now uses the same shared Markdown viewer instead of maintaining a separate document renderer.

## Theme Manager

Theme Manager now exposes semantic appearance roles for application surfaces, component/card surfaces, borders, accents, primary text, and secondary text.

The editor reports contrast diagnostics without silently changing the user's colors. Low-contrast themes remain allowed, with a confirmation warning enabled by default.

Local theme-package import is also available with validation, explicit activation, and safety checks that reject executable/script content and unsafe package paths.

## Workspace Manager, Resizing, and Shared Overlays

Workspace Manager continues to control registered page components using portable base layouts and responsive Wide/Standard/Compact/Narrow presentations.

Registered components can now expose persistent drag/keyboard resizing. A reusable workspace overlay capability also supports drawer and focused presentation, click-outside/Escape collapse, resizable drawer width, and an edge restore tab.

Asset Details is the first major consumer of the shared drawer/focus behavior, but the capability is implemented in the shared component/workspace registry for reuse by other workspaces.

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

Queued work can be paused and resumed per item, and active multi-image work can pause at a safe image boundary.

Paused/restart-recovered queued jobs can now be cancelled even when no live worker is attached. Runtime-event handling also protects cancelled jobs from being revived by late/buffered events, and watchdog accounting avoids treating computer sleep/event-loop suspension as normal generation stall time.

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

## Improved Hires Recovery and Prompt Inheritance

When Hires has no valid upscaler selected, the UI focuses the affected Hires control and provides recovery guidance instead of relying only on a detached corner error.

The preferred Hires upscaler is preserved as a user preference across unrelated replay actions.

Hires prompt inheritance has also been hardened: leaving the Hires prompt blank continues to mean **inherit the current base prompt**, including after cancellation/edit/replay workflows, instead of allowing an old copied prompt to persist as a hidden override.

## Better REGION Selection

New REGION boxes prefer nearby free/low-overlap placement when possible, and overlapping/contained regions can be selected through a region-stack list so buried boxes are easier to edit.

## In-Program User Configuration Editing

`user_config/user-config.yml` can now be edited from inside ImageGen. YAML is validated before save, the write is atomic, and the previous file is retained as a backup.


## PNG, Lossless WebP, and Embedded Replay Metadata

Generated images can now be saved as PNG or lossless WebP. ImageGen can embed either full replay metadata or compatibility-oriented parameter metadata directly in the image.

Full WebP replay uses XMP plus EXIF-compatible parameter text. If the local Pillow/libwebp runtime cannot persist the XMP payload, ImageGen preserves compatibility metadata and reports a warning rather than silently claiming the full replay record was embedded.

Image Details has also been expanded to surface replay-essential information more clearly, including base/final dimensions and Hires inheritance state.

## Phase-Aware Live Preview and CFG Telemetry

Live progress/CFG tracing now keeps base generation and Hires refinement as distinct passes. Step counts, active phase, previews, and effective CFG trajectories can therefore transition into Hires without presenting the second pass as a continuation of the base denoising sequence.

## Replayable and Inspectable Prompt Parser Semantics

The reconstructed Classic prompt semantics are versioned and replayable. New generations can record PromptIR, the conditioning plan, semantic/structure digests, parser/compiler contracts, model-family semantic state, and safe-fallback diagnostics instead of relying only on visible punctuation.

Prompt Inspector includes a Semantic Structure view with parsed group weights, owner/relation scope, schedules, fallbacks, encoder-visible text, and effective-final weights for static prompts. The parser test gate provides model-free replay/cutover evidence, while separate opt-in real-checkpoint runners can create image/request/log evidence and contact sheets for image-level qualification.

Relationship scopes, owner sequences, structural terminators, typed numeric values, schedules/alternates, nested parent scope, model-family conditioning contracts, semantic replay, and inspection have all received substantial updates.

### Experimental grouping and binding A/B tests

The latest parser update adds an explicit alternative to the existing `{...}` grouping behavior rather than silently replacing it:

```text
{...}             existing grouping control
⦃...⦄             experimental cohesive grouping
modifier^target   target-only binding
modifier*target   target + descendant binding
```

The cohesive-group candidate keeps the group's shared context present while locally reinforcing each member. The binding operators test whether attributes can be attached more reliably to the intended concept: `^` applies only at the explicit target, while `*` can propagate through structural descendants until an explicit child binding creates a new barrier/scope.

These additions are **experimental**. They are intended for same-seed, multi-image A/B qualification focused on attribute attachment, unwanted color/concept leakage, composition stability, and diversity. They are not yet final prompt-language guarantees.

