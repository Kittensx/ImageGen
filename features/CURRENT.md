# Current ImageGen Features

This page describes capabilities present in the current ImageGen source/runtime. It is a feature inventory, not a chronological changelog.

ImageGen is still an alpha application, so “available” means implemented in the current alpha rather than frozen or guaranteed API compatibility.

## 1. Native Stable Diffusion Architecture Support

ImageGen runs its own modular Stable Diffusion generation pipeline rather than launching another image-generation WebUI as its backend.

The current enabled text-to-image architecture boundary is:

- **Stable Diffusion 1.x** using qualified full monolithic `.safetensors` checkpoints;
- **qualified Stable Diffusion 2.x** checkpoints using the dedicated OpenCLIP/runtime-profile path;
- **Stable Diffusion XL (SDXL)** through the qualified dual-tokenizer/dual-text-encoder, pooled-conditioning, added-conditioning/time-ID, and SDXL runtime-profile path;
- **Stable Diffusion 3 Medium** using the SD3 Flow Match transformer/16-channel latent contract; and
- **Stable Diffusion 3.5 Medium** using the corresponding qualified SD3.5 Medium runtime profile.

The shared runtime handles:

- checkpoint inspection and architecture validation;
- architecture-specific model/component loading;
- positive and negative conditioning;
- seeded latent creation;
- sampler/scheduler execution;
- CFG/guidance handling;
- VAE decode;
- output saving and metadata generation; and
- replay/provenance capture.

### SD 1.x conditioning

SD 1.x uses the established single-CLIP path with 768-wide text conditioning and the SD 1.x model/component contract.

### SD 2.x conditioning

Qualified SD 2.x generation uses a separate OpenCLIP path instead of reusing SD 1.x text-encoder assumptions.

The SD 2.x runtime includes:

- local OpenCLIP tokenizer/text-encoder assets;
- 1024-wide conditioning validation;
- 77-token context validation;
- checkpoint-derived text-encoder conversion/loading;
- explicit SD 2.x runtime profiles;
- prediction-type resolution;
- known checkpoint qualification evidence; and
- optional explicit profile override when automatic evidence is insufficient.

Runtime-profile definitions cover SD 2.0/2.1 base and 768-oriented families, but an arbitrary checkpoint is not accepted solely because its filename or provider metadata says “SD2.” ImageGen still requires the model to satisfy the active qualification contract.

### SDXL conditioning and runtime profiles

SDXL generation is active in the current runtime. The SDXL path includes:

- dual tokenizer and dual text-encoder handling;
- pooled prompt embeddings;
- added conditioning/time IDs;
- SDXL-specific UNet invocation;
- architecture/runtime preflight;
- base SDXL runtime profiles; and
- profile-aware recommendations for specialized families such as Lightning and Turbo without silently replacing user choices.

Individual specialized SDXL checkpoints can still have profile-specific requirements. Architecture support does not mean that every community checkpoint or secondary workflow has been independently qualified.

### SD3 Medium and SD3.5 Medium

Normal WebUI txt2img generation is available for **SD3 Medium** and **SD3.5 Medium**. These paths have been verified with generated-image tests in the current source snapshot.

The SD3-family runtime includes:

- SD3.x checkpoint architecture detection;
- Flow Match denoising semantics;
- transformer rather than UNet execution;
- 16-channel latent/VAE handling;
- qualified runtime profiles for `sd3_medium` and `sd3_5_medium`;
- CLIP-L and CLIP-G conditioning;
- embedded-or-shared text-encoder source resolution;
- `flow_euler` sampler support;
- `flow_match_euler` scheduler support;
- architecture-specific preflight and recommendation handling; and
- staged residency so text encoders, transformer, and VAE do not need to remain simultaneously GPU-resident while idle.

The dedicated support installer can provision runtime configuration/tokenizer assets and shared text encoders without downloading the user's main model checkpoint:

```bat
install_sd3_support.bat
```

The normal WebUI path currently uses the qualified CLIP-L + CLIP-G mode. T5/T5XXL exists as an optional component path for Advanced Models, but ordinary SD3 WebUI T5 selection is not yet presented as a fully general default workflow.

### Advanced Models and component composition

ImageGen also has an **Advanced Models** mode for building a generation composition from registry-fingerprinted components instead of treating one checkpoint filename as the complete model identity.

Current family contracts cover:

- SD 1.x: model/UNet weights, VAE, and text encoder;
- SD 2.x: model/UNet weights, VAE, and text encoder;
- SDXL: model/UNet weights, VAE, text encoder 1, and text encoder 2; and
- SD3/SD3.5: transformer/model weights, VAE, CLIP-L, CLIP-G, and optional T5/T5XXL.

Advanced Models uses exact component fingerprints and registry evidence. A physical filename or donor-checkpoint name is a source-location hint, not compatibility proof. Digital checkpoint components and standalone components can satisfy the same exact component identity when their fingerprints and provider role contracts match.

Auto selection is intentionally conservative: a required role is selected automatically only when exactly one eligible compatible component fingerprint is available. Ambiguous roles require an explicit user choice.

Advanced Models selections persist into generation settings, replay, batch import/export, queue composition, and output manifests by component identity rather than by filename alone.

## 2. Exact Requested Dimensions

Users can request non-standard widths and heights instead of being restricted to a simple eight-pixel UI increment.

ImageGen resolves the model-compatible internal geometry required for generation and preserves the requested final output dimensions in the generation workflow and replay data.

This exact-size behavior also feeds Hires and Canvas Expansion workflows.

## 3. Samplers and Schedulers

The current built-in sampler paths include:

- **KES**;
- **DPM++ 2M**; and
- **Simple Euler**.

Current scheduler paths include:

- **Simple KES**; and
- **Standard Karras**.

Sampler and scheduler capabilities are described through runtime registries so incompatible feature combinations can be rejected or normalized explicitly rather than silently changing behavior.

## 4. Prompt Processing and Prompt Authoring

ImageGen has multiple prompt-processing paths and user-facing prompt tools.

Current source includes:

- the established ImageGen Legacy parser;
- Prompt Parser 21;
- **SuperHybrid** prompt parsing/authoring;
- combined/automatic parser routing;
- prompt shortcut profiles;
- parser presets;
- prompt validation and preview tools;
- configurable parser settings;
- prompt symbol/authoring helpers in the WebUI;
- a parser-neutral PromptIR boundary for the active Legacy structured syntax;
- canonical prompt contracts with serialized semantic IR and backward-compatible loading;
- tensor-free structured conditioning plans that consume `::`, `:::`, `!`, and `!!` before encoder calls;
- real `::` relation and `:::` owner-sequence compilation with typed sequence-local weights/activity windows and structural `!`/`!!` terminators;
- shared Classic relationship semantics across Legacy, Parser21, and SuperHybrid while preserving parser-specific extension grammars;
- typed numeric interpretation by grammar context;
- deterministic schedules and alternates resolved against the active generation pass;
- recursive/nested owner-scope handling;
- parser-neutral semantic replay/inspection records;
- experimental cohesive grouping using `⦃...⦄` alongside the existing `{...}` control behavior;
- experimental target-only `^` and inheriting/subtree `*` attribute bindings; and
- model-free parser validation through `run.bat parser-test`.

The parser's core relationship, sequence, terminator, numeric, temporal, replay, and inspection behaviors can be tested independently of the current grouping/binding experiment.

### Grouping experiment

Existing `{...}` grouping remains available as the comparison/control implementation. It uses the established branch-average group-conditioning behavior and should **not** be interpreted as the final answer to ImageGen's closer-concept-binding goal.

A second syntax, `⦃...⦄`, is now available as an experimental cohesive-group candidate. Its current algorithm keeps the group's shared context present in each encoder branch while locally reinforcing one member at a time. Group-local explicit weights remain relative inside the group.

The two forms intentionally coexist so fixed-seed image tests can compare semantic cohesion, concept leakage, composition stability, and diversity before a final grouping behavior is chosen.

### Attribute-binding experiment

Two experimental binding operators are implemented for image-level qualification:

```text
modifier^target   -> bind the modifier to this target only
modifier*target   -> bind the modifier to this target and its structural descendants
```

`^` acts as a local inheritance barrier: it blocks an inherited `*` modifier at that target and does not start a new descendant scope. `*` also blocks an inherited ancestor modifier at the explicit target, then establishes its own modifier for structural descendants. An explicit child `^` or `*` therefore overrides an inherited parent binding at that child.

The first lowering algorithm reinforces a modifier/target pair together rather than sending the modifier as a separate conditioning branch. For example, `red^hair` is lowered as a paired concept rather than as an independent `red` branch. This is an experiment in semantic attachment, not a promise of hard symbolic control.

Prompt Inspector can expose cohesive groups, normalized local focus weights, binding source operators, target/subtree scope, inheritance barriers, and the current lowering algorithm. Escaped `\^`, `\*`, `\⦃`, and `\⦄` remain literal text.

These new grouping and binding semantics are **experimental and under active A/B image testing**. They should not be treated as finalized prompt-language guarantees yet.

## 5. REGION / Regional Prompting Tools

ImageGen includes regional-prompting integration and a dedicated Region Builder surface for defining region-aware conditioning.

The current tooling includes:

- region geometry;
- regional positive/negative prompt data;
- weights;
- timing/start-stop behavior;
- blending controls;
- generation integration;
- preference for nearby empty/low-overlap placement when creating a new region; and
- a region-stack selector for choosing boxes that overlap or contain one another.

Manual editing can still create intentional overlap. The placement preference is a quality-of-life aid rather than a hard non-overlap rule.

REGION remains an advanced feature and should be treated separately from ordinary global prompting and from Canvas Expansion.

## 6. LoRA and Adapter Compatibility

LoRA is active in the current generation runtime on qualified paths.

Current general capabilities include:

- LoRA discovery;
- weighted application;
- multiple LoRAs in a single request;
- structured LoRA request records;
- inline LoRA syntax normalization;
- checkpoint-family compatibility checks;
- SHA-256 identity/provenance;
- activation text and sidecar metadata support;
- LoRA information in generation/replay records; and
- a dedicated WebUI LoRA workspace.

The compatibility layer separates four questions that are easy to conflate:

1. **Family** — which Stable Diffusion architecture the adapter targets.
2. **Format** — the adapter representation detected from metadata/tensor keys.
3. **Targets** — which denoiser/text-encoder components the adapter expects.
4. **Runtime support** — whether the current build has a qualified loader for that exact combination.

### Standard loader

The registered standard Diffusers/PEFT path handles conventional representations classified as:

```text
standard_kohya_lora
standard_diffusers_peft_lora
standard_lora_up_down
```

Conventional linear and supported convolutional LoRA targets can use the standard path when converted keys map cleanly to qualified model components.

Architecture-aware target contracts now include:

- **SD 1.x:** UNet + text encoder 1;
- **SD 2.x:** UNet + text encoder 1, without applying SD 1-specific text-encoder shape assumptions;
- **SDXL:** UNet + TE1 + TE2; and
- **SD3-family groundwork:** transformer + TE1 + TE2 + TE3 target mapping.

The SD3-family mapping is **unverified groundwork, not a support claim**. A suitable real SD3/SD3.5 LoRA has not yet been available for controlled end-to-end qualification, so SD3-family LoRA application should not currently be advertised as supported.

For qualified standard paths, the UI weight contract is normalized around native adapter behavior: **`1.0` means the adapter's normal/native effect after loader-internal rank/alpha normalization**. User weight then multiplies that native effect, so users do not need to know a format-specific internal scaling constant.

### Unsupported or partial adapter formats

ImageGen can identify LyCORIS-style formats such as **LoHa** and **LoKr**, but the current standard loader does not execute those algorithms. They are reported as unsupported rather than silently passed through a generic fallback.

DoRA magnitude data is also inventoried separately and is currently treated as requiring a dedicated qualified runtime path.

Potentially unsafe legacy serialized adapter files can be classified as inspection-restricted instead of being loaded merely to identify them.

Checkpoint-like Safetensors files found in the LoRA library are marked as misclassified rather than attempted as adapters.

The WebUI also contains Civitai-oriented LoRA metadata support, including hash-based matching and preview/metadata workflows where the required network/API configuration is available.

Default LoRA folder:

```text
models\StableDiffusion\Lora
```

## 7. Neural Hires / Second-Pass Generation

Hires is implemented as a pixel-neural second-pass workflow.

The active path is conceptually:

```text
base txt2img denoise
-> VAE decode to image
-> neural .pth upscale
-> exact target-size preparation
-> VAE encode
-> second denoising/refinement pass
-> final VAE decode
```

Current Hires tooling includes:

- supported neural `.pth` upscaler discovery;
- ESRGAN/RealESRGAN-oriented model roots;
- architecture/scale qualification;
- exact target dimensions or scale-based sizing;
- tiled neural upscaling;
- tile overlap and tile batch controls;
- bounded OOM recovery for eligible tiled workloads;
- separate Hires steps and denoising strength;
- sampler/scheduler overrides;
- CFG overrides;
- independent Hires prompt routing;
- optional preservation of intermediate/base artifacts;
- upscaler and VAE provenance in generation records;
- replay validation against recorded asset identity;
- persistent preferred-upscaler behavior; and
- focused UI recovery when Hires is enabled without a valid selected upscaler.

The exact low-resolution base artifact is no longer enabled by default. It remains an optional output preference.

Hires is available, but it is still an **alpha feature** and is sensitive to upscaler compatibility, target size, memory pressure, VAE behavior, and sampler/scheduler qualification.

## 8. Canvas Expansion / Shape Adaptation

ImageGen has an alpha generative Canvas Expansion workflow.

The goal is to preserve an existing composition and generate the missing space required by a larger canvas rather than stretching the source image.

Two user-facing source patterns are present:

### Expand Existing Image

A user can load an existing image, choose a larger target canvas, select placement, and generate the newly exposed area.

### Expand After Generation

A fresh txt2img result can be expanded inside the same generation flow before the expanded result becomes the primary saved output. When geometry permits, the runtime can qualify live latent reuse; otherwise it can use an explicit image/VAE re-encode handoff.

Current controls and runtime records cover areas such as:

- target width and height;
- square or custom targets;
- source anchor/placement;
- protected source region;
- preserve/feather/generate mask semantics;
- feather width;
- expansion denoising strength;
- Edge Pad context seeding;
- Reflect Pad as an advanced alternative;
- extension positive/negative prompts;
- source/extension prompt composition policy;
- source handoff mode;
- latent alignment reporting;
- geometry and inference fingerprints; and
- replay/provenance data.

Canvas Expansion is intended to be useful even before general Img2Img exists, but it is deliberately treated as an **intermediate composition-expansion stage**. It is not advertised as a complete final-image restoration or redraw system.

## 9. Local WebUI

`run_webui.bat` launches the local browser interface.

The current WebUI includes dedicated workspaces and controls for areas such as:

- Generation;
- checkpoint selection and model status;
- LoRA browsing/selection;
- Asset Hub;
- prompts and parser tools;
- exact dimensions;
- Hires;
- Canvas Expansion;
- CFG/guidance controls;
- live generation progress;
- queue status;
- recent outputs;
- replay;
- output details;
- variation tools;
- memory/runtime information;
- Help Center;
- Theme Manager;
- Workspace Manager; and
- in-program user-configuration editing.

The server binds to localhost and is designed for a local single-user workflow.

## 10. Live Preview, Progress, and Guidance Telemetry

The runtime can report generation progress independently from final image output.

Current telemetry includes items such as:

- active step and total steps;
- completion progress;
- elapsed and per-step timing;
- active seed/model/sampler/scheduler information;
- requested/effective CFG behavior;
- decoded preview frames when enabled;
- runtime stage information; and
- pass-aware tracing that keeps base-generation and Hires-refinement step/CFG progress distinct.

Preview decoding can be throttled or suspended by memory policy without disabling the underlying generation progress/CFG telemetry. Hires transitions can therefore report the active refinement pass without incorrectly presenting base-pass step counts or guidance trajectories as if they were one continuous denoising sequence.

## 11. CFG Lab, Presets, Randomization, and Guidance Controls

The KES/guidance path includes more than a single flat CFG number.

Current source supports configurable guidance shaping such as:

- classic guidance;
- low-CFG shaping;
- sigma/step-shaped guidance;
- multiple curve styles;
- early/high-sigma adjustments;
- late-step tapering;
- CFG rescale; and
- seed-locked CFG comparison/sweep tooling.

CFG Lab also supports user presets that can be saved inside ImageGen and imported/exported as preset files.

Advanced numeric range support can randomize CFG-related numeric settings per generated image. Hard minimum/maximum locks are separate from the random range and can clamp the effective runtime CFG trajectory.

The shared range editor is field-aware. CFG Scale uses CFG-appropriate examples such as `[5.0, 7.5]` rather than Seed-sized example values.

## 12. Generation Queue and Batch Workflows

The WebUI includes a local job queue and batch-oriented tools.

Current functionality includes:

- queued/running/paused/completed/failed/cancelled states;
- active-job cancellation;
- cancellation of paused queued jobs, including restart-recovered held work that has no live worker attached;
- per-item pause and resume;
- pause-after-current-image boundaries for active multi-image work;
- multiple held/paused jobs;
- skipping paused queue items while other work remains schedulable;
- moving queued items higher or lower without displacing the active generation;
- persistence of recoverable queued work across application sessions;
- restoration of explicit queue order, individually paused queued jobs, and whole-queue hold state;
- safe recovery of an interrupted active job at the front of a held queue so reopening ImageGen does not unexpectedly restart GPU work;
- guards that prevent buffered/late runtime events from reviving work after cancellation;
- watchdog timing that avoids treating computer sleep/event-loop suspension as ordinary generation stall time;
- finite progress such as `2 of 20`;
- continuous-generation progress such as `2 of ∞`;
- queue filtering;
- recent run information;
- batch size and batch count;
- continuous generation;
- queue import/export;
- JSON, JSON Lines, and CSV request workflows;
- request remapping/validation;
- queue composition from prior outputs; and
- seed-strategy selection for sequential, random, and bounded-random batches.

Finite random seed ranges can avoid duplicates until their available values are exhausted, and already consumed values are preserved when a job pauses/resumes.

## 13. Replay and Variation Matrix

ImageGen records generation state so completed outputs can be inspected and reused.

Current replay tooling can:

- reconstruct a generation request from ImageGen output metadata;
- preflight required assets/settings before queueing;
- report missing or changed assets instead of silently substituting them;
- restore prior generation settings to the form; and
- send compatible requests back to the normal generation queue.

Replay intentionally does **not** treat every historical UI/output setting as reproducibility state. User-owned operational preferences such as TXT/JSON/diagnostic sidecars, low-resolution Hires artifact saving, and preferred Hires upscaler behavior remain current-user preferences unless they are genuinely required for exact generation reproduction.

The Variation Matrix can expand one or more requests through controlled combinations of settings and seed policies before submission. Preview/queue behavior transparently performs the required validation/revalidation so users do not need to manage a separate `validated job` state manually.

## 14. Output Gallery and Image Details

Recent Outputs provides a local output browsing surface with image details and replay integration.

Current source includes:

- thumbnail browsing;
- lightbox viewing;
- keyboard navigation;
- metadata/details inspection;
- replay-essential summaries that distinguish requested/base dimensions from final Hires dimensions;
- clearer base-vs-Hires prompt inheritance presentation;
- embedded replay/compatibility metadata detection for PNG/WebP;
- filters;
- multi-image selection;
- loading prior generation settings;
- replay preflight; and
- handoff into queue/variation workflows.

A larger durable Gallery/asset-library system is separately planned and should not be confused with the current Recent Outputs browser.

## 15. Replay and Output Storage

The output pipeline separates replay-essential data from deeper diagnostics and can save generated images as **PNG** or **lossless WebP**.

The normal structured sidecar uses a compact replay serialization profile. It is designed to keep the generation inputs and reproducibility identity required for replay while pruning duplicated or execution-only structures.

Image files can also carry embedded metadata in two modes:

- **full replay** — stores the ImageGen replay record in the image when the format/runtime supports it; or
- **compatibility** — stores conventional parameter text intended to remain useful to compatible external metadata readers.

PNG stores the embedded record in PNG text metadata. WebP uses XMP for the full ImageGen replay payload and EXIF-compatible parameter metadata for compatibility. If the local WebP runtime cannot persist XMP, ImageGen preserves compatibility metadata and emits a warning instead of silently claiming full embedded replay.

Current storage cleanup includes behavior such as:

- removing empty records;
- avoiding duplicate prompt-asset copies;
- pruning non-generation scheduler fields;
- avoiding redundant schedule representations when the authoritative representation is already stored;
- keeping compact runtime fingerprints instead of the entire conformance snapshot; and
- pruning duplicated diagnostic/runtime structures in the deeper diagnostics sidecar.

The current default output preferences are:

- TXT sidecar: **off**;
- compact replay JSON: **on**;
- diagnostics JSON: **off**; and
- exact low-resolution Hires base artifact: **off**.

These are user-owned preferences rather than exact replay requirements.

The save path uses temporary files and final atomic replacement so an interrupted save does not intentionally expose half-written output sets.

## 16. Memory-Aware Runtime

ImageGen contains explicit memory-management and component-residency behavior rather than leaving every component permanently resident on the GPU.

Current runtime capabilities include:

- memory profiles;
- VRAM safety margins;
- component retention/offload policies;
- attention slicing;
- VAE slicing/tiled work;
- CPU VAE fallback paths where supported;
- preview suspension under memory pressure;
- pre-Hires cleanup;
- physical GPU memory and PyTorch allocator telemetry;
- stage-aware residency information; and
- bounded CUDA OOM recovery.

Hires uses explicit stage transitions so the pipeline can release or move components as it passes through decode, neural upscale, re-encode, refinement, and final decode.

## 17. Attention Backends

The runtime supports selectable attention paths including:

- qualified xFormers;
- PyTorch scaled-dot-product attention; and
- eager/compatibility paths.

The public installer is profile-driven because optimized attention stacks can be hardware and environment specific.

## 18. Diagnostics and Reproducibility

ImageGen records detailed information for troubleshooting and replay without requiring the root README to document every internal field.

Depending on settings and workflow, diagnostics can include:

- normalized request/configuration;
- model and asset identity;
- sampler/scheduler information;
- stage timing;
- memory telemetry;
- runtime events;
- Hires and Canvas Expansion details;
- output/replay fingerprints; and
- failure context.

Deep diagnostic modes are intended for troubleshooting and can add overhead.

## 19. CLI and Config-Driven Generation

ImageGen is not WebUI-only.

Current launch paths include:

```bat
run.bat
```

for interactive CLI generation, and:

```bat
run_config.bat
```

for generation driven by the saved configuration/request format.

The WebUI, CLI, replay, and saved-request flows share the same underlying generation contracts rather than maintaining unrelated generation engines.

## 20. Asset Browser / Asset Hub — Experimental

Asset Hub is the provider-neutral system for discovering, downloading, verifying, classifying, installing, and tracking model-related assets. The current WebUI exposes this through the **Asset Browser**, with Civitai as the first supported provider.

> **Status:** Experimental — active bug testing. The workflow is implemented and available for testing, but search, previews, download recovery, classification, automatic installation, and library reconciliation are still receiving corrective qualification during alpha development.

### Search sessions and local discovery

The Asset Browser supports independent search tabs/sessions rather than one destructive global search. Starting another search can pause the previously active provider fetch while preserving that tab's results and continuation cursor, and paused searches can be resumed later. Continuous provider paging and manual paging are both available.

Provider results are retained in a local discovery index so previously discovered candidates can be filtered locally without re-querying the provider for every UI change. Current filtering/presentation work includes architecture/base-model information, support/library state, creator/provider metadata, preview availability/source, maturity/rating information, and installed/not-in-library style filtering.

Preview retrieval is staged separately from provider-page fetching. Resolved cards are published in small batches as previews become available instead of flooding the visible grid with large runs of blank cards.

### Asset details, previews, and saved items

Asset Details can inspect provider models, versions, available files, previews, compatibility/support information, and local-library state. Provider galleries can be cached according to configurable retention/size policies. Users can also keep assets in a local **Saved for Later** list.

### Provider discovery and authentication

Provider credentials are backend-owned. The browser can learn whether a credential is configured, but stored credentials are not returned after submission.

Civitai authentication can use supported session/environment/OS credential sources. Credentials are attached only to the expected provider host and are not forwarded to unexpected redirect destinations.

### Download Manager and staging

Downloads first enter a managed staging area rather than writing incomplete transfers directly into live checkpoint/LoRA/VAE/upscaler folders.

Current controls include:

- bounded simultaneous downloads;
- maximum queued downloads;
- bandwidth limiting;
- provider request spacing;
- retry limits;
- per-transfer and global pause/resume/cancel controls;
- restart recovery;
- download history and cleanup tools;
- stale partial cleanup;
- file-size verification; and
- SHA-256 verification.

Interrupted provider/CDN transfers can retain partial bytes and continue with a correct HTTP Range response when source identity and range/integrity checks prove continuation safe. Otherwise the partial is restarted/discarded instead of being blindly appended.

### Classification and installation

After verification, the normal managed flow can automatically continue into classification and library finalization when a safe destination is established.

ImageGen reuses its own technical inspectors for classification. Current paths include:

- LoRA Safetensors inspection;
- checkpoint Safetensors inspection;
- ESRGAN/RealESRGAN-oriented `.pth` upscaler inspection;
- recognizable Safetensors VAE layouts; and
- safely readable Safetensors textual-inversion payloads.

Unknown, unsafe, or ambiguous inputs are quarantined/reviewed rather than guessed into a live folder.

Final destinations come from configured ProjectContext asset roots. Installation rechecks the staged hash, copies into a destination-side temporary file, verifies the copied SHA-256, and then commits through an atomic replace/rename.

### Provenance

Installed assets can retain provider/model/version/file identity, source page, author/description/tags/trained words, base model, original filename/size, verified SHA-256, classification result, install timestamp/path, and scan metadata.

Durable metadata uses the existing `.imagegen.json` sidecar convention. A derived local Asset Hub index provides a faster searchable view.

## 21. Help Center and Home Changelog

The WebUI contains a user-facing Help Center backed by public documentation under `help_documentation/`.

Current Help Center behavior includes:

- category/topic navigation;
- search over title, summary, keywords, category, and Markdown body;
- search suggestions after a minimum useful query length;
- related-topic links;
- shared Markdown rendering;
- repository-owned local image/video support; and
- explicit external HTTPS links.

Local help media is served only from the public help root. External resources are not silently embedded as third-party executable content.

The Home Changelog uses the same shared Markdown capability. It can open recent development/release notes from the public repository and fall back to changelog content bundled with the local build when necessary.

## 22. Theme Manager

Theme Manager controls shared WebUI appearance using semantic roles.

Current roles include:

- accent;
- primary surface;
- secondary surface;
- component/card surface;
- component border;
- component accent;
- primary text; and
- secondary/muted text.

### Contrast diagnostics

Text contrast is measured against relevant surfaces. A 4.5:1 ratio is presented as a recommended readability target rather than a hard Save/Apply blocker.

Low-contrast themes are allowed. By default ImageGen asks for confirmation before saving/applying a low-contrast theme; that confirmation preference can be disabled while diagnostics remain visible.

### Local theme packages

Theme Manager can import ZIP-compatible local theme packages.

Import validates the package and adds it to the local theme library without activating it automatically. Packages can then be explicitly activated, disabled, or removed.

Theme packages are treated as visual data. Executable/script payloads, unsafe paths, symbolic links, and unsafe SVG content are rejected. Optional scoped CSS requires the corresponding declared capability.

If an active package later becomes missing or corrupt, ImageGen disables the broken package and falls back to the lower-layer/custom palette so the WebUI can still start.

## 23. Workspace Manager and Responsive Layouts

Workspace Manager controls registered components on supported ImageGen pages.

A component can declare:

- stable identity;
- compatible pages;
- presentation variants;
- size constraints;
- shared capabilities; and
- optional drawer/focused-overlay behavior.

The saved workspace is a portable base layout. Responsive Wide, Standard, Compact, and Narrow modes derive their effective spans/presentations from that base according to actual workspace-container width.

Registered components can expose drag/keyboard resize handles with persistent component spans. Overlay-capable components can opt into reusable **drawer** or **focused** presentation, including click-outside/Escape collapse, resizable drawer width, and an edge restore tab so a collapsed overlay is not stranded.

Asset Details is the first major consumer of the shared overlay behavior, but the capability lives in the reusable workspace/component registry rather than being hard-coded only for Asset Browser.

Responsive preview does not overwrite the user's base layout values merely because a narrower/wider preview is selected.

## 24. Advanced Seed and Numeric Randomization

ImageGen has a shared randomization layer rather than requiring every numeric setting to implement its own unrelated random-number behavior.

### Seed strategy

Normal Seed remains available as a simple fixed or `-1` random value. Advanced Seed adds structured strategy controls directly below the Seed field.

Current strategies include:

- sequential;
- unrestricted random; and
- random within range.

Range syntax accepts forms such as:

```text
[5000,15000]
-1 [5000,15000]
-1, [5000,15000]
```

The optional punctuation after `-1` is normalized rather than treated as a reason to reject an otherwise unambiguous range.

The structured range fields and the Seed expression synchronize in both directions.

### Other numeric parameters

Eligible numeric generation settings can use a structured advanced range containing:

- randomization enabled/disabled;
- random minimum;
- random maximum;
- whole-number-only resolution where appropriate;
- optional hard minimum lock; and
- optional hard maximum lock.

The expression and structured controls synchronize, and field-specific guidance is used where appropriate. CFG Scale, for example, uses a normal CFG-scale example such as `[5.0, 7.5]` instead of Seed-sized values.

Randomization state travels with the generation form/request state and can be included in supported preset/replay workflows.

## 25. In-Program User Configuration Editor

ImageGen can edit `user_config/user-config.yml` from inside the WebUI.

The editor:

- loads the active user configuration;
- validates YAML before saving;
- writes the replacement atomically; and
- keeps a `.bak` copy of the prior file.

This provides a supported recovery/configuration path for settings such as custom asset roots without requiring the user to leave the application for every configuration change.

## 26. Typed Prompt Numeric Semantics

ImageGen's structured prompt pipeline records numeric meaning by grammar context instead of relying on a general integer-versus-decimal guess.

Examples:

```text
(cat:2)         -> attention weight 2.0
[cat:dog:2]     -> absolute schedule step 2
[cat:dog:0.5]   -> fractional schedule boundary 0.5
{cat:2,dog:1}   -> parsed relative group weights 2:1
2{cat|dog}      -> quantity 2
```

Legacy single-colon sequence inference remains available for backward compatibility, but it is explicitly marked as a legacy inference in canonical metadata and Prompt Inspector output.

PromptIR and ConditioningPlan records serialize typed numeric semantics. Invalid values such as zero schedule steps, negative group-local weights, zero quantities, and `nan`/`inf` in known numeric grammar positions produce explicit validation records instead of silently changing numeric meaning.

Group-local numeric interpretation is implemented at the parser/IR level. The existing `{...}` control and experimental `⦃...⦄` cohesive-group candidate both preserve relative local weights, but the final image-conditioning choice between grouping algorithms remains under active qualification.

## 27. Prompt Temporal Composition and Nesting

ImageGen compiles standard prompt schedules and alternates before text encoding instead of leaving their bracket syntax to the legacy scheduler.

Examples:

```text
[cat:dog:2]      -> cat through step 2, dog afterward
[cat:dog:0.5]    -> boundary resolved against the active pass step count
[cat:dog:50%]    -> percentage boundary resolved against the active pass
[cat|dog]        -> deterministic 1-based step cycle
```

Temporal operations compose with structured relations, owner sequences, attention syntax, `AND` weights, and nested scopes without collapsing their local weighting boundaries. Group structure is preserved when temporal syntax appears inside either the existing grouping control or the experimental cohesive-group form, but the final grouping algorithm remains subject to image-level qualification.

Base and Hires passes compile the same semantic prompt against their own active step counts. Standard Classic temporal syntax follows the same shared compiler under Legacy, Parser21, and SuperHybrid unless a parser-specific extension owns the expression.

Conditioning records store temporal segments, resolved boundaries, deterministic alternate policy, expansion metrics, and safe-fallback diagnostics. Encoder-visible texts are deduplicated, and structural temporal punctuation is not sent to the text encoder unless the user explicitly escaped it as literal text.

## 28. Deep Prompt Parent Scope and Recursive Composition

Deep Classic prompt composition preserves grouped sequence owners and embedded structural scopes recursively instead of flattening them at relation boundaries.

The parser distinguishes structures such as:

```text
{lake:sky:clouds}:::X   -> grouped parent syntax owns the attachment
lake:sky:clouds:::X     -> `clouds` is the terminal attachment owner after the sequence
```

Embedded grouped modifiers such as `{pink, blue, yellow} sky` are recognized inside relation children and other nested structural positions, not only at prompt root. Quantity braces such as `2{cat|dog}` and escaped braces remain literal to their owning grammar.

Conditioning-plan diagnostics record parent scope and owner-composition information so the Prompt Inspector and parser test runner can show how grouped/ungrouped structures were interpreted.

The parser's ability to preserve parent/group structure is distinct from the open image-level question of **which grouping algorithm best strengthens semantic cohesion without unwanted leakage or loss of diversity**. Both the existing control behavior and the new cohesive-group candidate remain available for that comparison.

## 29. Model-Family Semantic Conditioning Contracts

ImageGen's structured prompt compiler qualifies semantic composition against explicit conditioning-runtime capabilities instead of assuming that every text encoder returns the same kind of tensor.

ImageGen-owned conditioning runtimes declare semantic-conditioning capabilities including their architecture, composable/required conditioning fields, sequence/temporal support, pooled support, safe-fallback policy, and SD3 T5 policy.

Qualified semantic channels are:

```text
SD1.x / local CLIP      cross_attention
SD2.x / OpenCLIP        cross_attention
SDXL                     cross_attention + pooled
SD3 / SD3.5              cross_attention + pooled
```

SDXL and SD3 use the same hierarchical branch weights for token and pooled conditioning where the active semantic operation is qualified. SD3 with T5 enabled applies each semantic branch to the same T5 text path; with T5 disabled, the zero replacement sequence remains zero after composition.

If a declared runtime cannot preserve a required structured semantic operation, ImageGen performs a punctuation-safe model-family fallback and records the degradation rather than leaking parser control syntax or pretending full support. Structured runtime fields not declared in the capability contract are rejected instead of being silently ignored.

The parser test reports local branch/sequence weights alongside effective-final contributions after nested normalizations. Existing and experimental group contributions can be inspected mathematically, while real-image qualification is used to judge whether those semantics actually improve concept cohesion and attribute attachment.

## 30. Semantic Prompt Replay and Inspection

ImageGen records parser-neutral semantic replay state for shared Classic prompt behavior instead of relying on visible punctuation alone. New semantic records include parser/compiler/canonical contract versions, recorded PromptIR, the conditioning plan, semantic and structure digests, fallback/degradation state, parser seed where relevant, and model-family semantic capability information.

Exact replay prefers recorded semantic state and validates the recompiled semantic digest. Harmless whitespace changes do not invalidate semantic replay, while meaningful changes to structure, weights, owner scope, schedule boundaries, or fallback behavior do. Existing older canonical metadata remains loadable through compatibility migration where supported.

Prompt Inspector exposes a **Semantic Structure** view for base and Hires positive/negative prompts. It can show parsed groups and normalized weights, owner/relation scope, schedules, categorized warnings/fallbacks, semantic digest, encoder-visible text, and effective-final contribution for static nested branches. Temporal branches are explicitly labeled dynamic by step.

Parser capability descriptors distinguish syntax recognition from implemented runtime semantics and advertise replay/digest/inspection only where the active path implements them.

`run.bat parser-test` remains model-free and emits replay/cutover and parser-performance evidence. A separate opt-in real-checkpoint qualification runner under `testing/test_validations/qualification/generation/` can produce timestamped image/request/log evidence, semantic parity/difference comparisons, exact manifest replay, and a visual contact sheet.

Prompt-parser development is still active. The existing grouping behavior, cohesive-group candidate, and `^` / `*` binding operators are intentionally exposed as an A/B qualification set rather than presented as finalized language semantics.

## 31. Experimental Prompt Grouping and Attribute Binding

ImageGen now keeps the existing grouping behavior and a new cohesive-group candidate side by side so they can be compared with the same checkpoint, seed, prompt, and generation settings.

```text
{...}             existing branch-average grouping control
⦃...⦄             experimental shared-context cohesive grouping
modifier^target   experimental target-only binding
modifier*target   experimental target + structural-descendant binding
```

The cohesive-group candidate keeps all group members in the shared encoder context and reinforces one member as the local focus for each weighted branch. The binding experiment keeps modifier and target attached during lowering instead of emitting a naked modifier branch.

`*` inheritance follows structural parent/child scope rather than textual proximity alone. Explicit child `^` or `*` bindings form inheritance barriers; child `*` can begin a new inheriting subtree. Binding inheritance is preserved through both grouping forms when they occur in structural descendants.

Prompt Inspector exposes the experimental group and binding structure. Prompts that do not use the new syntax retain the established semantic/replay contracts.

A dedicated fixed-seed multi-image comparison workflow is intended to evaluate attribute binding, color/concept leakage, composition stability, and diversity before any experimental syntax or algorithm is promoted to a final default. Until that evidence is accepted, these operators should be treated as **experimental**.
