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
- configurable parser settings; and
- prompt symbol/authoring helpers in the WebUI.

Some parser paths remain more experimental than the legacy path and may evolve during alpha development.

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

LoRA is active in the current generation runtime.

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

The compatibility layer now separates four questions that were previously easy to conflate:

1. **Family** — SD 1.x, SD 2.x, SDXL, or unknown for the currently qualified standard LoRA path. SD3/SD3.5 LoRA application remains a separate unqualified workflow.
2. **Format** — the adapter representation detected from metadata/tensor keys.
3. **Targets** — components such as UNet, text encoder 1, text encoder 2, linear layers, or convolutional layers.
4. **Runtime support** — whether the current build has a qualified loader for that exact combination.

### Standard loader

The registered standard Diffusers/PEFT path handles conventional representations classified as:

```text
standard_kohya_lora
standard_diffusers_peft_lora
standard_lora_up_down
```

Conventional linear and supported convolutional LoRA targets can use the standard path when converted keys map cleanly to qualified model components.

Current family target awareness includes:

- **SD 1.x:** UNet and text encoder 1;
- **SD 2.x:** UNet and text encoder 1, without applying SD 1-specific text-encoder shape assumptions; and
- **SDXL:** UNet, TE1, and TE2 target identification/mapping for adapter compatibility work.

SDXL base generation is now enabled, but adapter compatibility remains a separate question: an SDXL checkpoint being runnable does not automatically make every SDXL-targeted adapter format/runtime combination qualified.

### Unsupported or partial adapter formats

ImageGen can identify LyCORIS-style formats such as **LoHa** and **LoKr**, but the current standard loader does not execute those algorithms. They are reported as unsupported rather than silently passed through a generic fallback.

DoRA magnitude data is also inventoried separately and is currently treated as requiring a dedicated qualified runtime path.

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
- decoded preview frames when enabled; and
- runtime stage information.

Preview decoding can be throttled or suspended by memory policy without disabling the underlying generation progress/CFG telemetry.

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
- per-item pause and resume;
- pause-after-current-image boundaries for active multi-image work;
- multiple held/paused jobs;
- skipping paused queue items while other work remains schedulable;
- moving queued items higher or lower without displacing the active generation;
- persistence of recoverable queued work across application sessions;
- restoration of explicit queue order, individually paused queued jobs, and whole-queue hold state; and
- safe recovery of an interrupted active job at the front of a held queue so reopening ImageGen does not unexpectedly restart GPU work;
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
- filters;
- multi-image selection;
- loading prior generation settings;
- replay preflight; and
- handoff into queue/variation workflows.

A larger durable Gallery/asset-library system is separately planned and should not be confused with the current Recent Outputs browser.

## 15. Compact Replay and Output Storage

The output pipeline separates replay-essential data from deeper diagnostics.

The normal structured sidecar uses a compact replay serialization profile. It is designed to keep the generation inputs and reproducibility identity required for replay while pruning duplicated or execution-only structures.

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

## 20. Asset Hub

Asset Hub is the current provider-neutral system for discovering, downloading, verifying, classifying, installing, and tracking model-related assets. Civitai is the first supported provider.

### Provider discovery and authentication

Provider credentials are backend-owned. The browser can learn whether a credential is configured, but stored credentials are not returned after submission.

Civitai authentication can use supported session/environment/OS credential sources. Credentials are attached only to the expected provider host and are not forwarded to unexpected redirect destinations.

### Secure download staging

Asset Hub downloads first enter a temporary staging area rather than writing directly into live checkpoint/LoRA/VAE/upscaler folders.

Current download behavior includes:

- queueing;
- bounded concurrent transfers;
- cancellation;
- safe resume when the remote transfer supports it;
- restart recovery;
- file-size verification; and
- SHA-256 verification.

A verified staged file is not considered installed or `In Library` until the install phase succeeds.

### Classification and installation

Asset Hub creates an install plan before changing a live asset directory.

ImageGen reuses its own technical inspectors for classification. Current paths include:

- LoRA Safetensors inspection;
- checkpoint Safetensors inspection;
- ESRGAN/RealESRGAN-oriented `.pth` upscaler inspection;
- recognizable Safetensors VAE layouts; and
- safely readable Safetensors textual-inversion payloads.

Unknown, unsafe, or ambiguous inputs are quarantined/reviewed rather than guessed into a live folder.

Final destinations come from configured ProjectContext asset roots. Installation rechecks the staged hash, copies into a destination-side temporary file, verifies the copied SHA-256, and then commits through an atomic replace/rename.

### Provenance

Installed assets can retain:

- provider/model/version/file IDs;
- source page;
- author/description/tags/trained words;
- base model;
- original filename and size;
- verified SHA-256;
- classification result;
- install timestamp;
- local install path; and
- scan metadata.

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
- size constraints; and
- shared capabilities.

The saved workspace is a portable base layout. Responsive Wide, Standard, Compact, and Narrow modes derive their effective spans/presentations from that base according to actual workspace-container width.

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
