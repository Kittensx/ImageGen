# Current ImageGen Features

This page describes capabilities present in the current ImageGen source/runtime. It is a feature inventory, not a chronological changelog.

ImageGen is still an alpha application, so “available” means implemented in the current alpha rather than frozen or guaranteed API compatibility.

## 1. Native SD 1.x Text-to-Image Generation

ImageGen runs its own modular Stable Diffusion generation pipeline rather than launching another image-generation WebUI as its backend.

The current qualified model family is **Stable Diffusion 1.x** using a full monolithic `.safetensors` checkpoint containing the components required by the SD 1.x pipeline.

The runtime handles:

- checkpoint inspection and architecture validation;
- UNet, CLIP text encoder, and VAE loading;
- positive and negative conditioning;
- seeded latent creation;
- sampler/scheduler execution;
- CFG/guidance handling;
- VAE decode;
- output saving and metadata generation; and
- replay/provenance capture.

SD 2.x and SDXL checkpoints may be identifiable by inspection, but generation is intentionally blocked until their different conditioning contracts are implemented.

## 2. Exact Requested Dimensions

Users can request non-standard widths and heights instead of being restricted to a simple eight-pixel UI increment.

ImageGen resolves the model-compatible internal geometry required for generation and preserves the requested final output dimensions in the generation workflow and replay data.

This exact-size behavior also feeds newer Hires and Canvas Expansion workflows.

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

The current tooling includes region geometry, regional prompts, weights, timing/start-stop behavior, blending controls, and integration with the ImageGen generation workflow.

REGION remains an advanced feature and should be treated separately from ordinary global prompting and from Canvas Expansion.

## 6. LoRA Support

LoRA is active in the current generation runtime.

Current LoRA capabilities include:

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

The WebUI also contains CivitAI-oriented LoRA metadata support, including hash-based matching and preview/metadata workflows where the required network/API configuration is available.

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
- upscaler and VAE provenance in generation records; and
- replay validation against recorded asset identity.

Hires is available, but it is still an **alpha feature** and is sensitive to upscaler compatibility, target size, memory pressure, VAE behavior, and sampler/scheduler qualification.

## 8. Canvas Expansion / Shape Adaptation

ImageGen now has an alpha generative Canvas Expansion workflow.

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
- memory/runtime information; and
- configurable workspace/layout behavior.

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

## 11. CFG Lab and Guidance Controls

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

## 12. Generation Queue and Batch Workflows

The WebUI includes a local job queue and batch-oriented tools.

Current functionality includes:

- queued/running/completed/failed/cancelled states;
- active-job cancellation;
- queue filtering;
- recent run information;
- batch size and batch count;
- continuous generation;
- queue import/export;
- JSON, JSON Lines, and CSV request workflows;
- request remapping/validation; and
- queue composition from prior outputs.

## 13. Replay and Variation Matrix

ImageGen records generation state so completed outputs can be inspected and reused.

Current replay tooling can:

- reconstruct a generation request from ImageGen output metadata;
- preflight required assets/settings before queueing;
- report missing or changed assets instead of silently substituting them;
- restore prior settings to the generation form; and
- send validated requests back to the normal generation queue.

The Variation Matrix can expand one or more requests through controlled combinations of settings and seed policies before submission.

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

The output pipeline now separates replay-essential data from deeper diagnostics.

The normal structured sidecar uses a compact replay serialization profile. It is designed to keep the generation inputs and reproducibility identity required for replay while pruning duplicated or execution-only structures.

Current storage cleanup includes behavior such as:

- removing empty records;
- avoiding duplicate prompt-asset copies;
- pruning non-generation scheduler fields;
- avoiding redundant schedule representations when the authoritative representation is already stored;
- keeping compact runtime fingerprints instead of the entire conformance snapshot; and
- pruning duplicated diagnostic/runtime structures in the deeper diagnostics sidecar.

This makes replay files smaller and easier to inspect while preserving a separate path for deeper troubleshooting data.

The save path also uses temporary files and final atomic replacement so an interrupted save does not intentionally expose half-written output sets.

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

The public installer is profile-driven because the custom attention stack is hardware and environment specific.

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
