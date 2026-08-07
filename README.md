# IMAGE_GEN v0.1.0 Alpha

IMAGE_GEN is a local text-to-image application built around a custom, modular Stable Diffusion pipeline. This first public alpha focuses on reproducible **SD 1.x text-to-image generation**, a responsive local WebUI, command-line generation, configurable samplers and schedulers, detailed output metadata, replay tools, and GPU-memory management.

> **Alpha notice**
>
> This is an early public build intended for testing and feedback. Interfaces, configuration fields, metadata formats, and runtime behavior may change between alpha releases. Keep backups of important presets and generated metadata.
## Project Update — August 6, 2026

ImageGen remains in active alpha development. This update summarizes the latest hires-generation, neural-upscaling, replay, asset-management, and runtime work completed through August 6, 2026.

Where older sections of this README conflict with this update—particularly statements that hires generation is limited to latent interpolation or that neural `.pth` upscalers are not supported—this August 6 update describes the current implementation.

### August 6 Updates

* **Pixel-neural hires generation** — the active hires pipeline has been rebuilt around image-space neural upscaling. ImageGen now decodes the completed base generation to RGB, enlarges it with the selected neural upscaler, re-encodes the enlarged image through the effective VAE, and performs the second denoising/refinement pass from those new latents.

* **Neural `.pth` upscaler support** — added native discovery, inspection, loading, and execution of supported ESRGAN/RealESRGAN-style `.pth` upscaler models through the qualified Spandrel backend.

* **Retired legacy hires interpolation paths** — latent nearest, latent bilinear, latent bicubic, and the former pixel Lanczos hires methods have been removed from the active hires runtime. Enabled hires generation now requires a discovered supported neural upscaler.

* **Upscaler discovery and catalog** — ImageGen recursively scans the configured ESRGAN and RealESRGAN directories, along with explicitly configured additional upscaler roots. Discovered models receive stable SHA-256-backed identities and are exposed through a reusable upscaler catalog.

* **Upscaler classification and qualification** — initial architecture detection covers legacy ESRGAN RRDBNet, RealESRGAN/BasicSR RRDBNet, and RealESRGAN SRVGGNetCompact models. Initial support targets three-channel RGB upscalers with native 2x, 4x, or 8x scaling. Files outside the currently qualified contracts are retained as deferred or unavailable instead of being silently treated as compatible.

* **Safer `.pth` inspection** — upscaler discovery uses tensor-only PyTorch inspection rather than executing arbitrary serialized model objects. Configured-root boundaries are enforced during discovery, including protection against paths or links resolving outside an authorized upscaler directory.

* **Upscaler scan caching and duplicate handling** — inspection results are cached using file metadata, hashes, and loader-version information. Identical model content found in multiple configured locations is collapsed into a single catalog identity while retaining alias-location information.

* **Tiled neural upscaling** — large hires operations can use deterministic tiled inference with configurable tile size, overlap, and tile batch size. Tile blending and exact target-size correction are handled by the upscaling system rather than the sampler.

* **Bounded upscaler OOM recovery** — neural upscaling can perform a controlled smaller-tile retry after a qualifying out-of-memory failure. Recovery is bounded rather than repeatedly retrying an impossible request.

* **Hires memory preflight** — pixel-neural hires jobs now have a dedicated admission/preflight path that estimates important VRAM, system-memory, intermediate-image, and disk requirements before committing to the complete pipeline.

* **Stage-owned memory lifecycle** — the hires pipeline explicitly transitions between base denoising, VAE decode, neural upscale, VAE encode, second-pass denoising, and final decode stages so inactive GPU components can be released or offloaded when appropriate.

* **Host-memory staging controls** — intermediate hires images can be staged through pageable or pinned CPU memory at controlled pipeline boundaries, reducing unnecessary simultaneous GPU residency during high-resolution jobs.

* **Expanded hires WebUI** — hires controls now include neural upscaler selection and refresh, upscaler diagnostics, scale-based or explicit output dimensions, refinement steps, denoising strength, hires sampler/scheduler overrides, CFG overrides, tile controls, tile overlap, tile batch size, and exact-resize filtering.

* **Independent hires prompt routing** — the second pass can inherit the base prompt-processing configuration or use its own prompt parser, shortcut profile, positive prompt, negative prompt, and parser settings.

* **VAE image-to-latent encoding path** — added a dedicated deterministic VAE encoding system for converting the neural-upscaled RGB image back into sampling latents before the hires denoising pass.

* **VAE provenance tracking** — hires generation records the identity and SHA-256 provenance of the effective VAE used during the pixel-to-latent transition. This supports stronger diagnostics and reproducibility for both embedded and externally selected VAE workflows.

* **Hires diagnostic artifacts** — users can optionally preserve the exact neural-upscaled pre-denoise image and a deterministic VAE encode/decode round-trip image for troubleshooting quality changes introduced before the second denoising pass.

* **Low-resolution source preservation** — the original base-generation image can continue to be saved beside the final hires output, allowing direct comparison between the initial generation, neural upscale, VAE transition, and refined result.

* **Intermediate artifact hashing** — saved hires diagnostic and intermediate images can be assigned SHA-256 hashes and explicit artifact roles in generation metadata.

* **Stronger hires manifests** — generation records now preserve the neural upscaler ID and hash, architecture and native scale, tile configuration, exact-resize settings, VAE identity, hires dimensions, second-pass settings, schedule information, memory behavior, and relevant intermediate-stage metadata.

* **Exact pixel-neural replay validation** — replay now verifies that the recorded neural upscaler and VAE still match their original SHA-256 identities. Missing or changed assets are reported as replay errors rather than silently substituting another upscaler or VAE.

* **Improved hires cancellation and progress reporting** — cancellation boundaries and stage reporting have been extended across decode, neural upscaling, VAE encoding, second-pass preparation, and other hires-specific runtime stages.

* **LoRA library metadata improvements** — the LoRA workspace has expanded metadata support including display names, activation text, preferred weights, model-family information, categories, tags, descriptions, notes, source information, compatibility data, and ImageGen sidecar metadata.

* **CivitAI LoRA metadata integration** — installed LoRAs can be matched against CivitAI using their hashes to retrieve available model/version metadata, trained-word activation text, source information, and preview imagery. API credentials are handled by the backend rather than exposed to browser-side JavaScript.

* **LoRA preview management** — LoRA cards can use downloaded metadata previews, locally selected preview files, or compatible recent ImageGen outputs, providing a more visual model-library workflow.

* **Output and replay UI updates** — Recent Outputs, image details, generation-form restoration, and replay processing have been extended to understand the newer hires, VAE, LoRA, upscaler, and intermediate-artifact metadata.

* **CLI and request-schema updates** — command-line, interactive, saved-request, manifest, and runtime contracts now carry the expanded pixel-neural hires configuration so the WebUI and CLI continue to use the same underlying generation system.

* **Dependency update** — Spandrel `0.4.2` is now included as the qualified neural-upscaler model backend.

### Current Hires Status

Hires generation should still be considered an **alpha/experimental feature**, but it is no longer limited to latent interpolation. The active implementation is now based on a real pixel-neural pipeline:

`base denoise → VAE decode → neural .pth upscale → exact target resize → VAE encode → second denoise → final VAE decode`

Supported results still depend on the selected upscaler architecture, available VRAM and system memory, requested output dimensions, tile settings, VAE behavior, sampler/scheduler combination, and the current hardware qualification level.

Unsupported or unqualified upscaler files are intentionally reported rather than automatically executed, and exact replay intentionally fails when the required recorded upscaler or VAE identity no longer matches.

---

## Project Update — August 4, 2026

IMAGE_GEN remains in active alpha development. This dated update supplements the original alpha README below rather than replacing it. Where an older section conflicts with this update—particularly statements that LoRA support is unavailable or that setup must be completed manually—this August 4, 2026 update describes the current behavior.

### Recent additions

* **SuperHybrid Parse Prompter** — added the new SuperHybrid prompt-authoring and parsing workflow.
* **Model and LoRA workspaces** — reorganized the WebUI with dedicated workspaces for browsing, inspecting, selecting, and managing checkpoint models and LoRAs.
* **LoRA support** — added LoRA discovery, compatibility scanning, persistent scan results, visual selection, weighted application, multi-LoRA generation, and generation-manifest recording. Generated PNG metadata now records applied LoRA names, weights, and compatibility hashes for Automatic1111/CivitAI-style resource recognition.
* **Hardware-aware setup** — added an installer that scans NVIDIA GPUs, the installed driver, and locally installed CUDA Toolkits; presents supported environment choices; creates or updates the project `.venv`; installs the matching PyTorch stack; installs the required custom MSLK and xFormers builds; installs the remaining IMAGE_GEN requirements; and validates the completed environment.
* **Profile-driven compatibility** — the installer uses tested hardware profiles instead of guessing at package combinations. The bundled profile covers the currently validated Windows/NVIDIA SM120 environment. Additional GPU architectures can be added as corresponding MSLK/xFormers builds and compatibility profiles are tested and published.

### Setup

For a new installation or a clean environment setup, run:

```bat
install.bat
```

The installer will inspect the machine and guide the user through the compatible GPU, CUDA, and PyTorch environment choices available for that system. After setup completes, launch the local WebUI with:

```bat
run_webui.bat
```

Users must still provide their own legally obtained compatible Stable Diffusion checkpoint and any LoRA files they want to use. Models are not included with IMAGE_GEN.

---
---


## Current Support at a Glance

| Capability | Current status |
|---|---|
| Text-to-image generation | Supported |
| Full SD 1.x `.safetensors` checkpoints | Supported |
| Positive and negative prompts | Supported |
| Exact requested output dimensions | Supported |
| WebUI generation | Supported |
| Interactive CLI and config-file generation | Supported |
| Batch size, batch count, and continuous generation | Supported |
| KES, DPM++ 2M, and Simple Euler samplers | Supported |
| Simple KES and Standard Karras schedulers | Supported |
| Live previews, step progress, timing, and CFG telemetry | Supported |
| Output gallery, metadata inspector, and replay tools | Supported |
| Queue import/export and variation matrices | Supported |
| Runtime memory profiles and low-VRAM controls | Supported |
| Hires / second-pass generation | **Experimental** |
| Neural `.pth` hires upscalers | Not yet supported |
| Image-to-image | Not yet supported |
| Inpainting | Not yet supported |
| LoRA loading or application | Not yet supported |
| SD 2.x generation | Not yet supported |
| SDXL generation | Not yet supported |
| ControlNet, textual inversion, and hypernetworks | Not yet supported |

## Major Working Systems

### 1. Native SD 1.x Text-to-Image Pipeline

IMAGE_GEN uses its own generation pipeline rather than launching another Stable Diffusion WebUI. The current pipeline performs the complete txt2img process:

* Inspects a full monolithic SD 1.x checkpoint before loading it.
* Loads the checkpoint UNet, CLIP text encoder, and VAE.
* Creates positive and negative prompt conditioning.
* Builds the selected denoising schedule.
* Creates deterministic seeded latent noise.
* Runs the selected sampler and scheduler combination.
* Decodes the completed latent through the VAE.
* Saves the final image and generation metadata.

The supported model format for this alpha is a **full SD 1.x `.safetensors` checkpoint** containing the UNet, text encoder, and VAE components.

The model browser may discover `.ckpt`, `.pt`, or `.pth` files, but the current checkpoint inspector and generation loader require `.safetensors`. Files in other formats should not be considered supported merely because they appear in the model list.

### 2. Exact Output Dimensions

Users can request widths and heights that are not divisible by eight.

IMAGE_GEN generates on the next model-compatible latent canvas and then center-crops the decoded image back to the exact requested dimensions. The original requested dimensions remain recorded in the generation metadata.

This allows sizes such as `641 x 959` without forcing the user interface to an eight-pixel increment.

### 3. Deterministic Seeds and Batch Generation

The generation system supports:

* Fixed seeds for reproducible runs.
* Random seed selection.
* Sequential per-image seeds within a batch.
* Batch size and batch count controls.
* Continuous generation until cancelled.
* Seed-aware output filenames and metadata.

A repeated request with the same model, software environment, settings, seed, sampler, and scheduler is designed to reproduce the same result. Exact bit-level identity across different GPUs, PyTorch versions, attention backends, or package versions is not guaranteed.

### 4. Local WebUI

Launch the browser interface with:

```bat
run_webui.bat
```

The WebUI starts on `127.0.0.1`, beginning with port `7860`, and selects the next available port when necessary. The launcher opens the browser after the local health endpoint responds.

The current WebUI includes:

* Checkpoint model selection and activation status.
* Model architecture and loading diagnostics.
* Positive and negative prompt editors.
* Width, height, steps, CFG scale, and seed controls.
* Batch size and batch count controls.
* Sampler and scheduler selection.
* Descriptor-driven advanced sampler and scheduler settings.
* Generation profiles and scheduler presets.
* Prompt presets and parser presets.
* Generate once, generate continuously, and cancel controls.
* A live generation queue with status filters.
* Recent-run status and logs.
* Runtime, attention, memory, and model-residency status.
* Adjustable interface scale and resizable workspace panels.
* Saved layouts for different interface scales.
* Custom accent and surface colors.
* Optional restoration of the previous generation form after restart.

The WebUI is currently a local single-user interface. It does not provide public hosting, user accounts, authentication, or multi-user isolation.

### 5. Resident Model Runtime

The WebUI uses a long-lived model runtime process.

When a checkpoint is selected, IMAGE_GEN can preload it and keep reusable components available between generations according to the active memory policy. The selected model is not intentionally reloaded for every image. Changing the selected checkpoint causes the runtime to transition to the new model.

The runtime reports:

* Selected model path.
* Currently loaded model path.
* CPU and GPU residency state.
* Active generation stage.
* Worker process health.
* Runtime restart count.
* Model loading or activation failures.

Retention is still governed by the chosen memory profile and available VRAM. A low-memory profile may offload components that a high-VRAM profile would retain.

### 6. Model Discovery and Checkpoint Inspection

The model catalog scans the configured checkpoint folder and any additional model roots listed in `user_config/user-config.yml`.

For supported checkpoints, IMAGE_GEN records information such as:

* File path and filename.
* File size.
* SHA-256 hash.
* Detected architecture.
* Prediction type.
* Conditioning width.
* Component coverage.
* Tensor dtypes and selected tensor shapes.
* Previous load attempts and failures.

Current generation support is deliberately restricted to SD 1.x. SD 2.x and SDXL checkpoints may be identified during inspection, but generation is blocked because their required conditioning contracts are not implemented in this alpha.

### 7. Prompt Processing

The prompt system supports positive and negative prompt processing, validation, parser presets, and shortcut profiles.

The available parser paths are:

* **IMAGE_GEN Legacy Prompt Parser** — the established and recommended alpha path.
* **Prompt Parser 21** — experimental.
* **Combined / Auto Dispatch** — experimental.

The WebUI can validate a prompt before generation and show parser-routing or shortcut-expansion information. User shortcut profiles can be created, validated, imported, exported, duplicated, and saved.

Experimental parser modes may change or produce behavior that differs from the legacy parser. Users seeking the most stable alpha experience should use the legacy parser.

### 8. Samplers and Schedulers

The current built-in sampler registry includes:

* **KES Sampler** — configurable Euler or Heun integration with KES guidance and noise controls.
* **DPM++ 2M** — deterministic fixed-step DPM++ 2M-style sampling.
* **Simple Euler** — a minimal fixed-step Euler sampler.

The current built-in scheduler registry includes:

* **Simple Karras Exponential Scheduler** — the feature-rich KES scheduler path.
* **Standard Karras** — a conventional model-bounded Karras schedule intended for fixed-step sampling and comparison.

Sampler and scheduler capabilities are negotiated before generation. When a selected combination does not support a requested feature, the runtime can reject the request or apply a recorded compatibility clamp rather than silently changing the mathematical contract.

Advanced settings are provided by each sampler or scheduler descriptor, allowing the WebUI to build controls from the active plugin schema.

### 9. CFG Lab and Guidance Controls

The KES path includes configurable guidance behavior beyond a flat CFG value.

Current controls include:

* Classic flat guidance.
* Automatic low-CFG shaping.
* Sigma-shaped guidance.
* Step-shaped guidance.
* Smoothstep, cosine, linear, and exponential-decay curves.
* High-sigma guidance boost.
* Late-step guidance taper.
* Optional early-step guidance floor.
* Canonical CFG rescale.
* Legacy KES guidance compatibility controls for replaying older runs.

The live preview panel can display the requested and effective CFG trajectory by denoising step. A seed-locked CFG sweep tool is also available for controlled comparisons.

### 10. Live Preview and Progress Telemetry

During generation, the WebUI can show:

* The latest decoded preview frame.
* Current step and total steps.
* Completion percentage.
* Current step duration.
* Average step duration.
* Estimated remaining time.
* Elapsed generation time.
* Active model, sampler, scheduler, and seed.
* Requested and effective CFG telemetry.
* Preview-frame history for the active job.
* Pause, follow-latest, jump-to-latest, and view-final controls.

The currently qualified image preview mode is the lightweight **A1111-style Fast** preview. Balanced and Accurate preview modes are present in the interface but temporarily disabled.

Preview decoding can add VRAM use and processing overhead. It may be throttled, suspended, disabled during hires, or disabled entirely by the active runtime policy. CFG and progress telemetry can continue even when image decoding is suspended.

### 11. Generation Queue

The WebUI includes a local generation queue with:

* Pending, running, cancelling, completed, cancelled, and failed states.
* Queue filters.
* Active-generation cancellation.
* Clearing of completed queue records.
* Recent-run summaries.
* Per-job console logs and diagnostic links.

Queue and job data are session-oriented. The WebUI can clear cached job requests, logs, diagnostics, and preview frames without deleting completed images from the output folder.

### 12. Output Gallery and Image Details

The Recent Outputs system can scan completed images from the configured output folder and display them in a paged thumbnail gallery.

Current gallery and inspector features include:

* Recent-output thumbnails.
* Lightbox viewing.
* Keyboard navigation.
* Time, model, VAE, sampler, scheduler, size, and generation-mode filters.
* Multi-image selection.
* Output metadata inspection.
* Loading an earlier request back into the generation form.
* Validated replay preflight.
* Queue composition from selected outputs.
* Opening selected requests in the Variation Matrix.

The gallery can inspect PNG, JPEG, JPG, and WebP files. IMAGE_GEN-generated final outputs currently use PNG by default.

### 13. Output Metadata and Reproducibility Records

Each saved generation can include:

* The PNG image.
* A human-readable TXT sidecar.
* A structured JSON generation manifest.
* Embedded PNG metadata.

The PNG contains an Automatic1111/CivitAI-style `parameters` text chunk so compatible sites and tools can detect the prompt and core settings. It also contains a richer IMAGE_GEN manifest for replay and diagnostics.

Recorded data can include:

* Positive and negative prompts.
* Seed and resolved batch seeds.
* Requested dimensions.
* Steps and CFG scale.
* Sampler and scheduler names.
* Sampler and scheduler settings.
* Model path, model name, and hash.
* Runtime attention and memory settings.
* Hires settings when used.
* Timing and diagnostic information.

Output filename templates can use fields such as index, seed, date, time, model, VAE, sampler, scheduler, width, and height.

### 14. Replay, Queue Import, and Queue Export

IMAGE_GEN can reconstruct generation requests from its output metadata and run a preflight before placing them back in the queue.

Replay preflight checks for missing models, VAEs, samplers, schedulers, prompt parsers, and other required request data. Missing assets are reported instead of being silently substituted.

Batch request tools support:

* Native IMAGE_GEN queue JSON.
* JSON Lines.
* CSV.
* Automatic format detection from the filename.
* Filling missing fields from the current form or saved defaults.
* Common model, VAE, sampler, and scheduler remapping.
* Validation before queue submission.
* Exporting selected or composed jobs.

Replay quality depends on the same model and compatible runtime components being available. A request created by a later or earlier alpha may require migration as formats evolve.

### 15. Queue Composer and Variation Matrix

Multiple selected outputs or imported requests can be combined into new jobs through the Queue Composer.

The composer supports overrides for model, VAE, size, steps, CFG, sampler, scheduler, batch count, and seed policy.

The Advanced Variation Matrix can expand one or more validated base requests using:

* Cartesian products.
* Paired or zipped values.
* One-at-a-time sweeps.
* Multiple base requests.
* Sequential, original, or randomized seed policies.
* Exact duplicate removal.
* Configurable job limits.
* Native JSON, JSON Lines, or CSV export.

The matrix previews and validates the expanded jobs before they are queued.

### 16. GPU Attention Backends

The runtime supports the following attention selections:

* **Auto** — attempts a verified xFormers path, then PyTorch SDPA, then eager attention.
* **xFormers** — explicit memory-efficient attention; fails before sampling if activation cannot be verified.
* **PyTorch SDPA** — scaled-dot-product attention through the Diffusers SDPA processor.
* **PyTorch eager** — compatibility fallback.

The included Blackwell requirements file references the custom Windows SM120 xFormers and MSLK builds used during development and validation. These builds are hardware- and environment-specific.

`run_webui.bat` currently starts with `--xformers` by default. On a system without a compatible xFormers installation, launch with SDPA instead:

```bat
run_webui.bat --sdpa
```

### 17. Runtime Memory Profiles

IMAGE_GEN provides reusable process-start memory profiles:

* **Automatic** — lets the runtime select behavior from the environment.
* **Balanced** — retains the UNet when possible while offloading less frequently used components.
* **Low Memory** — uses stronger component offloading, preview suspension, VAE slicing, and pre-hires cleanup.
* **Maximum Memory Savings** — prioritizes fitting the job over speed, including maximum attention slicing, tiled and sliced VAE work, CPU VAE execution, and disabled image previews.

Individual controls include:

* High-VRAM, balanced, low-VRAM, and CPU-fallback policies.
* Configurable VRAM safety margin.
* UNet, VAE, and text-encoder retention policies.
* Attention slicing.
* VAE slicing.
* IMAGE_GEN-owned tiled VAE processing.
* VAE execution on automatic, CUDA, or CPU devices.
* Preview suspension under memory pressure.
* Pre-hires cleanup.
* CUDA allocator configuration.
* Bounded CUDA out-of-memory recovery.

Some runtime options are process-start settings. Changing them in the WebUI can require a WebUI restart. The runtime status panel identifies active settings and can copy the corresponding startup command.

### 18. VRAM and Runtime Telemetry

The WebUI reports physical GPU memory and PyTorch allocator memory separately.

Available measurements include:

* Physical VRAM used, free, and total.
* PyTorch allocated and reserved memory.
* Per-job peak allocated and reserved memory.
* Estimated next-stage memory requirement.
* Active GPU-resident components.
* Offloaded components.
* Active pipeline stage.
* Automatic memory actions.
* Active attention backend and kernel provider.

These values are diagnostic measurements, not promises that every requested resolution will fit. Model size, batch size, preview settings, hires settings, attention backend, and other applications using the GPU all affect available memory.

### 19. Bounded CUDA OOM Recovery

The runtime can perform a limited retry after a CUDA out-of-memory failure.

Depending on the selected recovery profile, it can:

* Release inactive component residency.
* Clear eligible caches.
* Suspend image-preview decoding for the remainder of the job.
* Enable stronger low-VRAM behavior.
* Use VAE slicing or tiled VAE fallback.
* Move supported VAE work to the CPU.

Recovery is intentionally bounded and recorded in diagnostics. It does not loop indefinitely, and it cannot make a single allocation fit when the allocation is larger than the available memory.

### 20. Diagnostics and Failure Bundles

Generation diagnostics can be configured as:

* Off.
* Failures only.
* Every run.
* Deep tensor analysis.

Diagnostics can record:

* Effective request and configuration.
* Model and component information.
* Schedule and sampler settings.
* Stage timings.
* Per-step timing data.
* Memory telemetry.
* Prompt-parser routing and failures.
* Output-quality classification.
* Structured runtime events.
* Optional tensor summaries or statistics.

When enabled, a failure bundle can contain the failed request, effective configuration, component report, schedule report, sampler report, timings, event stream, traceback, and a reproduction command.

Deep tensor diagnostics are intentionally expensive and can materially slow generation. They should be used for troubleshooting rather than normal image production.

## Experimental Hires / Second-Pass Generation

Hires generation is included for testing, but it is **not considered complete or production-ready** in this alpha.

The current experimental hires path can:

* Run a base txt2img pass followed by a second latent denoising pass.
* Scale from the base dimensions or use explicit second-pass dimensions.
* Use latent nearest, bilinear, or bicubic interpolation.
* Set separate second-pass refinement steps.
* Set denoising strength.
* Use fixed executed-step semantics or the legacy proportional-tail policy.
* Inherit or override the base sampler and scheduler.
* Inherit or override CFG scale and CFG rescale.
* Use separate positive and negative prompt overrides.
* Save the exact low-resolution base artifact beside the final output.
* Apply dedicated low-VRAM and pre-hires cleanup policies.

Current hires limitations:

* Quality and compatibility are still being tuned.
* Some sampler and scheduler combinations may not be fully qualified.
* Large second-pass sizes can exceed available VRAM.
* The current upscalers are latent interpolation modes only.
* `.pth` ESRGAN, RealESRGAN, and other neural upscaler models are not yet integrated.
* Hires behavior should not yet be assumed to match Automatic1111 in every case.
* Hires metadata and replay contracts may continue to change during alpha development.

Users should keep hires disabled when evaluating the stability of the core txt2img pipeline.

## Known Limitations

### Image-to-Image Is Not Yet Supported

The current public alpha does not accept an input image as generation conditioning. Image-to-image, inpainting, masking, outpainting, and related workflows are not yet implemented.

### LoRAs Are Not Yet Supported

LoRA folders, metadata fields, and future-facing asset structures exist in parts of the project, but the current generation pipeline does not load or apply LoRA weights. Placing LoRA files in the configured directory will not make them active.

### Neural Hires Upscalers Are Not Yet Supported

The project has configured directories for ESRGAN, RealESRGAN, GFPGAN, and related model assets, but those `.pth`-based processing paths are not part of this alpha. The experimental hires feature currently uses latent interpolation only.

### SD 2.x and SDXL Are Not Yet Supported

The checkpoint inspector can identify SD 2.x and SDXL evidence, but generation is intentionally blocked.

* SD 2.x requires the OpenCLIP tokenizer and text-encoder contract.
* SDXL requires two text encoders, pooled prompt embeddings, added conditioning time IDs, and an SDXL-specific UNet call contract.

### Separate VAE Replacement Is Not Yet a Qualified Release Feature

The WebUI can catalog VAE files and preserve VAE selections in request and replay metadata. The reliable alpha path is still the VAE embedded in the full checkpoint. External VAE replacement should be treated as unqualified until it has completed full end-to-end release validation.

### Experimental Prompt Parsers

Prompt Parser 21 and Combined / Auto Dispatch are experimental. The legacy parser is the recommended stable path for the first alpha.

### Fast Preview Is the Qualified Preview Mode

Balanced and Accurate preview modes are currently disabled. The Fast preview mode is intended for progress monitoring and should not be judged as final-image quality.

### Windows and NVIDIA Are the Primary Alpha Target

The supplied launchers are Windows batch files, and the most heavily validated runtime uses NVIDIA CUDA. The custom Blackwell attention stack specifically targets Windows and NVIDIA SM120 GPUs.

Linux, macOS, AMD, Intel GPU, and CPU-only generation are not qualified public-alpha targets even where some underlying Python code may be portable.

### No Models Are Included

Users must supply a legally obtained, compatible full SD 1.x `.safetensors` checkpoint. Model licenses and usage restrictions are determined by the model author or distributor.

### No Public Server Security Layer

The WebUI binds to localhost by default. It is not designed to be exposed directly to the public internet. There is currently no authentication, TLS configuration, multi-user access control, or remote-job isolation.

### Alpha Compatibility

Presets, queue files, manifests, and replay requests may require changes between alpha releases. Do not assume forward or backward compatibility until a stable format version is announced.

## Basic Setup

### Requirements

The recommended alpha environment is:

* Windows 10 or Windows 11.
* 64-bit Python 3.10.
* An NVIDIA GPU with a compatible PyTorch CUDA build.
* Sufficient system RAM and disk space for the selected checkpoint.

PyTorch is intentionally managed separately from the general Python dependency file because the correct build depends on the installed GPU and driver.

### Create a Virtual Environment

From the IMAGE_GEN folder:

```bat
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install the PyTorch build appropriate for the system, then install the general dependencies:

```bat
.venv\Scripts\python.exe -m pip install -r requirements\requirements.txt
```

For the specifically supported Windows Blackwell SM120 stack used by this project, the pinned hardware-specific dependencies are listed in:

```text
requirements\requirements-blackwell.txt
```

That file does not replace the general requirements file; both dependency groups are required for that environment.

### Add a Checkpoint

Create the checkpoint directory when it does not already exist:

```text
models\StableDiffusion\CheckPoints
```

Place a full SD 1.x `.safetensors` checkpoint in that directory.

Either update the default model path in:

```text
user_config\user-config.yml
```

or launch the WebUI, refresh the model catalog, and select the checkpoint there.

### Start the WebUI

```bat
run_webui.bat
```

When xFormers is unavailable or incompatible:

```bat
run_webui.bat --sdpa
```

### Start the Interactive CLI

```bat
run.bat
```

### Run a Saved YAML or JSON Request

```bat
run_config.bat
```

By default, this uses:

```text
configs\generation_config.yml
```

A different config can be passed as the first argument:

```bat
run_config.bat "C:\path\to\request.yml"
```

### Direct CLI Example

```bat
.venv\Scripts\python.exe -m modules.txt2img.cli run ^
  --model "models\StableDiffusion\CheckPoints\model.safetensors" ^
  --prompt "a detailed landscape painting" ^
  --negative-prompt "blurry, low detail" ^
  --width 640 ^
  --height 960 ^
  --steps 25 ^
  --cfg-scale 7 ^
  --seed 123456789 ^
  --sampler kes ^
  --scheduler simple_kes ^
  --save
```

## Default Output Location

Generated txt2img files are saved under:

```text
output\txt2image
```

The path can be changed in `user_config/user-config.yml` or overridden by a generation request.

## Reporting Alpha Problems

When reporting a generation failure, include as much of the following as possible:

* GPU model and VRAM capacity.
* Windows version.
* Python version.
* PyTorch and CUDA versions.
* IMAGE_GEN release version.
* Attention backend.
* Runtime memory profile.
* Model filename and architecture, without redistributing the model itself.
* The failed request or JSON manifest.
* Console log.
* Failure bundle or diagnostics folder when available.

Do not publish private prompts, local usernames, personal paths, or model files without reviewing the diagnostic contents first.

## Alpha Scope

This release is intended to establish a stable foundation for local SD 1.x txt2img generation, diagnostics, reproducibility, memory management, and future pipeline expansion.

The next major capability areas are expected to include further hires qualification, neural `.pth` upscaler integration, image-to-image, and LoRA support. Those features should be considered future work until they are present and explicitly marked supported in a later release.
