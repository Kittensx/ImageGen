# ImageGen

**Local Stable Diffusion image generation for Windows with SD 1.x, SD 2.x, SDXL, SD3 Medium, and SD3.5 Medium support.**

ImageGen is a local alpha image-generation application built around its own modular Stable Diffusion runtime. The current release supports SD 1.x, qualified SD 2.x, SDXL, SD3 Medium, and SD3.5 Medium text-to-image generation, a browser-based local WebUI, component-based Advanced Models composition, LoRA support on qualified families, neural Hires generation, replayable generation records, persistent generation queues, memory-aware execution, an experimental Asset Browser for Civitai discovery and managed downloads, structured prompt-parser tooling, PNG and lossless WebP output with embedded metadata, and an alpha Canvas Expansion workflow for adapting an image to a larger shape without stretching the protected source.

> [!IMPORTANT]
> ImageGen is still in alpha development. Interfaces, metadata, configuration fields, and experimental workflows may change between releases.
>
> **Current text-to-image architecture support includes SD 1.x, qualified SD 2.x, SDXL, SD3 Medium, and SD3.5 Medium.** Support remains architecture-aware, and individual checkpoints or specialized variants can still have model-specific requirements.
>
> Some implemented capabilities are explicitly marked **Experimental** while active bug testing and qualification continue. The Asset Browser and its checkpoint/LoRA download workflow are the most important current example. See [Experimental Features](features/EXPERIMENTAL.md).

## Start Here

- [Install ImageGen](#installation)
- [Current Support](#current-support)
- [First Generation](#first-generation)
- [Current Features](features/CURRENT.md)
- [What's New](features/NEW.md)
- [Known Limitations](features/LIMITATIONS.md)
- [Experimental Features](features/EXPERIMENTAL.md)
- [Upcoming Features](features/UPCOMING.md)
- [Changelog](changelog/README.md)

---

## Installation

### Requirements

The current public build is designed around:

- **Windows 10 or Windows 11, 64-bit**
- **Python 3.10.20, 64-bit — exactly**
- An NVIDIA CUDA GPU supported by a published ImageGen hardware profile
- A legally obtained, compatible **full SD 1.x, SD 2.x, SDXL, SD3 Medium, or SD3.5 Medium `.safetensors` checkpoint**
- Enough disk space for ImageGen, the Python environment, checkpoints, LoRAs, upscalers, and generated images

### Python 3.10.20 Is Required

Install **64-bit Python 3.10.20** before running the ImageGen installer.

The current installer intentionally checks the complete Python version. Python 3.10.19, 3.10.21, 3.11, 3.12, and other versions are not accepted by this build.

You do **not** need to manually build the ImageGen virtual environment or install the normal Python requirements one package at a time. Once Python 3.10.20 is available, the ImageGen setup process handles the project environment.

### Current Hardware Qualification

ImageGen uses explicit hardware profiles rather than silently guessing at CUDA, PyTorch, xFormers, or MSLK combinations.

The current published Windows profile is qualified for **NVIDIA Blackwell SM120 / compute capability 12.0** hardware. The installer rejects hardware combinations for which a validated profile has not been published.

For the current qualified profile, the PyTorch package supplies the required CUDA 12.8 runtime, so a separately installed CUDA Toolkit is optional. If a compatible local CUDA Toolkit is present, the installer can evaluate it against the validated profile.

### 1. Download or Clone ImageGen

Place ImageGen in a normal writable folder and extract the complete release before running setup.

Avoid running it directly from inside a ZIP archive.

### 2. Run the Installer

From the ImageGen folder, run:

```bat
install.bat
```

The installer handles the environment setup for the supported machine. It will:

1. verify Windows, 64-bit Python, and **Python 3.10.20**;
2. inspect the NVIDIA GPU and driver;
3. match the machine to a validated ImageGen hardware profile;
4. create the project `.venv`;
5. install the qualified PyTorch/CUDA package stack;
6. install the qualified xFormers/MSLK attention stack and remaining ImageGen requirements;
7. create machine-specific runtime configuration; and
8. validate the completed environment before reporting success.

If setup replaces an existing ImageGen `.venv`, the installer is designed to preserve the previous environment as a backup and restore it if installation fails.

### Optional SD3 / SD3.5 Runtime Support Setup

After the main ImageGen environment is installed, SD3 Medium and SD3.5 Medium use an additional runtime-support setup step:

```bat
install_sd3_support.bat
```

This installs the architecture runtime configuration/tokenizer assets and shared text-encoder files used by the SD3-family runtime. It does **not** download a Stable Diffusion checkpoint for you. Main model checkpoints must still be obtained separately and placed in your configured checkpoint library.

The normal WebUI SD3/SD3.5 path is currently qualified around CLIP-L + CLIP-G conditioning. The backend also contains T5/T5XXL component support for Advanced Models work, but normal WebUI T5 selection remains a separate advanced/qualification boundary.

### 3. Add a Supported Stable Diffusion Checkpoint

ImageGen does not include Stable Diffusion checkpoints.

Place a compatible full SD 1.x, SD 2.x, SDXL, SD3 Medium, or SD3.5 Medium `.safetensors` checkpoint in:

```text
models\StableDiffusion\CheckPoints
```

You can then select the checkpoint from the ImageGen WebUI.

> [!NOTE]
> The current generation loader supports qualified full SD 1.x, SD 2.x, SDXL, SD3 Medium, and SD3.5 Medium `.safetensors` checkpoints. A file appearing in an asset browser does not automatically mean that its exact format, architecture, specialized variant, LoRA ecosystem, or secondary workflow is supported.

### 4. Start the WebUI

Run:

```bat
run_webui.bat
```

ImageGen starts a local WebUI on `127.0.0.1`, beginning with port `7860` and moving to the next available port when necessary. The launcher opens the local page after the backend is ready.

The WebUI is intended for **local, single-user use**. It is not a public web server and should not be exposed directly to the internet.

### Other Launch Options

Interactive command-line generation:

```bat
run.bat
```

Run the saved YAML/JSON generation configuration:

```bat
run_config.bat
```

The default generation configuration is:

```text
configs\generation_config.yml
```

---
### Optional: Add or Update Your CivitAI API Key

ImageGen can use a CivitAI API key for provider-backed metadata, previews, Asset Browser discovery, and managed downloads where Civitai authentication is required.

The default private key file is:

```text
secrets\civitai_api_key.txt
```

If the `secrets` folder or file does not exist, create them inside the main ImageGen folder.

Open `civitai_api_key.txt` in a text editor and place your CivitAI API key on a **single line by itself**:

```text
your_civitai_api_key_here
```

Do not add quotation marks, labels, spaces, or additional lines.

To replace an existing key, simply replace the contents of `civitai_api_key.txt` with the new key and save the file. ImageGen reads the key when performing CivitAI metadata requests, so reinstalling ImageGen is not required.

> [!IMPORTANT]
> Your API key is private. Do not share `civitai_api_key.txt`, include it in support logs, or commit it to GitHub.

---

## First Generation

For a basic txt2img run:

1. open the **Generation** workspace;
2. select a supported SD 1.x, qualified SD 2.x, SDXL, SD3 Medium, or SD3.5 Medium checkpoint;
3. enter a positive prompt and, if desired, a negative prompt;
4. choose width, height, steps, CFG, sampler, scheduler, and seed;
5. optionally select LoRAs or enable Hires; and
6. press **Generate**.

Generated txt2img images are stored by default in:

```text
output\txt2image
```

The output location and model directories can be changed through ImageGen configuration.

---

## Current Support

| Capability | Status |
|---|---|
| SD 1.x text-to-image | **Available** |
| SD 2.x text-to-image | **Available** |
| SDXL text-to-image | **Available** |
| SD3 Medium text-to-image | **Available — verified** |
| SD3.5 Medium text-to-image | **Available — verified** |
| Full qualified SD 1.x / SD 2.x / SDXL / SD3 Medium / SD3.5 Medium `.safetensors` checkpoints | **Available** |
| Advanced Models component composition | **Available — alpha / evidence-based** |
| Local WebUI | **Available** |
| Interactive CLI / config-driven generation | **Available** |
| LoRA loading and weighted multi-LoRA generation on qualified paths | **Available** |
| SD3 / SD3.5 LoRA application | **Unverified — not currently claimed as supported** |
| Neural `.pth` Hires / second pass | **Available — alpha** |
| Exact requested output dimensions | **Available** |
| Queue, replay, batch import/export, and variation tools | **Available** |
| Queue persistence across application sessions | **Available** |
| Structured prompt-parser semantics and inspection | **Available — alpha** |
| Brace-based semantic grouping | **Experimental — behavior under revision** |
| PNG output | **Available** |
| Lossless WebP output | **Available** |
| Embedded full-replay / compatibility metadata | **Available** |
| Asset Browser / Civitai discovery and managed downloads | **Experimental — active bug testing** |
| Canvas Expansion / shape adaptation | **Available — alpha / intermediate workflow** |
| General Image-to-Image | **Planned — not yet available** |
| Inpainting | **Planned — not yet available** |
| Textual Inversion / Hypernetworks | **Not active in the current runtime** |
| ControlNet | **Not active in the current runtime** |

For the detailed feature inventory, see [Current Features](features/CURRENT.md). For implemented features that remain under active qualification or bug testing, see [Experimental Features](features/EXPERIMENTAL.md).

## Canvas Expansion Is an Intermediate Workflow

ImageGen now includes a generative **Canvas Expansion** workflow that can extend an existing composition into a larger target canvas.

Typical uses include:

```text
portrait rectangle -> square
square -> landscape
square -> taller portrait
existing image -> larger canvas
fresh txt2img result -> larger canvas before final save
```

The protected source region is preserved while Stable Diffusion generates the newly exposed canvas area. The workflow supports extension prompts, placement controls, feathering, context seeding, and replayable expansion settings.

This feature is intentionally treated as an **intermediate composition and aspect-ratio adaptation stage**, not as a complete general Img2Img replacement. The intended future workflow is:

```text
Txt2Img
-> optional Canvas Expansion
-> expanded intermediate image
-> future Img2Img refinement
```

or:

```text
Existing Image
-> Canvas Expansion
-> expanded intermediate image
-> future Img2Img refinement
```

General Img2Img and inpainting are separate planned modules and are not yet part of the public generation workflow.

---

## LoRA and Hires

### LoRA

Current ImageGen builds can discover and apply LoRAs during generation, including weighted multi-LoRA stacks on qualified runtime paths. The WebUI includes a dedicated LoRA workspace with compatibility information and metadata-oriented browsing.

A user-visible LoRA weight of `1.0` represents the adapter's normal/native effect after the underlying loader performs its own rank/alpha normalization. User weight is applied as a multiplier on that native behavior rather than exposing loader-internal scaling conventions to the user.

The source now contains architecture-aware standard-adapter mapping groundwork for SD3-family transformer and text-encoder targets. **SD3 / SD3.5 LoRA application is not currently claimed as supported**, because a suitable real adapter has not yet been available for end-to-end qualification.

Default LoRA location:

```text
models\StableDiffusion\Lora
```

### Hires

Hires uses a pixel-neural second-pass pipeline rather than the retired latent-only interpolation path. Supported `.pth` neural upscalers are discovered from configured ESRGAN/RealESRGAN roots, enlarged in image space, encoded back through the VAE, and refined through a second denoising pass.

Leaving the Hires prompt empty means **inherit the current base prompt**. The active source also protects that inheritance behavior across cancellation/edit/replay flows so a copied prompt from an earlier generation does not silently become a stale Hires override.

Hires remains an **alpha feature**. Very large targets, unsupported upscaler architectures, and aggressive memory settings can still exceed available VRAM or fall outside the currently qualified path.

## Asset Browser and Managed Downloads — Experimental

ImageGen now includes an Asset Browser / Asset Hub workspace for provider-backed model discovery. Civitai is the first supported provider.

The current workflow can search and browse provider results, retain independent search sessions, inspect model/version/file details, stage downloads, verify transfers, classify assets, automatically finalize safe installs into configured library roots, quarantine ambiguous files, and retain provider/provenance information.

Download controls include bounded concurrency, queue limits, bandwidth limits, provider request spacing, retries, pause/resume/cancel behavior, restart recovery, partial-download cleanup, and transfer history.

> [!WARNING]
> **Experimental — active bug testing.** The Asset Browser and its checkpoint/LoRA download workflow are implemented and available for testing, but search lifecycle, preview retrieval, version/file selection, transfer recovery, classification, automatic installation, and library reconciliation are still receiving corrective testing during alpha development. Keep backups of important model libraries and verify newly installed assets before relying on unattended asset management.

---

## Output, Replay, and Storage

ImageGen's save system is designed around reproducibility without forcing every output sidecar to carry the full diagnostic state of a generation.

Generated images can be saved as:

- **PNG**; or
- **lossless WebP**.

ImageGen can embed either **full replay metadata** or a smaller **compatibility-oriented metadata** record in the image itself. PNG uses embedded text metadata. WebP uses XMP for full replay plus EXIF-compatible parameter text; if the local Pillow/libwebp runtime cannot preserve WebP XMP, ImageGen falls back to compatibility metadata and reports a warning rather than silently pretending full replay metadata was embedded.

The output path can also include:

- a human-readable TXT sidecar when enabled;
- a **compact replay JSON** containing the generation inputs and reproducibility identity needed for replay; and
- a separately pruned diagnostics JSON when diagnostic detail is saved.

Recent output work removes duplicated prompt, schedule, asset, and runtime structures from the compact replay path. This reduces metadata redundancy, keeps replay records easier to inspect, and lowers save/finalization overhead compared with the earlier all-in-one records.

Output writes are staged through temporary files and committed atomically so a failed save is less likely to leave a partially written generation set.

See [What's New](features/NEW.md) for the recent output and runtime changes.

## Public Release vs. Development Source

The downloadable public build is intentionally runtime-focused.

Developer-only test suites, validation packages, benchmarks, internal phase plans, and development audit material are not required for normal ImageGen use and should not be treated as part of the end-user installation.

User-facing documentation should remain small and accessible from the repository root:

```text
README.md
features/
changelog/
```

Internal engineering plans can remain separate from the public user documentation.

---

## Models and Licenses

ImageGen does not redistribute Stable Diffusion checkpoints, LoRAs, neural upscalers, or other third-party model assets.

Users are responsible for obtaining compatible model files legally and following the license and usage terms supplied by each model or asset author.

---

## Reporting Problems

When reporting an alpha problem, useful information includes:

- ImageGen application/build version;
- Windows version;
- GPU model and VRAM;
- Python version;
- selected checkpoint architecture;
- sampler and scheduler;
- memory/attention configuration;
- the replay JSON for the affected generation; and
- relevant diagnostics or console output.

Review diagnostic files before posting them publicly. Prompts, local paths, filenames, and other machine-specific information may be present.

---

## Project Status

ImageGen is an actively developed alpha. The current product focus is a reliable, replayable, memory-aware **SD 1.x, qualified SD 2.x, SDXL, SD3 Medium, and SD3.5 Medium generation environment** while the architecture is extended toward general Img2Img/Inpainting, broader adapter support, stronger component-composition qualification, and additional image-generation workflows.

For planned work, see [Upcoming Features](features/UPCOMING.md). For chronological changes, see the Changelog.
