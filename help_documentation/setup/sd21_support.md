---
title: Stable Diffusion 2.1 Runtime Support
summary: IMAGE_GEN installs the lightweight tokenizer and configuration files needed to run user-supplied Stable Diffusion 2.x checkpoints.
category: Setup
audience: user
status: current
keywords:
- sd2
- sd2.1
- stable diffusion 2.1
- installer
- hugging face
- runtime assets
related:
- asset_hub/installation/classification_and_installation
featured: true
media: []
external_links:
- label: Stable Diffusion 2.1 Base on Hugging Face
  href: https://huggingface.co/sd-research/stable-diffusion-2-1-base
---

# Stable Diffusion 2.1 Runtime Support

IMAGE_GEN installs the lightweight Stable Diffusion 2.1 runtime support files during the normal program setup. The files are downloaded directly from `sd-research/stable-diffusion-2-1-base` after the normal Python requirements are installed, so `huggingface_hub` is already available.

## What IMAGE_GEN downloads

The standalone SD2.1 support script downloads only the runtime files needed by IMAGE_GEN's SD2.x pipeline:

```text
runtime_assets/stable_diffusion/sd2_1_base/
├── model_index.json
├── feature_extractor/
│   └── preprocessor_config.json
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   └── config.json
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet/
│   └── config.json
└── vae/
    └── config.json
```

These files are approximately 1.6 MB in total in the current SD2.1 Base repository.

## What IMAGE_GEN does not download

The SD2.1 support installer deliberately refuses model-weight files. It does not download:

- Stable Diffusion checkpoints;
- `.safetensors` model weights;
- `.ckpt`, `.bin`, `.pt`, `.pth`, `.onnx`, or `.gguf` weights;
- `model_tooling/reference_components` weights.

Users manage their own Stable Diffusion 2.x checkpoints through the normal IMAGE_GEN model folders and asset workflows.

## During normal IMAGE_GEN setup

The main IMAGE_GEN installer performs the sequence below:

1. Create the Python environment.
2. Install the normal IMAGE_GEN Python requirements.
3. Launch the separate SD2.1 runtime-support installer.
4. Wait for the support installer to finish.
5. Resume the remaining IMAGE_GEN setup stages.

There is no SD2.1 support prompt because these files are lightweight runtime dependencies rather than optional model assets.

The global `--no-download` setup option still skips network downloads, including SD2.1 runtime support.

## Installing or repairing support later

The same standalone installer can be run manually:

```bat
install_sd21_support.bat
```

The installer is idempotent. Existing non-empty support files are kept, and only missing files are retrieved.

IMAGE_GEN also checks the support bundle when an SD2.x checkpoint is installed through Asset Hub or activated later. If runtime support is missing, IMAGE_GEN launches the same standalone support script. Model activation remains blocked until the support files are ready.

## GPU qualification

Downloading the lightweight support files is hardware-independent and is not gated by GPU VRAM.

IMAGE_GEN separately qualifies the selected GPU when a user attempts to activate an SD2.x checkpoint. Under the current qualification policy, the selected NVIDIA GPU must report **more than 13 GiB of VRAM** for SD2.x model execution. This execution gate does not affect installation of the runtime support files.
