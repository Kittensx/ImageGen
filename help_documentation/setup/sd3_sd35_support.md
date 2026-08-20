---
title: Stable Diffusion 3 and SD3.5 Support
summary: Use SD3 Medium and SD3.5 Medium in the normal IMAGE_GEN WebUI with staged memory, shared CLIP encoders, and advisory model recommendations.
category: Setup
audience: user
status: current
keywords:
- sd3
- sd3.5
- stable diffusion 3
- flow match
- flow euler
- text encoders
- recommended settings
related:
- setup/sdxl_support
featured: true
media: []
external_links: []
---

# Stable Diffusion 3 and SD3.5 Support

IMAGE_GEN supports recognized SD3 Medium and SD3.5 Medium checkpoints in the normal txt2img WebUI path.

The initial normal-generation path uses CLIP-L and CLIP-G conditioning and the SD3 16-channel latent/VAE contract. IMAGE_GEN can use CLIP weights embedded in a checkpoint or resolve the shared standalone CLIP assets when a plain checkpoint does not contain them.

## Recommended starting settings

The qualified reference profile recommends:

```text
Steps:      20
CFG:        5.0
Sampler:    flow_euler
Scheduler:  flow_match_euler
```

These are recommendations, not locks.

The WebUI keeps Steps and CFG editable. Manual Steps and CFG are authoritative by default. Enable **Auto-use recommended steps** or **Auto-use recommended CFG** only when you want the model profile recommendation substituted for generation.

Sampler and scheduler recommendations are advisory only. IMAGE_GEN displays the recommended pair and may warn when your current selection differs, but the model profile does not replace or reject your choice.

A sampler/scheduler can still fail if its implementation is mathematically incompatible with the SD3 Flow Match denoising domain. That is a plugin capability issue, not a model-profile restriction.

## Text encoder sources

IMAGE_GEN uses this default source policy for normal SD3 generation:

```text
embedded CLIP available -> use it
embedded CLIP absent    -> use the shared standalone encoder library
```

The shared encoder locations are:

```text
models\StableDiffusion\TextEncoders\clip\clip_l.safetensors
models\StableDiffusion\TextEncoders\clip\clip_g.safetensors
```

This allows a plain SD3 or SD3.5 checkpoint to reuse one installed CLIP-L and CLIP-G pair instead of requiring duplicated copies inside every checkpoint.

Where the component registry contains an exact identity relationship, IMAGE_GEN uses that registry evidence when resolving explicit external replacements for embedded components.

## Memory behavior

SD3 models are large enough that keeping every learned component on the GPU at once is undesirable on lower-VRAM hardware.

IMAGE_GEN's normal SD3 path uses staged component residency:

```text
text encoders -> encode -> offload
transformer   -> denoise -> offload
VAE           -> decode
```

A model may therefore be fully ready for generation while its learned components are currently staged on CPU. The WebUI treats this state as ready instead of requiring the entire checkpoint to sit on CUDA while idle.

## T5XXL status

Standalone T5XXL has been qualified in the SD3 backend image tests, but SD3-12 does not expose normal WebUI T5 selection yet.

Normal WebUI SD3 generation currently uses the qualified no-T5 conditioning mode. T5 will be surfaced separately so its source selection and significantly higher CPU/memory cost remain explicit.

## Current workflow scope

Normal SD3/SD3.5 txt2img is the supported WebUI path in this phase.

The following SD3 workflows should still be treated as unqualified until their dedicated integration work is completed:

- Hires;
- Img2Img;
- SD3 LoRA application;
- REGION/Canvas/Outpaint combinations;
- other advanced model-composition workflows.

Those limitations do not prevent experimenting with ordinary txt2img settings. Recommended Steps, CFG, sampler, and scheduler values remain user-overridable.
