# Current ImageGen Limitations

This page documents the important boundaries of the current public alpha so unsupported or experimental behavior is not mistaken for a finished feature.

## 1. SD 1.x Is the Only Enabled Generation Family

The current generation capability contract enables **Stable Diffusion 1.x**.

SD 2.x and SDXL can require different text-conditioning, model, and UNet-call contracts. ImageGen intentionally blocks those architectures until their required runtime paths are implemented and qualified.

## 2. Full `.safetensors` Checkpoints Are the Qualified Model Format

The active checkpoint inspector/loader is designed around full monolithic `.safetensors` checkpoints.

Asset discovery code may recognize other filename extensions for cataloging or future work, but `.ckpt`, `.pt`, or `.pth` checkpoint files should not be advertised as supported txt2img model formats simply because they can appear in a browser or folder scan.

## 3. General Img2Img Is Not Yet Implemented

ImageGen does not yet provide the normal general-purpose workflow:

```text
input image
+ prompt
+ denoising strength
-> full image-conditioned redraw/refinement
```

That is the next major planned generation program.

## 4. Inpainting Is Not Yet Implemented

Localized masked editing and the full inpainting-ready workflow are part of the planned Img2Img/Inpainting program rather than the current public runtime.

## 5. Canvas Expansion Is Not a General Img2Img Substitute

Canvas Expansion can accept an existing image, but its role is narrower:

- protect the existing source composition;
- enlarge the canvas;
- generate newly exposed space; and
- produce an intermediate shape-adapted image.

It is not currently intended to fully redraw, restyle, repair, or refine the protected source area.

The future Img2Img module is expected to provide that refinement stage.

## 6. Canvas Expansion Is Still Alpha

Canvas Expansion is useful today, but background continuation can still produce seams, repeated edge content, or imperfect context.

Edge Pad is the preferred general context seed. Reflect Pad is available as an advanced alternative but can mirror recognizable subjects or objects near an edge and encourage duplication.

Expansion quality remains dependent on checkpoint, prompt, target geometry, denoising strength, placement, source content, sampler, and scheduler.

## 7. Hires Is Still Alpha

The neural Hires pipeline is active, but not every `.pth` upscaler is automatically considered compatible.

Current qualification is architecture- and scale-aware. Unsupported or unqualified files can be discovered but should be rejected/deferred rather than executed as if they were known-compatible.

Large targets and aggressive tile settings can still exceed GPU or system memory.

## 8. Public Installer Support Is Currently Narrow

The current installer is Windows-only and requires 64-bit Python 3.10.20 exactly.

The currently published validated hardware profile targets:

- Windows AMD64;
- NVIDIA GPU;
- compute capability **12.0 / SM120**; and
- the qualified PyTorch/CUDA/custom-attention package set shipped for that profile.

The profile manifest currently disallows unvalidated package/hardware combinations.

Linux, macOS, AMD GPUs, Intel GPUs, and other NVIDIA architectures should not be represented as qualified public-release targets until corresponding profiles are tested and published.

## 9. Python Patch Version Is Pinned

This build does not accept “any Python 3.10.”

The installer requires:

```text
Python 3.10.20 x64
```

exactly.

That requirement should remain explicit in release notes and installation instructions until the installer contract changes.

## 10. No Models Are Bundled

Users must provide their own compatible SD 1.x checkpoint, LoRAs, and optional neural upscalers.

ImageGen does not grant rights to third-party model files or override their licenses.

## 11. The WebUI Is Local-Only

The launcher binds the current WebUI to `127.0.0.1`.

ImageGen does not currently provide the security layer expected for a public multi-user service, including authentication, TLS termination, remote tenant isolation, or public job authorization.

Do not expose the local WebUI directly to the public internet.

## 12. Textual Inversion and Hypernetworks Are Not Active Generation Features

The current WebUI contains planned/placeholder asset workspace concepts for these asset types, and project paths exist for future use.

They are not active generation features in the current runtime and should remain listed as unsupported until application code actually loads/applies them.

## 13. ControlNet Is Not Active

ControlNet-related model paths/configuration exist in parts of the project structure, but the current runtime does not contain a qualified ControlNet generation path.

It should not be advertised as supported yet.

## 14. External VAE Replacement Should Not Be Advertised as Qualified Yet

The current source contains VAE catalog/replay/provenance fields and UI selection plumbing, but the inspected canonical txt2img model-loading path still constructs the active VAE from the checkpoint and does not expose a clear end-to-end external-VAE replacement operation in the runner.

Until that path is explicitly verified and qualified, the safest public documentation is to describe checkpoint-embedded VAE behavior and avoid promising external VAE replacement as a supported release feature.

## 15. Alpha Replay Formats Can Still Evolve

ImageGen has significantly improved replay serialization, but the application is still alpha software.

Replay fields, parser records, advanced feature metadata, and compatibility rules can still change between builds. Important presets and generation records should be backed up when moving between development releases.

## 16. Exact Reproduction Has Environment Boundaries

A replay request is designed to reproduce generation state as closely as possible when the same assets and compatible runtime are available.

Bit-for-bit identity is not guaranteed across changes in GPU architecture, PyTorch/CUDA packages, attention backend, sampler implementation, model file, VAE, LoRA content, or other execution dependencies.

## 17. Planned Work Is Not Current Support

The repository contains detailed phase documents for future systems. Those phase plans are useful development guidance, but their presence does not mean the feature is in the public runtime.

The user-facing rule is simple:

- implemented runtime behavior -> [Current Features](CURRENT.md)
- planned or researched behavior -> [Upcoming Features](UPCOMING.md)
- historical implementation changes -> `changelog/`
