# Current ImageGen Limitations

This page documents the important boundaries of the current public alpha so unsupported or experimental behavior is not mistaken for a finished feature.

## 1. SD 1.x and Qualified SD 2.x Are Enabled; SDXL Is Not

The current generation capability contract enables:

- **Stable Diffusion 1.x**; and
- **qualified Stable Diffusion 2.x** checkpoints that satisfy ImageGen's SD 2.x runtime-profile and OpenCLIP conditioning requirements.

SDXL base-model generation is still blocked. The active runtime does not yet provide the complete dual-tokenizer/dual-text-encoder, pooled-conditioning, time-ID, and SDXL-specific UNet-call contract required for qualified SDXL generation.

The LoRA compatibility layer may identify SDXL adapter targets, including TE2, before SDXL base generation itself is available. Those are separate capabilities.

## 2. SD 2.x Support Is Qualification-Bound

SD 2.x support should not be read as “every file labeled SD2 will run.”

The current runtime relies on explicit SD 2.x architecture evidence, local OpenCLIP/runtime assets, prediction-type/profile resolution, and known qualification rules. Ambiguous checkpoints can require an explicit runtime-profile override, and a model that cannot satisfy the active contract should remain blocked rather than guessed into an SD 2.x execution path.

## 3. Full `.safetensors` Checkpoints Are the Qualified Model Format

The active checkpoint inspector/loader is designed around full monolithic `.safetensors` checkpoints.

Asset discovery code may recognize other filename extensions for cataloging or future work, but `.ckpt`, `.pt`, or `.pth` checkpoint files should not be advertised as supported txt2img model formats simply because they can appear in a browser or folder scan.

## 4. Standard LoRA Support Does Not Mean Every Adapter Algorithm Is Supported

The active standard loader covers qualified conventional LoRA representations such as supported Kohya-style, Diffusers/PEFT, and `lora_up` / `lora_down` layouts.

ImageGen can inspect and classify additional formats without executing them.

Current important boundaries include:

- **LoHa:** detectable, not yet runtime-qualified;
- **LoKr:** detectable, not yet runtime-qualified;
- **DoRA-specific magnitude extensions:** detectable/partial, not yet runtime-qualified; and
- unknown/unmapped adapter tensor groups: blocked rather than silently discarded.

A compatible model-family label is not enough. Adapter format and target coverage must also be supported.

## 5. General Img2Img Is Not Yet Implemented

ImageGen does not yet provide the normal general-purpose workflow:

```text
input image
+ prompt
+ denoising strength
-> full image-conditioned redraw/refinement
```

That remains a major planned generation program.

## 6. Inpainting Is Not Yet Implemented

Localized masked editing and the full inpainting-ready workflow are part of the planned Img2Img/Inpainting program rather than the current public runtime.

## 7. Canvas Expansion Is Not a General Img2Img Substitute

Canvas Expansion can accept an existing image, but its role is narrower:

- protect the existing source composition;
- enlarge the canvas;
- generate newly exposed space; and
- produce an intermediate shape-adapted image.

It is not currently intended to fully redraw, restyle, repair, or refine the protected source area.

The future Img2Img module is expected to provide that refinement stage.

## 8. Canvas Expansion Is Still Alpha

Canvas Expansion is useful today, but background continuation can still produce seams, repeated edge content, or imperfect context.

Edge Pad is the preferred general context seed. Reflect Pad is available as an advanced alternative but can mirror recognizable subjects or objects near an edge and encourage duplication.

Expansion quality remains dependent on checkpoint, prompt, target geometry, denoising strength, placement, source content, sampler, and scheduler.

## 9. Hires Is Still Alpha

The neural Hires pipeline is active, but not every `.pth` upscaler is automatically considered compatible.

Current qualification is architecture- and scale-aware. Unsupported or unqualified files can be discovered but should be rejected/deferred rather than executed as if they were known-compatible.

Large targets and aggressive tile settings can still exceed GPU or system memory.

The exact low-resolution base image is now off by default, but users can still explicitly enable that artifact when it is useful to their workflow.

## 10. Asset Hub Is Current but Still Has Deliberate Boundaries

Asset Hub currently uses **Civitai as its first provider**. The provider-neutral architecture is intended to support additional providers later, but they should not be advertised as available until implemented.

Asset Hub deliberately stages and verifies downloads before installation. A completed transfer is not the same as an installed asset.

Automatic classification is intentionally conservative. Unknown or ambiguous files can be quarantined/reviewed instead of being guessed into a model directory.

Installing an asset through Asset Hub also does not automatically make its asset type an active generation capability. For example, installing a textual-inversion or VAE file is separate from proving that the current generation runtime consumes that asset end to end.

## 11. External VAE Replacement Should Not Be Advertised as Fully Qualified Yet

The source contains VAE catalog/provenance fields, Asset Hub VAE classification/install support, replay fields, and UI selection plumbing.

That should not be confused with a fully qualified end-to-end external-VAE replacement path in every generation flow. Until the active model-loading/runner path is explicitly validated for that workflow, the safest public documentation is to describe checkpoint-embedded VAE behavior and treat external replacement as not yet fully qualified.

## 12. Public Installer Support Is Currently Narrow

### NVIDIA Hardware Compatibility and Validation

ImageGen's current public installer targets **64-bit Windows, Python 3.10.20, and NVIDIA CUDA GPUs**.

ImageGen is **not limited to NVIDIA SM120 / Blackwell hardware**.

The current reference and fully validated development environment uses an NVIDIA SM120 GPU. The custom xFormers and MSLK packages distributed with ImageGen include the support required for SM120, but SM120 support should not be interpreted as an SM120-only requirement.

Other NVIDIA GPU architectures may also run ImageGen successfully when they are compatible with the installed PyTorch, CUDA, xFormers, MSLK, and runtime components.

#### Hardware qualification

We distinguish between hardware that is **compatible** and hardware that has been **individually validated**.

**Currently validated:**

- Windows AMD64;
- Python 3.10.20 x64;
- NVIDIA CUDA;
- NVIDIA compute capability 12.0 / SM120; and
- the currently distributed PyTorch, CUDA, xFormers, and MSLK package set.

Other NVIDIA GPUs are not automatically considered unsupported simply because a dedicated validation profile has not yet been published.

ImageGen can also use alternative attention paths, including PyTorch SDPA and eager attention, when a particular optimized attention path is unavailable or incompatible.

Hardware that has not yet been validated should be considered **community-tested / unverified**, rather than unsupported.

Linux, macOS, AMD GPUs, Intel GPUs, and CPU-only generation are not currently qualified public-release targets.

## 13. Python Patch Version Is Pinned

This build does not accept “any Python 3.10.”

The installer requires:

```text
Python 3.10.20 x64
```

exactly.

That requirement should remain explicit in release notes and installation instructions until the installer contract changes.

## 14. No Models Are Bundled

Users must provide their own compatible SD 1.x/qualified SD 2.x checkpoints, LoRAs, and optional neural upscalers.

ImageGen does not grant rights to third-party model files or override their licenses.

## 15. The WebUI Is Local-Only

The launcher binds the current WebUI to `127.0.0.1`.

ImageGen does not currently provide the security layer expected for a public multi-user service, including authentication, TLS termination, remote tenant isolation, or public job authorization.

Do not expose the local WebUI directly to the public internet.

Asset Hub provider credentials are handled by the backend and are not returned to the normal browser settings surface, but that does not turn the WebUI into a remotely hardened hosted service.

## 16. Textual Inversion and Hypernetworks Are Not Active Generation Features

The project contains asset concepts/paths for these types, and Asset Hub can safely classify/install some textual-inversion Safetensors payloads.

That does not mean the current generation runtime loads/applies Textual Inversion or Hypernetworks end to end. They should remain listed as unsupported generation features until the application code actually applies them during conditioning/generation.

## 17. ControlNet Is Not Active

ControlNet-related model paths/configuration exist in parts of the project structure, but the current runtime does not contain a qualified ControlNet generation path.

It should not be advertised as supported yet.

## 18. Theme Packages Are Visual Packages, Not General Extensions

Theme Manager packages are intentionally constrained to appearance data.

Executable/script content is rejected. Importing a theme package does not grant it permission to run Python, JavaScript, executables, DLLs, shell/batch files, or arbitrary active content.

Optional scoped CSS is also capability-gated rather than treated as unrestricted extension code.

## 19. Alpha Replay and Workspace Formats Can Still Evolve

ImageGen has significantly improved replay serialization and now has persistent workspace/theme/configuration systems, but the application remains alpha software.

Replay fields, parser records, advanced randomization metadata, workspace schemas, theme package contracts, and compatibility rules can still change between builds. Important presets, theme packages, workspace exports, and generation records should be backed up when moving between development releases.

## 20. Exact Reproduction Has Environment Boundaries

A replay request is designed to reproduce generation state as closely as possible when the same assets and compatible runtime are available.

Bit-for-bit identity is not guaranteed across changes in GPU architecture, PyTorch/CUDA packages, attention backend, sampler implementation, model file, VAE, LoRA content, SD 2.x runtime profile, or other execution dependencies.

Replay also intentionally does not restore every historical user preference. Output-sidecar choices and similar operational settings remain user-owned preferences rather than generation identity.

## 21. Planned Work Is Not Current Support

The repository contains detailed phase documents for future systems. Those phase plans are useful development guidance, but their presence does not mean the feature is in the public runtime.

The user-facing rule is simple:

- implemented runtime behavior -> [Current Features](CURRENT.md)
- planned or researched behavior -> [Upcoming Features](UPCOMING.md)
- historical implementation changes -> `changelog/`
