# Current ImageGen Limitations

This page documents the important boundaries of the current public alpha so unsupported or experimental behavior is not mistaken for a finished feature.

## 1. Architecture Support Is Explicit and Qualification-Bound

The current text-to-image runtime supports:

- **Stable Diffusion 1.x**;
- **qualified Stable Diffusion 2.x** checkpoints;
- **SDXL**;
- **SD3 Medium**; and
- **SD3.5 Medium**.

Architecture support should not be read as “every file with a matching family label will run.” ImageGen uses checkpoint structure, architecture evidence, runtime profiles, component contracts, and preflight validation rather than trusting filenames alone.

SD 2.x remains explicitly qualification-bound because OpenCLIP conditioning width, prediction type, runtime assets, and profile resolution must agree with the checkpoint.

SDXL has an active generation path, including its dual text encoders, pooled conditioning, added-conditioning/time-ID contract, and architecture-specific runtime profiles. Specialized SDXL families can still have checkpoint-specific recommendations or unsupported secondary workflows.

SD3 support is currently intentionally narrower than the generic label “all SD3.x.” The normal verified generation profiles are **Stable Diffusion 3 Medium** and **Stable Diffusion 3.5 Medium**. Unknown SD3-family variants should remain blocked or unqualified until a matching runtime profile and empirical validation exist.

### SD3 / SD3.5 workflow boundary

Normal txt2img generation is the currently qualified SD3/SD3.5 WebUI workflow. The following should still be treated as separate/unqualified integration areas unless subsequently documented otherwise:

- SD3 Hires;
- SD3 / SD3.5 LoRA application (architecture-aware adapter groundwork exists, but no suitable real adapter has yet been available for end-to-end qualification);
- general Img2Img;
- REGION / Canvas Expansion / outpaint combinations; and
- broader SD3-family variants beyond the qualified Medium profiles.

The backend has optional T5/T5XXL component support for Advanced Models, but normal WebUI SD3 generation currently uses the qualified CLIP-L + CLIP-G path rather than presenting T5 as a universal default.

### Advanced Models compatibility evidence

Advanced Models can compose model roles from registry-fingerprinted components, but the existence of individually compatible components does not prove that every free-form cross-checkpoint combination has been empirically validated.

Component identity, source availability, structural role compatibility, and runtime validation are separate evidence stages. Unknown or ambiguous combinations should remain explicit rather than being silently treated as equivalent to a known-good whole checkpoint.

## 2. Full `.safetensors` Checkpoints Are the Qualified Whole-Model Format

The active checkpoint inspector/loader is designed around full monolithic `.safetensors` checkpoints for normal whole-checkpoint generation.

Asset discovery code may recognize other filename extensions for cataloging or future work, but `.ckpt`, `.pt`, or `.pth` checkpoint files should not be advertised as supported txt2img whole-model formats simply because they can appear in a browser or folder scan.

## 3. Standard LoRA Support Does Not Mean Every Adapter Algorithm Is Supported

The active standard loader covers qualified conventional LoRA representations such as supported Kohya-style, Diffusers/PEFT, and `lora_up` / `lora_down` layouts.

ImageGen can inspect and classify additional formats without executing them. Potentially unsafe legacy serialized adapter formats can be treated as inspection-restricted rather than being loaded merely to identify them.

Current important boundaries include:

- **LoHa:** detectable, not yet runtime-qualified;
- **LoKr:** detectable, not yet runtime-qualified;
- **DoRA-specific magnitude extensions:** detectable/partial, not yet runtime-qualified;
- unknown/unmapped adapter tensor groups: blocked rather than silently discarded; and
- **SD3 / SD3.5 LoRA:** architecture-aware transformer/text-encoder mapping groundwork exists, but end-to-end support remains unverified because a suitable real test adapter has not yet been available.

A compatible model-family label is not enough. Adapter format, target coverage, loader behavior, and empirical generation evidence must also agree.

For qualified standard paths, the public user weight is intentionally normalized around the adapter's native behavior: **`1.0` means normal/native adapter strength after loader-internal rank/alpha normalization**. Lower or higher user values scale that native effect; users should not have to know the loader's internal normalization constant.

## 4. General Img2Img Is Not Yet Implemented

ImageGen does not yet provide the normal general-purpose workflow:

```text
input image
+ prompt
+ denoising strength
-> full image-conditioned redraw/refinement
```

That remains a major planned generation program.

## 5. Inpainting Is Not Yet Implemented

Localized masked editing and the full inpainting-ready workflow are part of the planned Img2Img/Inpainting program rather than the current public runtime.

## 6. Canvas Expansion Is Not a General Img2Img Substitute

Canvas Expansion can accept an existing image, but its role is narrower:

- protect the existing source composition;
- enlarge the canvas;
- generate newly exposed space; and
- produce an intermediate shape-adapted image.

It is not currently intended to fully redraw, restyle, repair, or refine the protected source area.

The future Img2Img module is expected to provide that refinement stage.

## 7. Canvas Expansion Is Still Alpha

Canvas Expansion is useful today, but background continuation can still produce seams, repeated edge content, or imperfect context.

Edge Pad is the preferred general context seed. Reflect Pad is available as an advanced alternative but can mirror recognizable subjects or objects near an edge and encourage duplication.

Expansion quality remains dependent on checkpoint, prompt, target geometry, denoising strength, placement, source content, sampler, and scheduler.

## 8. Hires Is Still Alpha

The neural Hires pipeline is active, but not every `.pth` upscaler is automatically considered compatible.

Current qualification is architecture- and scale-aware. Unsupported or unqualified files can be discovered but should be rejected/deferred rather than executed as if they were known-compatible.

Large targets and aggressive tile settings can still exceed GPU or system memory.

The exact low-resolution base image is now off by default, but users can still explicitly enable that artifact when it is useful to their workflow.

## 9. Asset Browser / Asset Hub Is Experimental

Asset Hub currently uses **Civitai as its first provider**. The provider-neutral architecture is intended to support additional providers later, but they should not be advertised as available until implemented.

The current Asset Browser is implemented and usable, but its checkpoint/LoRA discovery and managed-download workflow remains under **active bug testing**. Search lifecycle, preview loading, version/file selection, transfer recovery, classification, automatic installation, and library reconciliation can still receive corrective alpha updates.

Downloads are staged and verified before live-library finalization. Transfer completion and installation remain distinct internal states, but the normal managed workflow can automatically continue from a verified transfer into classification and installation when a safe destination is established. Unknown, unsafe, or ambiguous files can be quarantined/reviewed instead of being guessed into a model directory.

Interrupted transfers may retain verified partial bytes for safe HTTP range recovery when identity/range checks permit it. If safe continuation cannot be proven, ImageGen restarts or discards the partial instead of blindly appending incompatible bytes.

Installing an asset through Asset Hub also does not automatically make its asset type an active generation capability. For example, downloading an asset type or placing a path in the managed library does not prove that the current generation runtime consumes that asset end to end.

Users should keep backups of important model libraries and verify newly installed assets while this workflow remains experimental.

## 10. Prompt Parser Brace Grouping Is Still Under Revision

The updated prompt parser implements and records a broad set of structured semantics, but brace grouping is not yet considered complete.

Brace groups are intended to bind contained concepts more closely as a related semantic unit than ordinary comma-separated prompting. Current testing found that the existing group-conditioning behavior does not yet reproduce that intended semantic relationship reliably enough for a finished-support claim.

Group syntax and group-local numeric weights can still be parsed and inspected, but users should treat brace-based grouping as experimental until the revised conditioning behavior is validated against real image generation.

Other prompt-parser operators and behaviors can be qualified independently; this limitation should not be read as a statement that all structured prompt syntax is currently broken.

## 11. External VAE Replacement Should Not Be Advertised as Fully Qualified Yet

The source contains VAE catalog/provenance fields, Asset Hub VAE classification/install support, replay fields, and UI selection plumbing.

That should not be confused with a fully qualified end-to-end external-VAE replacement path in every generation flow. Until the active model-loading/runner path is explicitly validated for that workflow, the safest public documentation is to describe checkpoint-embedded VAE behavior and treat external replacement as not yet fully qualified.

## 12. WebP Replay Metadata Depends on Runtime XMP Support

Lossless WebP output is supported. Full ImageGen replay metadata is embedded through WebP XMP, while compatibility-oriented parameter text is stored through EXIF-compatible metadata.

Some Pillow/libwebp combinations may be unable to persist the full XMP payload. In that case ImageGen preserves compatibility metadata and reports a warning. A WebP that contains compatibility metadata after this fallback should not be assumed to contain the complete ImageGen replay record.

## 13. Public Installer Support Is Currently Narrow

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

## 14. Python Patch Version Is Pinned

This build does not accept “any Python 3.10.”

The installer requires:

```text
Python 3.10.20 x64
```

exactly.

That requirement should remain explicit in release notes and installation instructions until the installer contract changes.

## 15. No Models Are Bundled

Users must provide their own compatible SD 1.x, qualified SD 2.x, SDXL, SD3 Medium, or SD3.5 Medium checkpoints, plus any LoRAs and optional neural upscalers they choose to use.

ImageGen does not grant rights to third-party model files or override their licenses.

## 16. The WebUI Is Local-Only

The launcher binds the current WebUI to `127.0.0.1`.

ImageGen does not currently provide the security layer expected for a public multi-user service, including authentication, TLS termination, remote tenant isolation, or public job authorization.

Do not expose the local WebUI directly to the public internet.

Asset Hub provider credentials are handled by the backend and are not returned to the normal browser settings surface, but that does not turn the WebUI into a remotely hardened hosted service.

## 17. Textual Inversion and Hypernetworks Are Not Active Generation Features

The project contains asset concepts/paths for these types, and Asset Hub can safely classify/install some textual-inversion Safetensors payloads.

That does not mean the current generation runtime loads/applies Textual Inversion or Hypernetworks end to end. They should remain listed as unsupported generation features until the application code actually applies them during conditioning/generation.

## 18. ControlNet Is Not Active

ControlNet-related model paths/configuration exist in parts of the project structure, but the current runtime does not contain a qualified ControlNet generation path.

It should not be advertised as supported yet.

## 19. Theme Packages Are Visual Packages, Not General Extensions

Theme Manager packages are intentionally constrained to appearance data.

Executable/script content is rejected. Importing a theme package does not grant it permission to run Python, JavaScript, executables, DLLs, shell/batch files, or arbitrary active content.

Optional scoped CSS is also capability-gated rather than treated as unrestricted extension code.

## 20. Alpha Replay and Workspace Formats Can Still Evolve

ImageGen has significantly improved replay serialization and now has persistent workspace/theme/configuration systems, but the application remains alpha software.

Replay fields, parser records, advanced randomization metadata, workspace schemas, theme package contracts, and compatibility rules can still change between builds. Important presets, theme packages, workspace exports, and generation records should be backed up when moving between development releases.

## 21. Exact Reproduction Has Environment Boundaries

A replay request is designed to reproduce generation state as closely as possible when the same assets and compatible runtime are available.

Bit-for-bit identity is not guaranteed across changes in GPU architecture, PyTorch/CUDA packages, attention backend, sampler implementation, model file, VAE, LoRA content, architecture runtime profile (including SD 2.x, SDXL, or SD3-family profiles), or other execution dependencies.

Replay also intentionally does not restore every historical user preference. Output-sidecar choices and similar operational settings remain user-owned preferences rather than generation identity.

## 22. Planned Work Is Not Current Support

The repository contains detailed phase documents for future systems. Those phase plans are useful development guidance, but their presence does not mean the feature is in the public runtime.

The user-facing rule is simple:

- implemented runtime behavior -> [Current Features](CURRENT.md)
- planned or researched behavior -> [Upcoming Features](UPCOMING.md)
- historical implementation changes -> `changelog/`
