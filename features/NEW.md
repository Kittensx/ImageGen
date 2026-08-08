# What's New in the Current ImageGen Alpha

This page highlights recent user-visible improvements present in the current source. It is intentionally shorter and more product-focused than the chronological changelog.

## Canvas Expansion and Shape Adaptation

ImageGen now includes a generative Canvas Expansion workflow for adapting an existing composition to a larger target shape without stretching the protected source image.

Current workflows include:

```text
fresh txt2img result
-> optional Expand After Generation
-> expanded intermediate image
```

and:

```text
existing image
-> Expand Existing Image
-> expanded intermediate image
```

Typical uses include portrait-to-square, square-to-landscape, square-to-taller-portrait, and other larger-canvas conversions.

The runtime supports source placement, preserve/feather/generate areas, Edge Pad context initialization, an optional Reflect Pad mode, extension prompts, expansion denoising, source-handoff tracking, and replayable geometry/inference metadata.

This is intentionally an **intermediate image product**. A future general Img2Img module is planned to provide the next refinement stage.

## End-to-End LoRA Generation

LoRA has moved beyond future-facing folders/metadata and is now part of the active generation runtime.

Current work includes:

- discovery and compatibility scanning;
- weighted LoRA application;
- multiple LoRAs per generation;
- prompt/structured LoRA normalization;
- generation/replay provenance;
- dedicated LoRA WebUI tooling; and
- CivitAI-oriented metadata/preview support.

## Pixel-Neural Hires

Hires now uses supported neural `.pth` upscalers in image space rather than the retired latent-only interpolation modes.

The active Hires flow decodes the base image, runs the selected neural upscaler, resolves the exact requested target, re-encodes through the VAE, and performs the refinement pass.

Recent Hires work also adds stronger tiling, memory preflight, stage-owned component residency, replay identity, and intermediate diagnostic support.

## Leaner Replay Files and Output Metadata

ImageGen's replay format has been streamlined around a compact authoritative replay record.

Recent cleanup removes several forms of duplicated data from the normal replay JSON, including redundant prompt assets, scheduler/schedule structures, and execution-only runtime records. Deeper troubleshooting information remains available through a separately pruned diagnostics record.

The result is a clearer division between:

```text
replay data -> what is needed to reproduce the request

diagnostics -> what is needed to investigate execution
```

This reduces save-file redundancy and lowers output finalization/storage overhead.

## Safer Output Commits

Image, TXT, replay JSON, and diagnostic sidecars are staged before the final paths are committed. If a write fails, ImageGen cleans up the transaction rather than intentionally leaving a partially committed batch.

## Faster Generation and Finalization Path

Recent runtime, Hires, memory-lifecycle, and output-path cleanup has reduced unnecessary work in the generation pipeline and in post-generation serialization.

Current alpha testing has shown materially faster normal txt2img and Hires workflows than earlier development builds. These are development observations rather than universal performance guarantees: actual time depends on GPU, model, target size, sampler/scheduler, attention backend, preview behavior, Hires configuration, and memory pressure.

## Improved Prompt Authoring

The current source includes the newer **SuperHybrid** prompt workflow alongside the legacy and experimental parser paths, plus shortcut profiles, prompt presets, validation/preview tools, and regional prompt tooling.

## More Focused Asset Workspaces

The WebUI now separates major asset tasks more clearly, including dedicated checkpoint and LoRA workspaces instead of forcing all asset behavior into one generation form.

## Stronger Replay and Provenance for Advanced Workflows

Hires and Canvas Expansion now carry more of the information required to inspect how a result was produced, including relevant asset identities, exact target geometry, handoff behavior, and advanced-generation settings.

The goal is for newer multi-stage workflows to remain reviewable instead of becoming opaque one-off actions.
