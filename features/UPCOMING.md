# Upcoming ImageGen Features

This page summarizes user-relevant work that is planned but **not yet supported** in the current public runtime.

It is based on the current phase-program direction while deliberately avoiding implementation details that belong in internal engineering documents.

No dates are promised here. Program order can change after the next major generation milestone when priorities or technical dependencies change.

## Coming Next: Image-to-Image and Inpainting

The current phase-program index identifies **Image-to-Image and Inpainting** as the next major program after the completed neural-Hires continuation work.

The planned progression begins with a normal external-image conditioning foundation and then expands toward:

- standard Img2Img generation;
- source-image analysis;
- selective person/object preservation;
- scene enhancement;
- localized editing;
- inpainting-ready masks and workflows;
- architecture-aware conditioning;
- WebUI controls for image analysis/preservation; and
- replay, queue, metadata, and diagnostics integration.

This program is also the intended refinement destination for the current Canvas Expansion workflow:

```text
Txt2Img or Existing Image
-> Canvas Expansion when needed
-> Img2Img refinement
```

## Near-Future Model-Family Expansion

ImageGen is currently SD 1.x only.

Near-future model-family work is expected to add support for:

- **Stable Diffusion 2.x**; and
- **Stable Diffusion XL (SDXL)**.

These are not simple model-browser toggles. Each family requires its own conditioning/model contract to be implemented and validated before generation is enabled.

Until that work lands, SD 2.x and SDXL should remain clearly marked as planned.

## Asset Hub and Interoperability

A planned asset-management program expands the current model/LoRA tooling into a broader asset hub.

Planned areas include:

- asset discovery and installed-asset management;
- secure downloads and install routing;
- stronger provenance;
- CivitAI integration;
- workflow/recipe analysis;
- ComfyUI-oriented interoperability and translation; and
- optional external workflow bridging where appropriate.

This work is intended to build on ImageGen's existing asset identities and generation contracts rather than creating a second unrelated runtime.

## Durable Gallery and Image Library

The current Recent Outputs browser is useful for recent generations, but a larger Gallery program is planned for durable libraries.

Planned capabilities include:

- multiple registered image roots;
- persistent metadata indexing;
- scalable thumbnail browsing;
- favorites;
- user tags;
- custom galleries;
- notes;
- filtering/search;
- source-aware metadata ingestion;
- CivitAI synchronization; and
- replay-readiness analysis using ImageGen's existing replay pipeline.

## Prompt Favorites and Shared Workspace Navigation

A separate planned UI/workflow program covers reusable prompt-card storage and stronger navigation between ImageGen workspaces.

Planned user-facing areas include:

- prompt favorites;
- prompt cards/thumbnails;
- prompt-only or full-generation loading;
- saved-prompt actions;
- tags and search; and
- shared/responsive workspace navigation.

## Persistent CLI / `run.bat` Quality-of-Life Work

The command-line launcher is planned to evolve from a mostly one-shot/interactive entry point into a more persistent local control surface.

Planned improvements include:

- a persistent console menu;
- editable runner state;
- model/sampler/scheduler menus;
- VAE and LoRA configuration surfaces;
- Hires controls;
- replay history/favorites;
- CFG profiles; and
- stronger interoperability with the normal ImageGen configuration format.

## Additional UI Reorganization

Internal phase planning also contains further workspace/UI organization work, including a stronger home/dashboard concept, shared navigation, changelog/community surfaces, workflow profiles, and more modular workspace behavior.

These are planned product improvements rather than requirements for the current generation runtime.

## Longer-Range Research and Feature Programs

Several additional programs exist as longer-range research or optional feature work. They should not be read as near-term promises.

### Model-Native Prompt Reconstruction

Research into interpreting generated/source images through model-native cues, reconstructing prompt guidance, scoring reconstruction quality, and using that information in future generation/replay workflows.

### Multi-Image Semantic Composition

Research into generation contracts that accept multiple image sources and combine their semantic influence in controlled, replayable ways.

### Model Weight Paging and Block-Streamed Inference

Runtime research into CPU/file-backed weight staging, block-level paging, pinned-memory transfer, and bounded execution caches for lower-memory inference.

### Continuous-Resolution Research

Isolated research into more flexible latent/exact-resolution approaches. This remains a research track and should not be represented as a production feature until an evidence-based integration gate is passed.

## Roadmap Documentation Policy

The root README should only name a small number of high-value upcoming items.

This file can carry the broader user-facing roadmap, while detailed implementation sequences remain in internal phase documents. A planned item should move from this page to [Current Features](CURRENT.md) only after the release runtime actually implements it.
