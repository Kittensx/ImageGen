# Upcoming ImageGen Features

This page summarizes user-relevant work that is planned but **not yet supported** in the current public runtime.

It is based on the current product/program direction while deliberately avoiding implementation details that belong in private/internal engineering documents.

No dates are promised here. Program order can change as technical dependencies, qualification results, and user priorities change.

## Coming Next: Image-to-Image and Inpainting

The next major generation program remains **Image-to-Image and Inpainting**.

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

## Stable Diffusion XL (SDXL)

Stable Diffusion 2.x has moved into the current qualified generation runtime, so the remaining major Stable Diffusion family expansion is **SDXL**.

SDXL generation requires more than model discovery. Planned work still needs to qualify:

- dual tokenizers;
- dual text encoders;
- pooled prompt embeddings;
- added conditioning/time IDs;
- the SDXL-specific UNet call contract;
- model-family-specific validation; and
- end-to-end sampler/scheduler/replay behavior.

The standard LoRA compatibility layer can already identify SDXL UNet/TE1/TE2 adapter targets, but that adapter mapping work should not be mistaken for a completed SDXL base-generation runtime.

## Broader Adapter / LyCORIS Runtime Support

The current adapter inspector can identify formats beyond standard LoRA, but the runtime does not yet execute every detected algorithm.

Future adapter work can expand qualified support for formats such as:

- LoHa;
- LoKr;
- DoRA;
- LoCon-specific algorithm variants that are not safely representable by the current standard path;
- DyLoRA;
- IA3;
- OFT/Diag-OFT;
- BOFT; and
- newer/emerging adapter representations as their runtime contracts stabilize.

Longer-term adapter architecture may also support conversion/analysis between representations rather than treating external adapter formats as ImageGen's internal model.

Until a loader is implemented and qualified, detection should remain informational and unsupported adapters should remain blocked.

## Asset Hub Expansion and Interoperability

The core Asset Hub lifecycle is now present. Future work can build on it rather than creating a second unrelated asset manager.

Planned or expandable areas include:

- additional providers beyond Civitai;
- richer installed-asset browsing and lifecycle actions;
- deeper library-status filtering/search;
- stronger update/version management;
- provider synchronization;
- workflow/recipe analysis;
- ComfyUI-oriented interoperability and translation; and
- optional external workflow bridging where appropriate.

These features should continue using ImageGen's existing asset identities, provenance, staged-download safety, and generation contracts.

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
- Civitai synchronization; and
- replay-readiness analysis using ImageGen's existing replay pipeline.

## Prompt Favorites and Reusable Prompt Cards

Workspace Manager and responsive navigation/layout foundations are now current features. A separate planned prompt-productivity program can build on those foundations.

Planned user-facing areas include:

- prompt favorites;
- prompt cards/thumbnails;
- prompt-only or full-generation loading;
- saved-prompt actions;
- tags; and
- prompt search/filtering.

## Further Workspace and Home-Surface Expansion

The component-based Workspace Manager, responsive layouts, Help Center, and shared Changelog/Markdown capability are now implemented foundations.

Future UI work can extend those foundations with additional registered components, saved workflow profiles, stronger dashboard/home composition, and more reusable cross-workspace surfaces without returning to one monolithic page layout.

## Theme Ecosystem Expansion

Theme Manager now has semantic roles, contrast diagnostics, and local validated packages.

Potential future work includes:

- richer theme-library browsing;
- additional safe visual capabilities;
- more preview surfaces;
- optional sharing/import workflows; and
- expanded package metadata/version management.

Theme packages should remain visual/appearance packages unless a separate extension system explicitly defines a broader security contract.

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

## Additional Generation and Conditioning Systems

Several generation capabilities remain outside the current runtime and should stay in the planned category until their end-to-end application path exists.

Examples include:

- ControlNet;
- active Textual Inversion application;
- Hypernetworks;
- fully qualified external VAE replacement across generation workflows; and
- additional image-conditioned controls built on the future Img2Img foundation.

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
