---
title: Advanced Models and Component Composition
summary: Build generation models from registry-fingerprinted weights, VAEs, and text encoders while keeping normal checkpoint generation available.
category: Setup
subcategory: Models
audience: user
status: current
keywords:
- advanced models
- components
- component registry
- vae
- text encoder
- t5
- checkpoint
- model family
- composition
related:
- setup/sd3_sd35_support
- setup/sdxl_support
featured: true
media: []
external_links: []
---

# Advanced Models and Component Composition

IMAGE_GEN supports two model-selection modes.

## Whole Checkpoint Mode

With **Use Advanced Models** turned off, generation continues to use the normal checkpoint selector. Existing `.safetensors` checkpoint behavior remains the default and the separate VAE override continues to work as before.

## Advanced Models Mode

Enable **Use Advanced Models** when you want IMAGE_GEN to assemble a generation model from component fingerprints already known to the component registry.

When Advanced Models is enabled:

* the normal checkpoint selector is disabled and ignored for generation;
* the normal standalone VAE override is disabled and ignored;
* you choose a model family first;
* IMAGE_GEN shows compatible registry components for the roles used by that family; and
* the selected component fingerprints become the authoritative generation composition.

The source checkpoint that originally contained a selected component is only a tensor donor. It does not make the other components from that checkpoint active.

## Required Components and Auto

A required role can use **Auto** only when the registry contains exactly one unique compatible component fingerprint for that role.

If exactly one unique compatible component exists, Auto selects it.

If multiple unique compatible components exist, IMAGE_GEN requires an explicit choice instead of guessing.

If no compatible component exists, generation cannot proceed until the component registry contains one.

Multiple physical copies of the same fingerprint count as one component choice. The component fingerprint, not the filename, is the identity.

IMAGE_GEN also scans the configured standalone component libraries under `models\StableDiffusion`, including `VAE` and `TextEncoders`. Folder placement tells the registry that a Safetensors file is a candidate VAE or text encoder; it does not establish model-family compatibility. The registry fingerprints the component tensor content and structure, then uses those fingerprints, tensor manifests, exact/structural matches, and the selected family provider's role contract to decide where the component may appear.

Text-encoder filenames are not used to decide whether an encoder is CLIP-L, CLIP-G/OpenCLIP, or T5/T5XXL. For example, a structurally identified CLIP-L encoder can be offered for provider roles that accept CLIP-L even when the file has a custom filename. Conversely, placing an encoder in a `t5` or `clip` subfolder does not make it T5 or CLIP by itself.

## Source Status and Digital Components

Each component choice now shows its current source status:

* **Physical** means a standalone or physically separated exact component source is currently available.
* **Digital** means the component is currently available only from one or more donor checkpoints.
* **Physical + Digital** means both forms are available for the same exact fingerprint.
* **Unavailable** means the registry still knows the fingerprint, but no current usable source remains.

A fingerprint appears only once even when several donor checkpoints or standalone files contain the same exact component. The source count is shown separately from the component identity.

The **Allow digital checkpoint components** option is enabled by default. Turning it off keeps digital-only fingerprints visible but makes them ineligible for Auto and explicit selection. Required-role Auto is recalculated from the fingerprints still eligible under the current source policy.

The source policy does not change the component fingerprint or composition identity. It only controls which known occurrences may currently satisfy a selected fingerprint.

## Registry Locations and Library Refresh

Advanced Models uses the component registry as the source of truth for component identity while keeping physical locations separate from those identities.

The **Scan Model Library** action refreshes the project model tree and any configured additional model-library roots that are currently reachable. Unreachable or disconnected roots are reported instead of causing the whole refresh to fail.

The registry keeps unavailable locations as historical source records. By default the Advanced Models picker filters out component fingerprints that have no currently accessible registered source. Enable **Show unavailable registered components** when you need to inspect those historical entries.

If an existing checkpoint is copied or moved to another reachable location, IMAGE_GEN compares verified whole-file SHA-256 identity rather than filenames. When the new file exactly matches a previously registered checkpoint, the new path becomes an active location and the old unavailable path can be recorded as an exact-SHA relink. The component fingerprints do not change, and their source occurrence list gains the new reachable path.

This means a checkpoint copied from a disconnected drive into the local `models\StableDiffusion\CheckPoints` tree can satisfy the same registered component identities after the next library scan without rebuilding those identities from names or folder assumptions.

## Optional Components

Optional roles are opt-in.

They default to **Off** and are never silently enabled by Auto behavior. To use an optional component, choose its fingerprint explicitly.

For SD3 and SD3.5, T5/T5XXL is an optional role. Leaving T5 Off preserves the qualified CLIP-L + CLIP-G conditioning path.

Families that do not define the `text_encoder_3` T5 role do not offer T5 as a component. The T5 device control is shown only after an SD3/SD3.5 T5 component has been explicitly selected.

## T5 / T5XXL Device Policy

When an SD3 T5/T5XXL component is selected, IMAGE_GEN exposes a separate device choice:

* **CPU** uses the qualified low-VRAM path. T5 is hydrated and encoded on CPU, and the resulting conditioning is transferred for the later generation stages.
* **CUDA** explicitly runs T5 conditioning on the GPU for systems with enough VRAM.
* **Auto** keeps the conservative CPU behavior unless the active memory plan resolves to a high-VRAM profile, in which case CUDA may be used.

Choosing a T5 device does not choose or enable a T5 model. Component identity and execution placement are separate decisions.

## Current Model Families

The first Advanced Models contract covers the learned component roles already represented by the component registry:

* SD 1.x: UNet/model weights, VAE, and text encoder.
* SD 2.x: UNet/model weights, VAE, and text encoder.
* SDXL: UNet/model weights, VAE, text encoder 1, and text encoder 2.
* SD3/SD3.5: transformer/model weights, VAE, CLIP-L, CLIP-G, and optional T5/T5XXL.

Advanced Models currently uses model-family compatibility and registry role evidence as the selection boundary. More detailed compatibility evidence can be added as component combinations are qualified.

## Replay and Batch Behavior

Advanced Models selections are persisted with generation settings, replay data, batch export/import, queue composition, and output manifests. IMAGE_GEN stores the selected fingerprints and resolves them again through the registry instead of treating a remembered filename as the component identity.

## Relationship to Blueprints

Advanced Models does not require a reconstruction blueprint for ordinary generation. Free-form compatible component selection is intentional.

Blueprints are a separate model-composition feature intended for matching a known original composition and for reconstructing/exporting a complete checkpoint when desired.

## Component Registry Contract Foundation

Advanced Models now resolves its family and role definitions through architecture-family provider contracts rather than a separate hardcoded UI/service table. The current production providers cover SD 1.x, SD 2.x, SDXL, and SD3/SD3.5.

The registry keeps exact component identity separate from component location. Existing embedded component snapshots are normalized into one component identity with one or more source records. A source can describe an embedded digital checkpoint component, a standalone shared component, or a future physically extracted component without changing the component fingerprint.

Registry schema upgrades reuse already-stored component fingerprints. They do not require a whole-library strong rehash simply because the application was updated.

Provider support and validation evidence are tracked separately. A family can have an implemented component-composition path without IMAGE_GEN claiming that every free-form combination has been empirically validated. Real-model validation results can strengthen that evidence in later phases without changing the component identity contract.
