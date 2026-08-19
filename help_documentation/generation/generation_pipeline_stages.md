---
title: Generation Pipeline Stages
summary: Understand the major runtime stages IMAGE_GEN executes for a generation request and where stage-specific diagnostics originate.
category: Generation
audience: user
status: current
keywords:
- generation
- stages
- diagnostics
- hires
- vae
- memory
related:
- generation/model_loading_and_runtime_reuse
- generation/replay_and_preflight
- generation/persistent_queue
featured: false
media: []
external_links: []
---

# Generation Pipeline Stages

IMAGE_GEN executes a generation request as an ordered set of runtime stages. The stage split is primarily an internal reliability and diagnostics boundary; it does not change the meaning of an existing generation request.

## Stage order

A normal request progresses through these major stages:

1. **Request preparation** — resolves dimensions, hires planning, seeds, preview policy, and request-scoped memory information.
2. **Conditioning** — encodes positive/negative prompts and any architecture-specific conditioning contract.
3. **Latent preparation** — builds the active schedule and prepares the starting latent tensor. Existing-image expansion also prepares its protected/source regions here.
4. **Base denoising** — runs the selected sampler against the active scheduler/sigma sequence.
5. **Hires transition** — when enabled, performs the existing upscale/VAE/image-conditioned scheduling workflow and second denoising pass.
6. **Decode** — converts final latents to image space, applies required final-size/preservation handling, and records output-quality diagnostics.
7. **Finalization** — assembles runtime, memory, attention, performance, schedule, prompt, and hires metadata into the final generation result.

When hires is disabled, the hires stage is still part of the ordered runtime but performs no second-pass generation work.

## What this means for diagnostics

Errors and diagnostic artifacts can now be associated more directly with the stage that owns the work. For example, a conditioning failure is distinct from a latent/schedule failure, and a final VAE decode failure is distinct from a hires second-pass sampling failure.

The existing detailed subsystem diagnostics remain authoritative. Stage names provide an additional high-level boundary rather than replacing sampler, scheduler, VAE, memory, prompt, or architecture diagnostics.

## Memory and preview behavior

The stage architecture preserves the existing adaptive memory manager. In particular:

- Preview image decoding may be suspended when memory pressure requires it.
- Once automatic preview suspension occurs for a generation job, IMAGE_GEN keeps image decoding suspended for the remainder of that job rather than automatically restoring it mid-job.
- Hires memory admission, cleanup, host staging, VAE placement, and attention controls continue to use their existing policies.
- Pixel-neural hires cleanup remains protected by the outer generation coordinator even when a stage fails or is cancelled.

## Hires generation

Hires remains one complete runtime transition. IMAGE_GEN does not treat the upscale, VAE encode, schedule construction, and second denoising pass as unrelated requests. They remain part of the same generation context and provenance record.

This matters for replay and diagnostics because the final result can record both the base pass and hires pass while preserving the relationship between their schedules, conditioning, model components, and VAE identity.

## Existing-image expansion

Existing-image expansion uses the same stage order with additional work during conditioning, latent preparation, sampling, and decode. Protected source regions and expansion masks remain part of the request-scoped generation context until the final image is assembled.

## Output and replay compatibility

The stage refactor does not introduce a new output format. Existing generation results, manifests, replay metadata, prompt metadata, scheduler/sampler records, memory summaries, and output ownership continue to use their established contracts.

If a generation result differs after a runtime update, treat that as a parity issue to investigate rather than an expected consequence of the stage split.
