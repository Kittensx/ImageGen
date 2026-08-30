---
title: Replay
summary: Learn how ImageGen preserves the historical generation record for completed images and runs.
category: Generation
audience: user
status: current
keywords:
- replay
- sidecar
- embedded metadata
- provenance
- exact replay
- image details
related:
- generation/saved_generation_concepts
- generation/generation_profiles
- generation/prompt_presets
- generation/replay_and_preflight
featured: false
media: []
external_links: []
---

# Replay

Replay is ImageGen's historical generation record. It exists to preserve enough information about a completed image or run to inspect what happened and, where supported, reconstruct or rerun it later.

## Where replay information can live

Depending on output settings, replay-capable metadata can be preserved:

* in metadata embedded in the generated image;
* in a sidecar record written with the output;
* in ImageGen's output/job records used by Image Details and replay tools.

An embedded record travels with the image. A sidecar can be easier to inspect and can hold structured information without relying on the image container's metadata fields. Keeping one or both is about provenance and replay fidelity, not about creating a reusable preset.

## What Replay is for

Replay answers questions such as:

* Which model and settings produced this image?
* Which seed, prompts, sampler, scheduler, dimensions, and other request values were used?
* Which parser/replay semantics or resolved request state were recorded?
* Can this completed run be queued again through ImageGen's replay/preflight workflow?

The exact recorded fields depend on the generation path and the metadata available for that output.

## Replay is not a Generation Profile

A Replay belongs to a particular historical result. A Generation Profile is a reusable user preset. You can use **Save as Generation Profile** in Image Details to turn the supported settings from a replay into a reusable profile, but that does not modify or replace the replay record.

Some replay information may also be more exact than a profile should be. For example, a historical replay can preserve a specific resolved choice, while a reusable profile may intentionally keep a setting that generates a new choice on the next run.

## Replay is not a Prompt Preset

A Prompt Preset is intentionally focused on reusable prompt text. Loading a Prompt Preset should not silently replace the full model/runtime/generation configuration recorded by Replay.

## Related topics

* [Saved Generation Concepts: Replay, Generation Profiles, and Prompt Presets](saved_generation_concepts.md)
* [Generation Profiles](generation_profiles.md)
* [Prompt Presets](prompt_presets.md)
* [Replay and Preflight Validation](replay_and_preflight.md)
