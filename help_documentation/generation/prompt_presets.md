---
title: Prompt Presets
summary: Save reusable positive and negative prompt text without replacing the rest of the current generation setup.
category: Generation
audience: user
status: current
keywords:
- prompt preset
- prompt
- negative prompt
- positive prompt
- saved prompt
related:
- generation/saved_generation_concepts
- generation/replay
- generation/generation_profiles
featured: false
media: []
external_links: []
---

# Prompt Presets

Prompt Presets are the lightweight way to save reusable prompt text. They are intentionally separate from Generation Profiles and Replay.

## What a Prompt Preset saves

The Prompt Preset record preserves:

* the preset name;
* positive prompt text;
* negative prompt text;
* prompt-preset notes/tags where supported by the preset record.

Loading a Prompt Preset is meant to change the prompt text without unexpectedly replacing your checkpoint, dimensions, sampler, scheduler, hires setup, or other generation controls.

## When to use Prompt Presets

Use a Prompt Preset when the reusable thing is primarily the wording. For example, the same portrait prompt may be useful with SD1.5, SDXL, or SD3.5 and with several different samplers or resolutions.

## Prompt Presets are not Generation Profiles

A Generation Profile is a broader reusable generation setup and can include prompts plus many other generation controls. Use a Generation Profile when the combination of model/settings/assets is what you want to preserve.

## Prompt Presets are not Replay

Replay is tied to a completed historical image/run and preserves what produced that result. A Prompt Preset is not provenance and is not intended to reproduce the full generation by itself.

## Related topics

* [Saved Generation Concepts: Replay, Generation Profiles, and Prompt Presets](saved_generation_concepts.md)
* [Replay](replay.md)
* [Generation Profiles](generation_profiles.md)
