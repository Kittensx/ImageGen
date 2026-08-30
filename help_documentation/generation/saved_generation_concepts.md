---
title: "Saved Generation Concepts: Replay, Generation Profiles, and Prompt Presets"
summary: Understand the three ways ImageGen preserves generation information and when to use each one.
category: Generation
audience: user
status: current
keywords:
- replay
- generation profile
- prompt preset
- sidecar
- metadata
- saved settings
related:
- generation/replay
- generation/generation_profiles
- generation/prompt_presets
featured: true
media: []
external_links: []
---

# Saved Generation Concepts: Replay, Generation Profiles, and Prompt Presets

ImageGen has three related but intentionally different ways to preserve generation information. They overlap, but they are not substitutes for one another.

## At a glance

| Concept | Main purpose | Tied to one completed image/run? | Reusable as a preset? | Typical scope |
| --- | --- | --- | --- | --- |
| Replay | Preserve what actually produced an image/run | Yes | Replays the recorded run | Exact or normalized historical generation record |
| Generation Profile | Save a reusable generation setup | No | Yes | Model and generation controls, prompts, parser/settings, assets, and other supported form values |
| Prompt Preset | Save prompt text for reuse | No | Yes | Positive prompt, negative prompt, and prompt-preset metadata |

## Replay

A replay is the historical record of a generation. ImageGen can preserve replay information in an image's metadata and/or in a sidecar file, depending on the configured output settings. Replay exists so ImageGen can answer: **What produced this image?**

Use Replay when provenance, inspection, exact reruns, or reconstruction of a particular completed image matters.

## Generation Profile

A Generation Profile is a user-named, reusable generation configuration. It is designed to answer: **I like this setup; how do I use it again?**

A profile can be saved from the current Generation controls or created from the supported replay fields shown in Image Details. Saving a profile from Image Details does not replace the replay record attached to that image.

Generation Profile JSON files have two identities:

* the **JSON filename**, which is the filesystem/command-line identity;
* the internal **`name`**, which is the user-facing display name shown and sorted in the WebUI.

This distinction lets filesystem tools address a profile unambiguously even when two profile files use the same display name.

## Prompt Preset

A Prompt Preset is deliberately smaller. It preserves reusable prompt text without replacing the rest of the current generation setup. Use it when you want to reuse the words while freely changing models, samplers, dimensions, hires settings, or other generation controls.

## Which one should I use?

Use **Replay** when you care about a specific historical result. Use a **Generation Profile** when you want to reuse a broader setup. Use a **Prompt Preset** when you primarily want to reuse prompt text.

They can work together. A successful image can keep its Replay for provenance, become the source of a Generation Profile for future setups, and contribute prompt text to a Prompt Preset without any of those records replacing the others.

## Related topics

* [Replay](replay.md)
* [Generation Profiles](generation_profiles.md)
* [Prompt Presets](prompt_presets.md)
