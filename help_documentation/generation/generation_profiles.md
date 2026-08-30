---
title: Generation Profiles
summary: Save and reuse broader ImageGen generation configurations without replacing historical replay metadata.
category: Generation
audience: user
status: current
keywords:
- generation profile
- profile
- preset
- saved settings
- model settings
- image details
related:
- generation/saved_generation_concepts
- generation/replay
- generation/prompt_presets
featured: false
media: []
external_links: []
---

# Generation Profiles

A Generation Profile is a reusable, user-named generation configuration. It is intended for setups you want to apply again, not as the authoritative historical record of one completed image.

## What a Generation Profile saves

Generation Profiles use the supported Generation form values available to the WebUI. Depending on the active features, this can include items such as:

* model/checkpoint selection;
* positive and negative prompts;
* dimensions, steps, CFG, seeds, and batch controls;
* sampler and scheduler selections and supported advanced values;
* prompt parser/profile settings;
* hires settings;
* supported LoRA, textual-inversion, and related generation assets;
* other supported generation controls captured by the current form/replay contract.

The exact profile schema can grow as ImageGen's generation controls grow.

## Saving a profile

You can save the current Generation controls from the **Generation Profile** control. Image Details also provides **Save as Generation Profile**, which builds a reusable profile from that output's supported replay fields.

Saving from Image Details means **save the settings associated with this image as a reusable preset**. It does not replace the image's embedded replay metadata or sidecar.

## Filename identity and display name

Every Generation Profile JSON file has two intentionally separate identities:

* **Filename** — the filesystem and command-line identity. For example, `woman_bikini_sd3.5.json` is addressed as `woman_bikini_sd3.5` by command-line tools such as the HMR-04 probe.
* **Internal `name`** — the display name shown in the WebUI. The WebUI loads all profile JSON files and sorts them by this internal name.

If you rename only the JSON file in Explorer, the WebUI display name does not automatically change because the internal `name` is unchanged. Command-line tools that explicitly accept a Generation Profile filename use the renamed filename/stem.

If two files contain the same internal display name, they remain distinct profiles because their filenames are distinct. The WebUI disambiguates duplicate display names with the filename rather than silently collapsing one record.

## Generation Profiles are not Replay

Replay preserves a historical generation. A Generation Profile is an editable/reusable setup. A Replay can be used as the source for a profile, but the profile should not be treated as the provenance record for the original image.

## Generation Profiles are not Prompt Presets

Generation Profiles deliberately cover much more than prompt text. Use a Prompt Preset when you want to reuse positive/negative prompt text while keeping the rest of the current generation controls unchanged.

## Related topics

* [Saved Generation Concepts: Replay, Generation Profiles, and Prompt Presets](saved_generation_concepts.md)
* [Replay](replay.md)
* [Prompt Presets](prompt_presets.md)
