---
title: Model and Asset Catalog
summary: How IMAGE_GEN discovers checkpoints, LoRAs, VAEs, textual inversions, previews, metadata, and recent generated outputs.
category: Models
audience: user
status: current
keywords:
- asset catalog
- checkpoints
- lora
- vae
- textual inversion
- preview
- civitai
related:
- models/model_component_registry
- generation/replay_and_preflight
featured: false
media: []
external_links: []
---

# Model and Asset Catalog

IMAGE_GEN uses one asset-catalog interface for the model and generation resources shown throughout the WebUI.

## Cataloged asset types

The catalog currently manages first-class records for:

- checkpoints
- LoRAs
- VAEs
- textual inversions

It also supports recent-output browsing so generated images can be reused by workflows such as preview selection and replay.

## Stable file identity and display labels

The technical asset `name` identifies the local file and remains stable when you assign a nickname or other display metadata.

The label shown in the UI may instead come from a user nickname or compatible embedded metadata. Changing that label does not rename the underlying model file or change the path used for generation.

## Refreshing assets

Asset discovery is refresh-based. A refresh rescans configured model locations and rebuilds the relevant catalog records. Metadata edits and preview changes increment that catalog's revision so open UI views can refresh the affected card without requiring a full application restart.

## LoRA inspection

LoRA records can include local technical inspection and compatibility information. IMAGE_GEN caches inspection results when the underlying LoRA file has not changed.

## CivitAI enrichment

When configured, IMAGE_GEN can enrich supported local assets with CivitAI metadata. Provider metadata is additional information attached to the local asset; it does not replace the local file's technical identity.

CivitAI authentication remains optional. A failed or unavailable provider lookup does not remove the local asset from the catalog.

## Asset previews

IMAGE_GEN can associate a local preview image with an asset. Existing generated outputs can also be selected as previews where that workflow is exposed by the WebUI.

## Recent outputs

The output catalog scans configured managed output roots for supported image files. Output details are read from IMAGE_GEN metadata when available and are used by recent-gallery and replay workflows.
