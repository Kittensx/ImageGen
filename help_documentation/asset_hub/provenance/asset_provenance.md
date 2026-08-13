---
title: Asset Provenance and Offline Metadata
summary: How installed assets retain provider identity and useful metadata for offline
  use.
category: Asset Hub
audience: user
status: current
keywords:
- provenance
- metadata
- offline
- sidecar
- imagegen json
related:
- asset_hub/installation/classification_and_installation
- asset_hub/index
- asset_hub/troubleshooting/asset_hub_troubleshooting
featured: false
media: []
external_links: []
---

# Asset Provenance and Offline Metadata

Assets installed through Asset Hub keep local provenance so useful information remains available even when the provider is offline.

IMAGE_GEN stores durable metadata in the asset's existing `.imagegen.json` sidecar convention and maintains a derived searchable index at:

`data/cache/asset-hub/installed-assets-v1.json`

Captured fields can include provider/model/version/file IDs, source page, author, description, tags, trained words, base model, original filename, file size, verified SHA-256, classification result, install timestamp, scan metadata, and local install path.

Signed or temporary provider delivery URLs are **not persisted**. The durable human-facing source page and normalized provider identity are stored instead.

Provider descriptions are also converted to sanitized plain text for safe card/search use. Later Asset Browser phases can use this local metadata to build info cards without requiring an online provider request.

The sidecar is the durable attachment. The JSON index is a derived fast-search view and can be rebuilt from local metadata in future lifecycle phases.
