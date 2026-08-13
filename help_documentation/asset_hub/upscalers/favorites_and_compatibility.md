---
title: Upscaler Favorites and Compatibility
summary: How local upscaler favorites interact with compatibility and qualification.
category: Asset Hub
audience: user
status: current
keywords:
- upscaler
- favorites
- compatibility
- esrgan
- realesrgan
related:
- asset_hub/index
- asset_hub/installation/classification_and_installation
- asset_hub/troubleshooting/asset_hub_troubleshooting
featured: false
media: []
external_links: []
---

# Upscaler Favorites and Compatibility

Upscaler favorites are local user preferences, not portable provider provenance. They are stored separately under:

`data/cache/upscalers/favorites-v1.json`

The favorite record uses IMAGE_GEN's stable upscaler IDs so a rescan does not depend on a display filename.

Compatibility remains backend-owned. The Asset Hub compatibility API can return all installed upscalers, currently compatible/selectable upscalers, favorite compatible upscalers, and incompatible/hidden favorite IDs for diagnostics.

A favorite never overrides qualification or compatibility. If a favorite upscaler is missing or becomes incompatible, IMAGE_GEN keeps the favorite preference but does not offer that item as a valid generation choice.
