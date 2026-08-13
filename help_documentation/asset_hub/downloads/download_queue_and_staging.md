---
title: Asset Hub Download Queue and Staging
summary: How Asset Hub queues downloads and keeps them outside live model folders
  until verified.
category: Asset Hub
audience: user
status: current
keywords:
- download
- queue
- staging
- temporary
- asset hub
related:
- asset_hub/downloads/resume_and_verification
- asset_hub/installation/classification_and_installation
- asset_hub/authentication/provider_authentication
featured: false
media: []
external_links: []
---

# Asset Hub Download Queue and Staging

Asset Hub downloads are first written to temporary staging under IMAGE_GEN's configured `temporary_root`.

A Phase 02 download never writes directly into checkpoint, LoRA, VAE, embedding, ControlNet, ESRGAN, or RealESRGAN directories. This staging boundary lets IMAGE_GEN verify the download before any live library location changes.

By default, Asset Hub permits at most two simultaneous transfers. Jobs can be queued, cancelled, resumed when safe, and recovered after an IMAGE_GEN restart.

A completed download is a **verified staged artifact**, not an installed library asset.
