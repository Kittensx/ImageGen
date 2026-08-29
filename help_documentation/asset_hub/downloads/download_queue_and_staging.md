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

After a transfer completes and passes verification, IMAGE_GEN automatically classifies and commits the asset into the configured library. There is no separate Install button for the normal completed-download path. Assets that cannot be safely routed are quarantined for review instead of being written into a live model folder.

The Downloads panel keeps persistent history and provides status/text filters, bulk pause/resume/cancel controls, safe history clearing, and a **Clean old partials** action. Clearing history does not remove resumable partial data.

On startup, IMAGE_GEN cleans stale partial payloads only when they are no longer recoverable. Paused jobs and failed jobs that can still resume are preserved. Completed verified staging is also preserved if automatic library finalization is still pending.
