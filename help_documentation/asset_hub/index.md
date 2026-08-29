---
title: Asset Hub
summary: Discover, download, verify, install, and manage provider assets safely.
category: Asset Hub
audience: user
status: current
keywords:
- asset hub
- civitai
- models
- download
- install
related:
- asset_hub/downloads/download_queue_and_staging
- asset_hub/installation/classification_and_installation
- asset_hub/provenance/asset_provenance
- asset_hub/troubleshooting/asset_hub_troubleshooting
featured: true
media: []
external_links: []
---

# Asset Hub

Asset Hub is IMAGE_GEN's provider-neutral system for discovering, downloading, verifying, classifying, installing, and later managing model-related assets. Civitai is the first supported provider.

## Current lifecycle

1. Discover a compatible provider asset.
2. Create a secure download plan from provider identities.
3. Download into temporary staging.
4. Verify file size and SHA-256.
5. Create an install plan.
6. Safely inspect and classify the staged file.
7. Install only into a configured IMAGE_GEN asset root, or quarantine it when automatic classification is unsafe.
8. Store provider provenance in a portable `.imagegen.json` sidecar and a searchable local index.

Interrupted transfers preserve partial bytes and can reconnect with HTTP Range requests. Recent download rows also retain the provider model/version/file identity so you can click back to the originating asset or save it for later.

A downloaded file is **not** considered `In Library` until the install phase completes successfully.

## Help categories

* [Provider authentication](authentication/provider_authentication.md)
* [Download queue and staging](downloads/download_queue_and_staging.md)
* [Resume and verification](downloads/resume_and_verification.md)
* [Classification and installation](installation/classification_and_installation.md)
* [Conflicts and quarantine](installation/conflicts_and_quarantine.md)
* [Asset provenance](provenance/asset_provenance.md)
* [Upscaler favorites and compatibility](upscalers/favorites_and_compatibility.md)
* [Troubleshooting](troubleshooting/asset_hub_troubleshooting.md)
