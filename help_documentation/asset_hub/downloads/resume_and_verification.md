---
title: Asset Hub Resume and Verification
summary: How interrupted downloads resume safely and how hashes and sizes are verified.
category: Asset Hub
audience: user
status: current
keywords:
- resume
- verification
- sha256
- etag
- download
related:
- asset_hub/downloads/download_queue_and_staging
- asset_hub/installation/classification_and_installation
- asset_hub/troubleshooting/asset_hub_troubleshooting
featured: false
media: []
external_links: []
---

# Asset Hub Resume and Verification

Asset Hub verifies downloaded bytes before installation.

When the provider supplies an expected file size or SHA-256, IMAGE_GEN compares the completed staged file to that information. A mismatch stops the lifecycle before installation.

Partial downloads are resumed only when the remote server supports Range requests and the saved remote identity remains compatible with the previous transfer. If the ETag or Last-Modified identity changes, IMAGE_GEN discards the unsafe continuation and restarts the transfer instead of joining bytes from two different remote files.

Verification reports intentionally omit credentials and signed delivery URLs.
