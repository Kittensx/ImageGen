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

Asset Hub verifies downloaded bytes before automatic library finalization.

When the provider supplies an expected file size or SHA-256, IMAGE_GEN compares the completed staged file to that information. A mismatch stops the lifecycle before automatic library finalization.

If a provider connection closes or times out during the file body, IMAGE_GEN preserves the partial file and treats the interruption as recoverable. The downloader retries automatically according to the configured retry count, and the Resume action uses an HTTP Range request starting at the exact staged byte count.

ETag or Last-Modified values are used as `If-Range` validators when the provider supplies them. They are not required for resume: provider/model/version/file identity, an exact `Content-Range` start, and the provider SHA-256 (when available) allow IMAGE_GEN to safely continue delivery paths that omit HTTP validators. The completed payload still must pass final size/hash verification before automatic library finalization.

If the provider returns a changed ETag/Last-Modified value, an invalid Content-Range, changed provider file identity, or a full `200` response instead of honoring Range, IMAGE_GEN does not append the response to the old partial. The download history records when a provider forced a restart from byte zero.

Verification reports intentionally omit credentials and signed delivery URLs.
