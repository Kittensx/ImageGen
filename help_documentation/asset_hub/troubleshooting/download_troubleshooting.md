---
title: Asset Hub Download Troubleshooting
section: Asset Hub / Troubleshooting
Audience: user
feature_id: asset_hub.downloads.troubleshooting
status: current
---

# Download troubleshooting

## Authentication required

If Civitai rejects a request with an authentication error, configure or validate a provider token. A token may be session-only, stored in the operating-system credential store, or supplied through `CIVITAI_API_TOKEN`.

## Hash mismatch

A hash mismatch means the staged file's SHA-256 differs from the provider's expected SHA-256. IMAGE_GEN leaves the job failed and does not treat the staged payload as installable.

Recreate the download plan and retry. Repeated mismatches should be treated as a provider/content integrity problem rather than bypassed.

## Content-length mismatch

The provider metadata or HTTP response described a different size than IMAGE_GEN received. The job fails rather than accepting an ambiguous file.

## Resume restarted from zero

This is expected when a saved partial file can no longer be proven to represent the same remote object. Common causes include a changed `ETag`, changed `Last-Modified`, changed provider file metadata, or a server that does not honor byte-range requests.

## Redirect blocked

Asset Hub follows only bounded HTTPS redirects to routable hosts. Redirects to localhost, private networks, link-local addresses, file URLs, or redirect loops are rejected intentionally.

## Insufficient disk space

Downloads are staged under the configured temporary root and require enough free space for the remaining transfer plus a safety reserve. Free space on the staging drive or move the configured temporary directory to a drive with sufficient capacity.

## A completed download is not In Library

This is intentional in Phase 02. A verified staged download becomes a local library asset only after a later Asset Hub installation phase classifies and commits it successfully.
