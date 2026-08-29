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

## Provider closed the connection

A provider or CDN can close a long transfer before the advertised file body is complete. IMAGE_GEN preserves the staged partial file, retries transient transfer failures automatically, and uses a byte-range request on the next attempt. If automatic retries are exhausted, use **Resume**; it should continue from the preserved byte count rather than silently starting over.

## Resume restarted from zero

IMAGE_GEN can resume even when a provider omits `ETag` and `Last-Modified`, provided the provider identity and returned `Content-Range` remain compatible and the final integrity checks succeed. A restart from zero is reserved for cases where safe append cannot be established, such as a changed validator/file identity, an invalid range response, or a provider that explicitly ignores the Range request and sends a complete `200` response. The Downloads panel records this decision in the job note.

## Find the asset behind a download

Recent download rows remain in the Downloads panel after they finish or fail. Click a row to reopen the matching provider model and reselect its version/file in Asset Browser. Use **Save** on a download row, or **Save for later** on the asset detail card, to keep a browser bookmark for later.

## Redirect blocked

Asset Hub follows only bounded HTTPS redirects to routable hosts. Redirects to localhost, private networks, link-local addresses, file URLs, or redirect loops are rejected intentionally.

## Insufficient disk space

Downloads are staged under the configured temporary root and require enough free space for the remaining transfer plus a safety reserve. Free space on the staging drive or move the configured temporary directory to a drive with sufficient capacity.

## A completed download is not In Library

This is intentional in Phase 02. A verified staged download becomes a local library asset only after a later Asset Hub installation phase classifies and commits it successfully.
