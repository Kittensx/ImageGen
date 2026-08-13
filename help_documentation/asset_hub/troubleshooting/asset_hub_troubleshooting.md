---
title: Asset Hub Troubleshooting
summary: Common Asset Hub download, install, quarantine, metadata, and compatibility
  issues.
category: Asset Hub
audience: user
status: current
keywords:
- troubleshooting
- error
- quarantine
- download
- install
related:
- asset_hub/index
- asset_hub/downloads/resume_and_verification
- asset_hub/installation/conflicts_and_quarantine
featured: false
media: []
external_links: []
---

# Asset Hub Troubleshooting

## Download completed but the asset is not In Library

This is expected until the verified staged file has also completed classification and installation.

## Asset was quarantined

IMAGE_GEN could verify the bytes but could not safely auto-classify or route the file. Review the quarantine reason. Quarantine is preferred over guessing an asset type.

## The installed filename has a hash suffix

A different file already occupied the requested destination name. The default conflict policy preserves the existing file and installs the new content under a hash-suffixed name.

## Provider metadata refresh fails

The locally captured provenance remains available. IMAGE_GEN marks provider synchronization stale rather than discarding the metadata captured at install time.

## An upscaler is installed but hidden

The backend may consider it unqualified or incompatible. Favorites do not override that decision.

## Removal behavior

Phase 03 does not permanently delete Asset Hub-owned model files. Until the dedicated lifecycle/Recycle Bin phase is implemented, the Phase 03 uninstall path moves an Asset Hub-owned file out of model scan roots into an IMAGE_GEN recovery location after containment checks.
