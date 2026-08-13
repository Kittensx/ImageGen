---
title: Asset Hub Conflicts and Quarantine
summary: What happens when filenames conflict or a downloaded asset cannot be classified
  safely.
category: Asset Hub
audience: user
status: current
keywords:
- conflict
- quarantine
- duplicate
- replace
- unsafe
related:
- asset_hub/installation/classification_and_installation
- asset_hub/troubleshooting/asset_hub_troubleshooting
- asset_hub/provenance/asset_provenance
featured: false
media: []
external_links: []
---

# Asset Hub Conflicts and Quarantine

Asset Hub does not silently overwrite model files.

## Name conflicts

* Same destination name and same SHA-256: reuse/deduplicate the existing file and attach Asset Hub provenance.
* Same destination name but different content: the default policy installs a hash-suffixed filename.
* Cancel policy: stop without changing the existing file.
* Replace policy: create a recovery backup before the atomic replacement.

If the same provider file identity is already installed but the local hash differs, IMAGE_GEN requires review rather than assuming the remote file should replace the local copy.

## Quarantine

Files that cannot be safely auto-classified are copied under:

`data/asset-hub/quarantine/<download-job-id>/`

A quarantine record stores the verified provider identity, reason code, safe inspection result, and sanitized original filename. Quarantine locations are outside model scan roots, so quarantined files are not offered for generation.

Examples include unsupported `.pth` architectures, unsafe checkpoint formats, unknown binary content, and other files without a qualified automatic inspector.
