---
title: Asset Hub Provider Authentication
summary: How provider credentials are stored, used, and protected.
category: Asset Hub
audience: user
status: current
keywords:
- authentication
- token
- civitai
- credential
- security
related:
- asset_hub/index
- asset_hub/downloads/download_queue_and_staging
- asset_hub/troubleshooting/asset_hub_troubleshooting
featured: false
media: []
external_links: []
---

# Asset Hub Provider Authentication

Provider credentials are handled by the IMAGE_GEN backend. The browser is told whether a credential is configured, but the stored secret is not returned to the browser after submission.

Civitai can use a session credential, the `CIVITAI_API_TOKEN` environment variable, or supported operating-system credential storage. Authentication is attached only to the expected provider host. If a download redirects to another host, the Civitai bearer credential is not forwarded.

Provider credentials are not stored in ordinary WebUI settings, install provenance, download reports, or theme/workspace configuration.
