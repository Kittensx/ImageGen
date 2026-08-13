---
title: Help Topic Media and External Links
summary: How public help safely uses local images/video and explicit external HTTPS
  links.
category: Help Center
audience: user
status: current
keywords:
- help
- media
- image
- video
- external links
- youtube
related:
- home/help_center
- home/changelog
- workspace/workspace_manager
featured: false
media: []
external_links: []
---

# Help Topic Media and External Links

Help topics may include images, graphics, and local video stored beneath the public `help_documentation/` root.

Images can be referenced from Markdown. Structured topic media may also be declared in front matter for image or video presentation. Local media is served only from the public help root; paths that escape that root are rejected.

External resources use explicit HTTPS links. IMAGE_GEN does not silently embed third-party remote media when a help topic opens, which avoids contacting an external service without the user choosing the link.

This makes future tutorial links, including a YouTube how-to or livestream, possible without giving external HTML or scripts access to the Help Center renderer.
