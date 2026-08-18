---
title: Persistent Generation Queue
summary: How IMAGE_GEN restores queued and paused generations after the application is restarted.
category: Generation
audience: user
status: current
keywords:
- generation queue
- queue persistence
- restart
- paused jobs
- recovery
related:
- home/help_center
- workspace/workspace_manager
featured: false
media: []
external_links: []
---

# Persistent Generation Queue

IMAGE_GEN keeps recoverable generation queue state between application sessions.

## What is restored

When IMAGE_GEN starts again, it can restore:

- queued generations
- their explicit queue order
- jobs that you paused individually while they were waiting
- a whole-queue pause/hold
- recoverable progress information for a resident-runtime generation that was interrupted while the application was closing

Completed jobs, failed jobs, and generations that you explicitly cancelled are not placed back into the queue.

## What happens if IMAGE_GEN closes during a generation

If a generation was still active when IMAGE_GEN stopped or the process was interrupted, IMAGE_GEN restores that job at the front of the queue when possible.

The queue is held instead of immediately restarting the recovered job. Use **Resume** after you have confirmed that you want generation to continue.

This prevents reopening IMAGE_GEN from unexpectedly starting GPU work and gives you a chance to review a job that may have been interrupted in the middle of an image.

## Multi-image jobs

Resident-runtime jobs keep confirmed batch progress and seed history. IMAGE_GEN resumes from the persisted image-slot progress when possible.

An image that was still in progress at the exact moment the application stopped may need to be generated again after you resume. Image slots already recorded as completed are preserved as completed progress.

## Where the state is stored

Queue ordering and queue-hold state are stored in IMAGE_GEN's runtime data area. Individual job requests and progress continue to use the existing per-job runtime files.

These files are application/runtime data. You normally do not need to edit them manually.

## If a queue cannot be restored

If a previous queue-state file is missing or unreadable, IMAGE_GEN attempts to reconstruct recoverable queued jobs from their individual job records. Jobs that cannot be parsed are skipped rather than blocking application startup.
