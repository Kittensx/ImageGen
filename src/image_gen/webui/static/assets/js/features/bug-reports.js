import { api } from "../api.js?v=bug-reporter1";
import { productName } from "../branding.js?v=brand1";
import { $ , notify } from "../utils.js";

const GITHUB_SYNC_INTERVAL_MS = 30 * 60 * 1000;
let currentPayload = null;
let syncTimer = null;

function percent(value) {
  const numeric = Number(value || 0);
  return `${Math.round(Math.max(0, Math.min(1, numeric)) * 100)}%`;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let scaled = bytes;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  return `${scaled >= 10 || index === 0 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[index]}`;
}

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = String(value ?? "");
}

function renderHomeProfile(payload) {
  const profile = payload?.profile || {};
  const github = payload?.github || {};
  const badge = $("#homeBugReporterBadge");
  if (badge) {
    badge.textContent = profile.badge || "Bug Reporter · 0";
    badge.classList.toggle("is-active", Number(profile.reported || 0) > 0);
  }
  setText("#homeBugReportedCount", profile.reported || 0);
  setText("#homeBugOpenCount", profile.open || 0);
  setText("#homeBugResolvedCount", profile.resolved || 0);
  setText("#homeBugResolutionRate", percent(profile.resolution_rate));
  setText("#homeBugPendingCount", profile.pending || 0);
  setText("#homeBugNeedsReproductionCount", profile.needs_reproduction || 0);

  const syncStatus = $("#homeBugSyncStatus");
  if (syncStatus) {
    if (github.status === "ready") {
      syncStatus.textContent = `GitHub checked · ${profile.known_existing || 0} known duplicate${Number(profile.known_existing || 0) === 1 ? "" : "s"}`;
    } else if (github.status === "unavailable") {
      syncStatus.textContent = "GitHub status unavailable; new reports stay locked until duplicates can be checked.";
    } else {
      syncStatus.textContent = `${profile.pending || 0} local report${Number(profile.pending || 0) === 1 ? "" : "s"} awaiting review.`;
    }
  }
}

function reportStatus(report, githubReady) {
  const issue = report.github_issue || {};
  if (report.github_match === "local_report") {
    return issue.state === "closed"
      ? { label: "Reported · Resolved", className: "is-resolved" }
      : { label: "Reported · Open", className: "is-open" };
  }
  if (report.github_match === "known_issue") {
    return issue.state === "closed"
      ? { label: "Known issue · Closed", className: "is-known" }
      : { label: "Known issue · Open", className: "is-known" };
  }
  if (!report.local_artifact_present) {
    return { label: "History", className: "" };
  }
  if (report.requires_reproduction) {
    return { label: "Needs current-build reproduction", className: "is-pending" };
  }
  if (report.classification === "validation_event") {
    return { label: "Validation event · Review only", className: "is-known" };
  }
  if (report.classification === "wrapper_failure") {
    return { label: "Wrapper · Review underlying failure", className: "is-known" };
  }
  if (!githubReady) {
    return { label: "Awaiting duplicate check", className: "is-pending" };
  }
  return { label: "Pending review", className: "is-pending" };
}

function makeButton(label, className = "secondary-button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  return button;
}

function makeLink(label, href, className = "secondary-button") {
  const link = document.createElement("a");
  link.className = className;
  link.textContent = label;
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function addMeta(container, label, value) {
  const item = document.createElement("span");
  item.textContent = `${label}: ${value}`;
  container.append(item);
}

async function revealBundle(report) {
  try {
    await api.revealBugBundle(report.fingerprint);
  } catch (error) {
    notify(`Unable to reveal the report ZIP: ${error.message}`, "error");
  }
}

async function openIssue(report) {
  const popup = window.open("about:blank", "_blank");
  if (!popup) {
    notify(`The browser blocked the GitHub issue window. Allow popups for the local ${productName()} WebUI and try again.`, "error");
    return;
  }
  popup.document.title = "Opening GitHub issue…";
  try {
    const prepared = await api.prepareBugIssue(report.fingerprint);
    if (!prepared?.url) throw new Error("GitHub issue URL was not prepared.");
    popup.location.href = prepared.url;
    if (!prepared.known_issue) {
      notify(`GitHub opened for ${report.bundle_filename}. Attach the ZIP before submitting the issue.`);
    }
    await loadLocalReports({ renderDialogAfter: true });
  } catch (error) {
    popup.close();
    notify(`Unable to prepare the GitHub issue: ${error.message}`, "error");
  }
}

function buildReportCard(report, githubReady) {
  const card = document.createElement("article");
  card.className = "bug-report-card";
  card.dataset.fingerprint = report.fingerprint;

  const header = document.createElement("div");
  header.className = "bug-report-card-header";
  const headingGroup = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = report.issue_title || `${report.component || "Runtime"}: ${report.error_type || "Error"}`;
  const meta = document.createElement("div");
  meta.className = "bug-report-card-meta";
  addMeta(meta, "Latest version", report.version || "unknown");
  addMeta(meta, "Occurrences", report.occurrence_count || 1);
  addMeta(meta, "ZIP", `${report.bundle_filename || "unavailable"} · ${formatBytes(report.bundle_size)}`);
  if (Array.isArray(report.versions_seen) && report.versions_seen.length > 1) {
    addMeta(meta, "Versions seen", report.versions_seen.join(", "));
  }
  headingGroup.append(title, meta);

  const status = reportStatus(report, githubReady);
  const statusNode = document.createElement("span");
  statusNode.className = `bug-report-status ${status.className}`.trim();
  statusNode.textContent = status.label;
  header.append(headingGroup, statusNode);
  card.append(header);

  const error = document.createElement("p");
  error.className = "bug-report-card-copy";
  error.textContent = report.error_message || "No error message was recorded.";
  card.append(error);

  if (!report.reportable && report.review_note) {
    const review = document.createElement("div");
    review.className = "bug-report-notice";
    const strong = document.createElement("strong");
    strong.textContent = report.requires_reproduction ? "Reproduction required" : "Local review only";
    const detail = document.createElement("span");
    detail.textContent = report.review_note;
    review.append(strong, detail);
    card.append(review);
  }

  if (report.compact_bundle) {
    const compact = document.createElement("div");
    compact.className = "bug-report-notice";
    const strong = document.createElement("strong");
    strong.textContent = "Compact ZIP created";
    const detail = document.createElement("span");
    detail.textContent = "The original diagnostic folder was too large for a GitHub attachment target. Non-text files were omitted and listed inside the ZIP.";
    compact.append(strong, detail);
    card.append(compact);
  }

  const preview = document.createElement("details");
  preview.className = "bug-report-preview";
  const summary = document.createElement("summary");
  summary.textContent = "Preview exact GitHub issue text";
  const textarea = document.createElement("textarea");
  textarea.readOnly = true;
  textarea.value = report.issue_body || "";
  textarea.setAttribute("aria-label", "GitHub issue body preview");
  preview.append(summary, textarea);
  card.append(preview);

  const actions = document.createElement("div");
  actions.className = "bug-report-card-actions";
  if (report.local_artifact_present && report.bundle_filename) {
    const download = makeLink("Download ZIP", api.bugBundleUrl(report.fingerprint));
    download.removeAttribute("target");
    download.removeAttribute("rel");
    download.setAttribute("download", report.bundle_filename);
    actions.append(download);

    const reveal = makeButton("Show ZIP in Folder");
    reveal.addEventListener("click", () => revealBundle(report));
    actions.append(reveal);
  }

  const knownIssue = report.github_issue || {};
  if (report.github_match === "known_issue" || report.github_match === "local_report") {
    if (knownIssue.url) actions.append(makeLink(`Open GitHub #${knownIssue.number || ""}`, knownIssue.url, "primary-button"));
  } else if (report.local_artifact_present && report.reportable) {
    const submit = makeButton("Open GitHub Issue", "primary-button");
    submit.disabled = !githubReady || !report.bundle_within_github_limit;
    if (!githubReady) submit.title = `${productName()} must check GitHub for an existing fingerprint before creating a new issue.`;
    if (!report.bundle_within_github_limit) submit.title = "The prepared ZIP is still larger than GitHub's attachment limit.";
    submit.addEventListener("click", () => openIssue(report));
    actions.append(submit);
  }

  const privacy = document.createElement("span");
  privacy.className = "bug-report-privacy-note";
  privacy.textContent = "ZIPs are sanitized locally but may include prompts/settings. Inspect the ZIP before attaching it.";
  actions.append(privacy);
  card.append(actions);
  return card;
}

function renderDialog(payload) {
  const list = $("#bugReportList");
  if (!list) return;
  const reports = Array.isArray(payload?.reports) ? payload.reports : [];
  const profile = payload?.profile || {};
  const github = payload?.github || {};
  const githubReady = github.status === "ready";

  setText(
    "#bugReportDialogSummary",
    reports.length
      ? `${reports.length} unique bug fingerprint${reports.length === 1 ? "" : "s"} · ${profile.pending || 0} pending`
      : "No diagnostic failures are currently available to report.",
  );
  setText("#bugReportGithubStatus", github.message || "GitHub status has not been checked yet.");

  list.replaceChildren();
  if (!reports.length) {
    const empty = document.createElement("div");
    empty.className = "bug-report-empty";
    empty.textContent = "No reportable failure bundles were found under artifacts/diagnostics/failures.";
    list.append(empty);
    return;
  }
  reports.forEach((report) => list.append(buildReportCard(report, githubReady)));
}

function newlyResolved(previous, next) {
  const before = new Map((previous?.reports || []).map((report) => [report.fingerprint, report]));
  return (next?.reports || []).filter((report) => {
    if (report.github_match !== "local_report" || report.github_issue?.state !== "closed") return false;
    const old = before.get(report.fingerprint);
    return old?.github_issue?.state === "open";
  });
}

async function loadLocalReports({ renderDialogAfter = false } = {}) {
  try {
    const payload = await api.bugReports();
    currentPayload = payload;
    renderHomeProfile(payload);
    if (renderDialogAfter || $("#bugReportDialog")?.open) renderDialog(payload);
    return payload;
  } catch (error) {
    setText("#homeBugSyncStatus", `Unable to scan local bug reports: ${error.message}`);
    if (renderDialogAfter) setText("#bugReportGithubStatus", `Unable to scan local reports: ${error.message}`);
    return null;
  }
}

async function syncGithub({ announce = false } = {}) {
  const button = $("#syncBugReportsButton");
  const homeButton = $("#homeSyncBugReports");
  if (button) button.disabled = true;
  if (homeButton) homeButton.disabled = true;
  const previous = currentPayload;
  try {
    const payload = await api.syncBugReports();
    currentPayload = payload;
    renderHomeProfile(payload);
    window.dispatchEvent(new CustomEvent("image-gen-profile-refresh"));
    if ($("#bugReportDialog")?.open) renderDialog(payload);
    const resolved = newlyResolved(previous, payload);
    resolved.forEach((report) => {
      notify(`GitHub issue #${report.github_issue?.number || ""} for a bug you reported is now closed.`);
    });
    if (announce) {
      if (payload?.github?.status === "ready") notify("Bug-report status synchronized with GitHub.");
      else notify(payload?.github?.message || "GitHub status could not be refreshed.", "error");
    }
    return payload;
  } catch (error) {
    if (announce) notify(`Unable to synchronize GitHub issues: ${error.message}`, "error");
    return null;
  } finally {
    if (button) button.disabled = false;
    if (homeButton) homeButton.disabled = false;
  }
}

async function openDialog() {
  const dialog = $("#bugReportDialog");
  if (!dialog) return;
  if (!dialog.open) dialog.showModal();
  window.dispatchEvent(new CustomEvent("image-gen-bug-reporter-opened"));
  setText("#bugReportGithubStatus", "Checking local failures and GitHub fingerprints…");
  await loadLocalReports({ renderDialogAfter: true });
  await syncGithub();
}

function closeDialog() {
  const dialog = $("#bugReportDialog");
  if (!dialog?.open) return;
  dialog.close();
  window.dispatchEvent(new CustomEvent("image-gen-bug-reporter-closed"));
}

export function bindBugReporter() {
  $("#homeReviewBugReports")?.addEventListener("click", openDialog);
  window.addEventListener("image-gen-open-bug-reporter", openDialog);
  $("#homeSyncBugReports")?.addEventListener("click", () => syncGithub({ announce: true }));
  $("#syncBugReportsButton")?.addEventListener("click", () => syncGithub({ announce: true }));
  $("#closeBugReportDialog")?.addEventListener("click", closeDialog);
  $("#bugReportDialog")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeDialog();
  });
  $("#bugReportDialog")?.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDialog();
  });
  window.addEventListener("image-gen-bug-report-refresh", () => {
    loadLocalReports({ renderDialogAfter: Boolean($("#bugReportDialog")?.open) });
  });

  loadLocalReports().then(() => syncGithub());
  if (syncTimer) window.clearInterval(syncTimer);
  syncTimer = window.setInterval(() => syncGithub(), GITHUB_SYNC_INTERVAL_MS);

  return {
    refreshLocal: loadLocalReports,
    syncGithub,
    open: openDialog,
  };
}
