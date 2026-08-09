import { api } from "../api.js?v=changelog1";
import { productName } from "../branding.js?v=brand1";
import { $ } from "../utils.js";
import { setActionIcon } from "../components/action-icons.js?v=0.1.1";
import { renderMarkdown } from "../components/markdown-reader.js?v=component-shell1";

const DEFAULT_VISIBLE_ENTRIES = 3;
const GITHUB_REPOSITORY_ROOT = "https://github.com/Kittensx/ImageGen";

let entries = [];
let showingAll = false;
let githubDirectoryUrl = "https://github.com/Kittensx/ImageGen/tree/main/changelog";

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = String(value ?? "");
}

function entryButton(entry, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "home-changelog-entry ui-icon-control";
  button.dataset.changelogDate = entry.date;
  button.setAttribute("aria-label", `Open changelog for ${entry.date}`);

  const date = document.createElement("strong");
  date.className = "home-changelog-date";
  date.textContent = entry.date;

  const label = document.createElement("span");
  label.className = "home-changelog-entry-label";
  label.textContent = index === 0 ? "Latest update" : "Release notes";

  const icon = document.createElement("span");
  icon.className = "ui-icon home-changelog-entry-icon";
  icon.dataset.icon = "chevron-right";
  icon.setAttribute("aria-hidden", "true");

  button.append(date, label, icon);
  button.addEventListener("click", () => openEntry(entry.date));
  return button;
}

function renderEntries() {
  const list = $("#homeChangelogList");
  const toggle = $("#homeChangelogViewAll");
  if (!list) return;
  list.replaceChildren();

  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "home-changelog-empty";
    empty.textContent = "No changelog entries are available yet.";
    list.append(empty);
    if (toggle) toggle.hidden = true;
    return;
  }

  const visible = showingAll ? entries : entries.slice(0, DEFAULT_VISIBLE_ENTRIES);
  visible.forEach((entry, index) => list.append(entryButton(entry, index)));

  if (toggle) {
    toggle.hidden = entries.length <= DEFAULT_VISIBLE_ENTRIES;
    const label = showingAll ? "Show recent changelog entries" : `View all ${entries.length} changelog entries`;
    setActionIcon(toggle, showingAll ? "chevron-up" : "chevron-down", { label, title: label, replace: true });
    toggle.setAttribute("aria-expanded", String(showingAll));
  }
}

function setReaderBusy(busy) {
  const reader = $("#homeChangelogReader");
  if (reader) reader.setAttribute("aria-busy", String(Boolean(busy)));
}

async function openEntry(entryDate) {
  const dialog = $("#homeChangelogReader");
  const content = $("#homeChangelogMarkdown");
  const source = $("#homeChangelogReaderSource");
  if (!dialog || !content) return;

  setText("#homeChangelogReaderDate", entryDate);
  setText("#homeChangelogReaderTitle", "Loading changelog...");
  content.replaceChildren();
  const loading = document.createElement("p");
  loading.className = "home-changelog-empty";
  loading.textContent = "Loading release notes...";
  content.append(loading);
  if (source) source.hidden = true;
  if (!dialog.open) dialog.showModal();
  setReaderBusy(true);

  try {
    const payload = await api.changelogEntry(entryDate);
    setText("#homeChangelogReaderDate", payload.date || entryDate);
    setText("#homeChangelogReaderTitle", payload.title || `${productName()} update - ${entryDate}`);
    renderMarkdown(content, payload.markdown || "", { repositoryRoot: GITHUB_REPOSITORY_ROOT, basePath: "changelog", entryDate: payload.date || entryDate });
    if (source) {
      source.href = payload.github_url || `${GITHUB_REPOSITORY_ROOT}/blob/main/changelog/${entryDate}.md`;
      source.hidden = false;
    }
  } catch (error) {
    setText("#homeChangelogReaderTitle", "Unable to load changelog");
    content.replaceChildren();
    const message = document.createElement("p");
    message.className = "home-changelog-error";
    message.textContent = String(error?.message || error || "Unable to load the selected changelog entry.");
    content.append(message);
  } finally {
    setReaderBusy(false);
  }
}

async function refreshCatalog() {
  setText("#homeChangelogStatus", "Checking for developer updates...");
  try {
    const payload = await api.changelog();
    entries = Array.isArray(payload.entries) ? payload.entries : [];
    entries.sort((left, right) => String(right.date || "").localeCompare(String(left.date || "")));
    githubDirectoryUrl = payload.github_directory_url || githubDirectoryUrl;
    const sourceLink = $("#homeChangelogGitHub");
    if (sourceLink) sourceLink.href = githubDirectoryUrl;
    if (payload.remote_available) {
      setText("#homeChangelogStatus", `${productName()} changelog entries from the public repository.`);
    } else {
      setText("#homeChangelogStatus", `GitHub is unavailable; showing the changelog bundled with this ${productName()} build.`);
    }
    renderEntries();
  } catch (error) {
    entries = [];
    renderEntries();
    setText("#homeChangelogStatus", String(error?.message || error || "Unable to load changelog entries."));
  }
}

export function bindChangelog() {
  const toggle = $("#homeChangelogViewAll");
  toggle?.addEventListener("click", () => {
    showingAll = !showingAll;
    renderEntries();
  });

  $("#homeChangelogReaderClose")?.addEventListener("click", () => {
    $("#homeChangelogReader")?.close();
  });

  $("#homeChangelogReader")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });

  refreshCatalog();
  return { refresh: refreshCatalog };
}
