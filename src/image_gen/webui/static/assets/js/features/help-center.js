import { api } from "../api.js?v=help-center1";
import { componentShellCapability } from "../components/component-shell.js?v=content-capabilities2";

let catalog = null;
let currentTopicId = "";
let currentTopicPayload = null;
let navigationHistory = [];
let historyIndex = -1;
let searchSequence = 0;
const searchTimers = new WeakMap();

function node(id) {
  return document.getElementById(id);
}

function ensureHelpCenterDialog() {
  let dialog = node("helpCenterDialog");
  if (dialog) return dialog;
  dialog = document.createElement("dialog");
  dialog.className = "help-center-dialog";
  dialog.id = "helpCenterDialog";
  dialog.setAttribute("aria-labelledby", "helpCenterTopicTitle");
  dialog.innerHTML = `
    <div class="help-center-shell">
      <header class="help-center-header">
        <div class="help-center-title-group"><span class="home-eyebrow">IMAGE_GEN Help Center</span><h3 id="helpCenterTopicTitle">Help Center</h3><small id="helpCenterTopicCategory">Documentation</small></div>
        <div class="help-center-header-actions">
          <button class="ui-action-button ui-icon-control" id="helpCenterBack" type="button" aria-label="Previous help topic" title="Previous help topic" disabled><span aria-hidden="true">←</span></button>
          <button class="ui-action-button ui-icon-control" id="helpCenterForward" type="button" aria-label="Next help topic" title="Next help topic" disabled><span aria-hidden="true">→</span></button>
          <button class="ui-action-button ui-icon-control" id="helpCenterClose" type="button" aria-label="Close Help Center" title="Close Help Center"><span class="ui-icon" data-icon="close" aria-hidden="true"></span></button>
        </div>
      </header>
      <div class="help-center-search-row">
        <div class="help-search-field">
          <label for="helpCenterSearch">Search help</label>
          <input id="helpCenterSearch" type="search" autocomplete="off" placeholder="Type at least 3 letters…" aria-controls="helpCenterSearchResults">
          <div class="help-search-dropdown" id="helpCenterSearchResults" role="listbox" hidden></div>
        </div>
      </div>
      <div class="help-center-layout">
        <aside class="help-center-navigation"><h4>Topics</h4><nav class="help-topic-tree" id="helpCenterTree" aria-label="Help topic tree"></nav></aside>
        <main class="help-center-document">
          <div class="help-center-media" id="helpCenterMedia" hidden></div>
          <article class="shared-markdown-content" id="helpCenterMarkdown" tabindex="0"></article>
          <section class="help-center-related" id="helpCenterRelatedTopics" hidden></section>
          <section class="help-center-external" id="helpCenterExternalLinks" hidden></section>
        </main>
      </div>
    </div>`;
  document.body.append(dialog);
  return dialog;
}

function encodeTopicPath(topicId) {
  return String(topicId || "").split("/").filter(Boolean).map((part) => encodeURIComponent(part)).join("/");
}

function topicCategory(topic = {}) {
  return Array.isArray(topic.categoryPath) && topic.categoryPath.length ? topic.categoryPath.join(" / ") : "Help Center";
}

function normalizeHelpRelativePath(rawHref, baseTopicId = currentTopicId) {
  const raw = String(rawHref || "").trim().replaceAll("\\", "/");
  if (!raw || /^https:\/\//i.test(raw)) return null;
  if (raw.startsWith("#")) return { topicId: baseTopicId, anchor: raw.slice(1) };
  const [withoutHash, anchor = ""] = raw.split("#", 2);
  const baseParts = String(baseTopicId || "index").split("/").filter(Boolean);
  baseParts.pop();
  const parts = withoutHash.split("/");
  for (const part of parts) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!baseParts.length) return null;
      baseParts.pop();
    } else {
      baseParts.push(part);
    }
  }
  const last = baseParts[baseParts.length - 1] || "";
  if (last.endsWith(".md")) baseParts[baseParts.length - 1] = last.slice(0, -3);
  return { topicId: baseParts.join("/"), anchor };
}

function helpMediaUrl(rawSrc, baseTopicId = currentTopicId) {
  const resolved = normalizeHelpRelativePath(rawSrc, baseTopicId);
  if (!resolved?.topicId) return "";
  const path = encodeTopicPath(resolved.topicId);
  return `/api/help/media/${path}`;
}

function setSearchHint(input, dropdown) {
  if (!catalog) return;
  const minimum = Number(catalog.minimumSearchCharacters || 3);
  const length = String(input.value || "").trim().length;
  if (length && length < minimum) {
    dropdown.replaceChildren();
    const hint = document.createElement("div");
    hint.className = "help-search-hint";
    hint.textContent = `Type at least ${minimum} characters to search help.`;
    dropdown.append(hint);
    dropdown.hidden = false;
    return true;
  }
  if (!length) dropdown.hidden = true;
  return false;
}

function searchResultButton(result) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "help-search-result";
  const title = document.createElement("strong");
  title.textContent = String(result.title || result.id || "Help topic");
  const category = document.createElement("small");
  category.textContent = topicCategory(result);
  const snippet = document.createElement("span");
  snippet.textContent = String(result.snippet || result.summary || "");
  button.append(title, category, snippet);
  button.addEventListener("click", () => {
    openHelpTopic(result.id);
    document.querySelectorAll(".help-search-dropdown").forEach((menu) => { menu.hidden = true; });
  });
  return button;
}

function bindSearch(input, dropdown) {
  if (!input || !dropdown || input.dataset.helpSearchBound === "true") return;
  input.dataset.helpSearchBound = "true";
  input.addEventListener("input", () => {
    if (setSearchHint(input, dropdown)) return;
    const query = String(input.value || "").trim();
    const minimum = Number(catalog?.minimumSearchCharacters || 3);
    if (query.length < minimum) return;
    const previousTimer = searchTimers.get(input);
    if (previousTimer) clearTimeout(previousTimer);
    const timer = setTimeout(async () => {
      const sequence = ++searchSequence;
      dropdown.replaceChildren();
      const loading = document.createElement("div");
      loading.className = "help-search-hint";
      loading.textContent = "Searching help…";
      dropdown.append(loading);
      dropdown.hidden = false;
      try {
        const payload = await api.helpSearch(query);
        if (sequence !== searchSequence || String(input.value || "").trim() !== query) return;
        dropdown.replaceChildren();
        const results = Array.isArray(payload.results) ? payload.results : [];
        if (!results.length) {
          const empty = document.createElement("div");
          empty.className = "help-search-hint";
          empty.textContent = "No matching help topics.";
          dropdown.append(empty);
        } else {
          results.forEach((result) => dropdown.append(searchResultButton(result)));
        }
      } catch (error) {
        dropdown.replaceChildren();
        const failure = document.createElement("div");
        failure.className = "help-search-hint help-search-hint--error";
        failure.textContent = `Unable to search help: ${error?.message || error}`;
        dropdown.append(failure);
      }
    }, 160);
    searchTimers.set(input, timer);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") dropdown.hidden = true;
  });
}

function topicButton(topic, className = "help-topic-button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.dataset.helpTopicId = String(topic.id || "");
  const title = document.createElement("strong");
  title.textContent = String(topic.title || topic.id || "Topic");
  button.append(title);
  if (topic.summary) {
    const summary = document.createElement("span");
    summary.textContent = String(topic.summary);
    button.append(summary);
  }
  button.addEventListener("click", () => openHelpTopic(topic.id));
  return button;
}

function renderTreeNodes(nodes, container, depth = 0) {
  for (const category of Array.isArray(nodes) ? nodes : []) {
    const details = document.createElement("details");
    details.className = "help-tree-category";
    details.open = depth < 1;
    const summary = document.createElement("summary");
    summary.textContent = String(category.title || "Topics");
    details.append(summary);
    const group = document.createElement("div");
    group.className = "help-tree-group";
    if (category.landingTopicId) {
      group.append(topicButton({ id: category.landingTopicId, title: "Overview" }, "help-tree-topic help-tree-topic--overview"));
    }
    for (const topic of Array.isArray(category.topics) ? category.topics : []) {
      group.append(topicButton(topic, "help-tree-topic"));
    }
    renderTreeNodes(category.children, group, depth + 1);
    details.append(group);
    container.append(details);
  }
}

function renderCatalog() {
  ensureHelpCenterDialog();
  const tree = node("helpCenterTree");
  const compactTree = node("homeHelpTopicTree");
  if (tree) {
    tree.replaceChildren();
    renderTreeNodes(catalog?.tree || [], tree);
  }
  if (compactTree) {
    compactTree.replaceChildren();
    renderTreeNodes(catalog?.tree || [], compactTree);
  }
  const suggested = node("homeHelpSuggestedTopics");
  if (suggested) {
    suggested.replaceChildren();
    for (const topic of Array.isArray(catalog?.featured) ? catalog.featured : []) {
      suggested.append(topicButton(topic, "home-help-suggested-topic"));
    }
  }
}

function updateHistoryButtons() {
  const back = node("helpCenterBack");
  const forward = node("helpCenterForward");
  if (back) back.disabled = historyIndex <= 0;
  if (forward) forward.disabled = historyIndex < 0 || historyIndex >= navigationHistory.length - 1;
}

function pushHistory(topicId, anchor = "") {
  const entry = { topicId, anchor };
  const current = navigationHistory[historyIndex];
  if (current?.topicId === topicId && current?.anchor === anchor) return;
  navigationHistory = navigationHistory.slice(0, historyIndex + 1);
  navigationHistory.push(entry);
  historyIndex = navigationHistory.length - 1;
  updateHistoryButtons();
}

function renderExternalLinks(topic) {
  const section = node("helpCenterExternalLinks");
  if (!section) return;
  section.replaceChildren();
  const links = Array.isArray(topic.externalLinks) ? topic.externalLinks : [];
  section.hidden = !links.length;
  if (!links.length) return;
  const heading = document.createElement("h4");
  heading.textContent = "External resources";
  section.append(heading);
  const list = document.createElement("div");
  list.className = "help-external-links";
  for (const item of links) {
    const link = document.createElement("a");
    link.href = String(item.href || "");
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = String(item.label || "Open external resource");
    list.append(link);
  }
  section.append(list);
}

function renderRelatedTopics(topic) {
  const section = node("helpCenterRelatedTopics");
  if (!section) return;
  section.replaceChildren();
  const related = Array.isArray(topic.relatedTopics) ? topic.relatedTopics : [];
  section.hidden = !related.length;
  if (!related.length) return;
  const heading = document.createElement("h4");
  heading.textContent = "Related topics";
  section.append(heading);
  const grid = document.createElement("div");
  grid.className = "help-related-grid";
  related.forEach((item) => grid.append(topicButton(item, "help-related-topic")));
  section.append(grid);
}

function markActiveTopic(topicId) {
  document.querySelectorAll("[data-help-topic-id]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.helpTopicId === topicId);
  });
}

async function renderTopic(topic, anchor = "") {
  const component = document.querySelector('[data-workspace-component="home.help-center"]');
  const markdown = componentShellCapability(component, "content.markdown");
  const media = componentShellCapability(component, "content.media");
  if (!markdown || !media) throw new Error("Help Center content capabilities are unavailable.");

  node("helpCenterTopicTitle").textContent = String(topic.title || "Help topic");
  node("helpCenterTopicCategory").textContent = topicCategory(topic);
  const mediaContainer = node("helpCenterMedia");
  mediaContainer.replaceChildren();
  media.renderCollection(mediaContainer, topic.media || []);
  mediaContainer.hidden = !(topic.media || []).length;

  const content = node("helpCenterMarkdown");
  markdown.render(content, topic.markdown || "", {
    preserveRelative: true,
    initialAnchor: anchor,
    resolveMedia: (src) => helpMediaUrl(src, topic.id),
    onNavigate: (href) => {
      const target = normalizeHelpRelativePath(href, topic.id);
      if (target?.topicId) openHelpTopic(target.topicId, { anchor: target.anchor || "" });
    },
    onAnchor: (nextAnchor) => {
      const current = navigationHistory[historyIndex];
      if (current) current.anchor = nextAnchor;
    },
  });
  renderExternalLinks(topic);
  renderRelatedTopics(topic);
  markActiveTopic(topic.id);
}

export async function openHelpTopic(topicId, { anchor = "", push = true } = {}) {
  const normalized = String(topicId || catalog?.rootTopicId || "index").trim();
  if (!normalized) return;
  const dialog = ensureHelpCenterDialog();
  if (!dialog.open) dialog.showModal();
  node("helpCenterTopicTitle").textContent = "Loading help…";
  node("helpCenterMarkdown").textContent = "Loading topic…";
  try {
    const payload = await api.helpTopic(normalized);
    const topic = payload.topic || {};
    currentTopicId = String(topic.id || normalized);
    currentTopicPayload = topic;
    if (push) pushHistory(currentTopicId, anchor);
    await renderTopic(topic, anchor);
  } catch (error) {
    node("helpCenterTopicTitle").textContent = "Unable to load help";
    node("helpCenterMarkdown").textContent = String(error?.message || error || "Unable to load the selected help topic.");
  }
}

export function openHelpCenter(topicId = "", anchor = "") {
  const dialog = ensureHelpCenterDialog();
  if (!dialog.open) dialog.showModal();
  return openHelpTopic(topicId || currentTopicId || catalog?.rootTopicId || "index", { anchor });
}

function bindHistory() {
  node("helpCenterBack")?.addEventListener("click", () => {
    if (historyIndex <= 0) return;
    historyIndex -= 1;
    updateHistoryButtons();
    const item = navigationHistory[historyIndex];
    openHelpTopic(item.topicId, { anchor: item.anchor, push: false });
  });
  node("helpCenterForward")?.addEventListener("click", () => {
    if (historyIndex >= navigationHistory.length - 1) return;
    historyIndex += 1;
    updateHistoryButtons();
    const item = navigationHistory[historyIndex];
    openHelpTopic(item.topicId, { anchor: item.anchor, push: false });
  });
}

export async function bindHelpCenter() {
  const component = document.querySelector('[data-workspace-component="home.help-center"]');
  if (!component) return null;
  if (!componentShellCapability(component, "content.markdown") || !componentShellCapability(component, "content.media")) {
    component.dataset.helpCenterState = "capability-error";
    return null;
  }
  try {
    ensureHelpCenterDialog();
    catalog = await api.helpCatalog();
    renderCatalog();
    bindSearch(node("homeHelpSearch"), node("homeHelpSearchResults"));
    bindSearch(node("helpCenterSearch"), node("helpCenterSearchResults"));
    node("homeHelpOpenButton")?.addEventListener("click", () => openHelpCenter());
    node("helpCenterClose")?.addEventListener("click", () => node("helpCenterDialog")?.close());
    node("helpCenterDialog")?.addEventListener("click", (event) => {
      if (event.target === node("helpCenterDialog")) node("helpCenterDialog")?.close();
    });
    bindHistory();
    updateHistoryButtons();
    window.imageGenHelpCenter = Object.freeze({ open: openHelpCenter, openTopic: openHelpTopic });
    component.dataset.helpCenterState = "ready";
    return catalog;
  } catch (error) {
    component.dataset.helpCenterState = "error";
    const status = node("homeHelpStatus");
    if (status) status.textContent = `Unable to load Help Center: ${error?.message || error}`;
    return null;
  }
}
