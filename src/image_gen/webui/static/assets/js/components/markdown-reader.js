import { componentCapability, registerComponentCapability } from "./capabilities.js?v=content-capabilities2";

const DEFAULT_REPOSITORY_ROOT = "https://github.com/Kittensx/ImageGen";
let sharedDialog = null;

export function safeMarkdownHref(rawHref, { repositoryRoot = DEFAULT_REPOSITORY_ROOT, basePath = "", entryDate = "", preserveRelative = false } = {}) {
  const value = String(rawHref || "").trim();
  if (!value) return "";
  if (/^https:\/\//i.test(value)) return value;
  if (value.startsWith("#")) return value;
  const normalized = value.replaceAll("\\", "/");
  if (preserveRelative && (normalized.startsWith("../") || normalized.startsWith("./") || normalized.endsWith(".md"))) return normalized;
  if (normalized.startsWith("../")) return `${repositoryRoot}/blob/main/${normalized.replace(/^\.\.\//, "")}`;
  if (normalized.startsWith("./")) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${normalized.slice(2)}`;
  if (normalized.endsWith(".md")) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${normalized}`;
  if (entryDate) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${entryDate}.md`;
  return "";
}

function headingSlug(value, used) {
  const base = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[`*_~]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "") || "section";
  let slug = base;
  let suffix = 2;
  while (used.has(slug)) slug = `${base}-${suffix++}`;
  used.add(slug);
  return slug;
}

function markdownLink(parent, label, rawHref, options = {}) {
  const href = safeMarkdownHref(rawHref, options);
  if (!href) {
    parent.append(document.createTextNode(label));
    return;
  }
  const link = document.createElement("a");
  link.href = href;
  link.textContent = label;
  link.dataset.markdownHref = String(rawHref || "");
  if (href.startsWith("#")) {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const target = parent.closest("[data-markdown-reader-content], .shared-markdown-content")?.querySelector(href);
      target?.scrollIntoView?.({ block: "start", behavior: "smooth" });
      options.onAnchor?.(href.slice(1));
    });
  } else if (typeof options.onNavigate === "function" && !/^https:\/\//i.test(rawHref)) {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      options.onNavigate(String(rawHref || ""));
    });
  } else {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  parent.append(link);
}

function appendInline(parent, value, options = {}) {
  const text = String(value || "");
  const tokenPattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|\*[^*]+\*)/g;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) parent.append(document.createTextNode(text.slice(cursor, index)));
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("**") && token.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else if (token.startsWith("[") && token.includes("](")) {
      const split = token.indexOf("](");
      markdownLink(parent, token.slice(1, split), token.slice(split + 2, -1), options);
    } else if (token.startsWith("*") && token.endsWith("*")) {
      const emphasis = document.createElement("em");
      emphasis.textContent = token.slice(1, -1);
      parent.append(emphasis);
    } else {
      parent.append(document.createTextNode(token));
    }
    cursor = index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function appendParagraph(container, lines, options) {
  if (!lines.length) return;
  const paragraph = document.createElement("p");
  appendInline(paragraph, lines.join(" ").trim(), options);
  container.append(paragraph);
  lines.length = 0;
}

function appendStandaloneImage(container, rawLine, options = {}) {
  const match = rawLine.match(/^\s*!\[([^\]]*)\]\(([^)\s]+)(?:\s+["']([^"']*)["'])?\)\s*$/);
  if (!match) return false;
  const media = componentCapability("content.media");
  if (!media) return false;
  const rawSrc = match[2];
  const resolved = typeof options.resolveMedia === "function" ? options.resolveMedia(rawSrc) : rawSrc;
  const node = media.renderImage({
    src: resolved,
    alt: match[1],
    caption: match[3] || "",
    allowExternal: options.allowExternalMedia === true,
    maxHeight: options.mediaMaxHeight || 640,
    height: options.mediaHeight || 320,
  });
  node.classList.add("shared-markdown-media");
  container.append(node);
  return true;
}

export function renderMarkdown(container, markdown, options = {}) {
  if (!container) return;
  container.replaceChildren();
  container.classList.add("shared-markdown-content");
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  const paragraph = [];
  let codeBlock = null;
  let list = null;
  let listType = "";
  let skippedFirstHeading = false;
  const usedHeadingIds = new Set();
  const closeList = () => { list = null; listType = ""; };

  for (const rawLine of lines) {
    if (rawLine.trim().startsWith("```")) {
      appendParagraph(container, paragraph, options);
      closeList();
      if (codeBlock) {
        container.append(codeBlock);
        codeBlock = null;
      } else {
        const pre = document.createElement("pre");
        pre.append(document.createElement("code"));
        codeBlock = pre;
      }
      continue;
    }
    if (codeBlock) {
      const code = codeBlock.querySelector("code");
      code.textContent += `${code.textContent ? "\n" : ""}${rawLine}`;
      continue;
    }
    if (!rawLine.trim()) {
      appendParagraph(container, paragraph, options);
      closeList();
      continue;
    }
    if (appendStandaloneImage(container, rawLine, options)) {
      appendParagraph(container, paragraph, options);
      closeList();
      continue;
    }
    const headingMatch = rawLine.match(/^\s*(#{1,6})\s+(.+?)\s*$/);
    if (headingMatch) {
      appendParagraph(container, paragraph, options);
      closeList();
      if (!skippedFirstHeading && options.skipFirstHeading !== false) {
        skippedFirstHeading = true;
        continue;
      }
      const cleanHeading = headingMatch[2].replace(/\s+#+\s*$/, "");
      const heading = document.createElement(`h${Math.min(6, headingMatch[1].length)}`);
      heading.id = headingSlug(cleanHeading, usedHeadingIds);
      appendInline(heading, cleanHeading, options);
      container.append(heading);
      continue;
    }
    const quoteMatch = rawLine.match(/^\s*>\s?(.*)$/);
    if (quoteMatch) {
      appendParagraph(container, paragraph, options);
      closeList();
      const quote = document.createElement("blockquote");
      appendInline(quote, quoteMatch[1], options);
      container.append(quote);
      continue;
    }
    if (/^\s*([-*_])\1\1+\s*$/.test(rawLine)) {
      appendParagraph(container, paragraph, options);
      closeList();
      container.append(document.createElement("hr"));
      continue;
    }
    const unorderedMatch = rawLine.match(/^\s*[-*+]\s+(.+)$/);
    const orderedMatch = rawLine.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      appendParagraph(container, paragraph, options);
      const desiredType = orderedMatch ? "ol" : "ul";
      if (!list || listType !== desiredType) {
        closeList();
        list = document.createElement(desiredType);
        listType = desiredType;
        container.append(list);
      }
      const item = document.createElement("li");
      appendInline(item, (orderedMatch || unorderedMatch)[1], options);
      list.append(item);
      continue;
    }
    paragraph.push(rawLine.trim());
  }
  appendParagraph(container, paragraph, options);
  if (codeBlock) container.append(codeBlock);
  const initialAnchor = String(options.initialAnchor || "").replace(/^#/, "");
  if (initialAnchor) queueMicrotask(() => container.querySelector(`#${CSS.escape(initialAnchor)}`)?.scrollIntoView?.({ block: "start" }));
}

function ensureSharedDialog() {
  if (sharedDialog?.isConnected) return sharedDialog;
  const dialog = document.createElement("dialog");
  dialog.className = "component-markdown-reader";
  dialog.innerHTML = `
    <div class="component-markdown-reader__shell">
      <header class="component-markdown-reader__header">
        <div><small>Markdown document</small><h3 data-markdown-reader-title>Document</h3></div>
        <div class="component-markdown-reader__actions">
          <a class="ui-action-button ui-icon-control" data-markdown-reader-source target="_blank" rel="noopener noreferrer" aria-label="Open Markdown source" title="Open Markdown source"><span class="ui-icon" data-icon="external-link" aria-hidden="true"></span></a>
          <button class="ui-action-button ui-icon-control" type="button" data-markdown-reader-close aria-label="Close Markdown reader" title="Close Markdown reader"><span class="ui-icon" data-icon="close" aria-hidden="true"></span></button>
        </div>
      </header>
      <article class="shared-markdown-content" data-markdown-reader-content tabindex="0"></article>
    </div>`;
  dialog.querySelector("[data-markdown-reader-close]")?.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  document.body.append(dialog);
  sharedDialog = dialog;
  return dialog;
}

export async function openMarkdownDocument({ title = "Document", markdown = null, href = "", sourceHref = href, loader = null, options = {} } = {}) {
  const dialog = ensureSharedDialog();
  const titleNode = dialog.querySelector("[data-markdown-reader-title]");
  const content = dialog.querySelector("[data-markdown-reader-content]");
  const source = dialog.querySelector("[data-markdown-reader-source]");
  titleNode.textContent = String(title || "Document");
  content.textContent = "Loading document…";
  source.hidden = !sourceHref;
  if (sourceHref) source.href = String(sourceHref);
  if (!dialog.open) dialog.showModal();
  try {
    let loaded = markdown;
    if (loaded == null && typeof loader === "function") loaded = await loader();
    if (loaded == null && href) {
      const response = await fetch(href, { headers: { Accept: "text/markdown,text/plain;q=0.9" } });
      if (!response.ok) throw new Error(`Markdown request failed (${response.status}).`);
      loaded = await response.text();
    }
    if (loaded && typeof loaded === "object" && !Array.isArray(loaded)) {
      if (loaded.title) titleNode.textContent = String(loaded.title);
      if (loaded.sourceHref !== undefined) {
        source.hidden = !loaded.sourceHref;
        if (loaded.sourceHref) source.href = String(loaded.sourceHref);
      }
      options = { ...options, ...(loaded.options || {}) };
      loaded = loaded.markdown ?? "";
    }
    renderMarkdown(content, loaded || "", options);
    content.focus({ preventScroll: true });
  } catch (error) {
    content.textContent = `Unable to load Markdown document: ${error?.message || error}`;
  }
  return dialog;
}

export const CONTENT_MARKDOWN_CAPABILITY = registerComponentCapability("content.markdown", {
  version: 1,
  description: "Safe shared Markdown rendering and document browsing for workspace components.",
  safeHref: safeMarkdownHref,
  render: renderMarkdown,
  openDocument: openMarkdownDocument,
});
