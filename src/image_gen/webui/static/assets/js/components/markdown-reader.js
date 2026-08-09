const DEFAULT_REPOSITORY_ROOT = "https://github.com/Kittensx/ImageGen";
let sharedDialog = null;

export function safeMarkdownHref(rawHref, { repositoryRoot = DEFAULT_REPOSITORY_ROOT, basePath = "", entryDate = "" } = {}) {
  const value = String(rawHref || "").trim();
  if (!value) return "";
  if (/^https?:\/\//i.test(value)) return value;
  if (value.startsWith("#")) return value;
  const normalized = value.replaceAll("\\", "/");
  if (normalized.startsWith("../")) return `${repositoryRoot}/blob/main/${normalized.replace(/^\.\.\//, "")}`;
  if (normalized.startsWith("./")) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${normalized.slice(2)}`;
  if (normalized.endsWith(".md")) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${normalized}`;
  if (entryDate) return `${repositoryRoot}/blob/main/${basePath ? `${basePath.replace(/\/$/, "")}/` : ""}${entryDate}.md`;
  return "";
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
      const label = token.slice(1, split);
      const href = safeMarkdownHref(token.slice(split + 2, -1), options);
      if (href) {
        const link = document.createElement("a");
        link.href = href;
        link.target = href.startsWith("#") ? "" : "_blank";
        if (link.target) link.rel = "noopener noreferrer";
        link.textContent = label;
        parent.append(link);
      } else {
        parent.append(document.createTextNode(label));
      }
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

export function renderMarkdown(container, markdown, options = {}) {
  if (!container) return;
  container.replaceChildren();
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  const paragraph = [];
  let codeBlock = null;
  let list = null;
  let listType = "";
  let skippedFirstHeading = Boolean(options.keepFirstHeading === false);
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
    const headingMatch = rawLine.match(/^\s*(#{1,6})\s+(.+?)\s*$/);
    if (headingMatch) {
      appendParagraph(container, paragraph, options);
      closeList();
      if (!skippedFirstHeading && options.skipFirstHeading !== false) {
        skippedFirstHeading = true;
        continue;
      }
      const heading = document.createElement(`h${Math.min(6, headingMatch[1].length)}`);
      appendInline(heading, headingMatch[2].replace(/\s+#+\s*$/, ""), options);
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
      <article class="home-markdown-reader" data-markdown-reader-content tabindex="0"></article>
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
    let text = markdown;
    if (text == null && typeof loader === "function") text = await loader();
    if (text == null && href) {
      const response = await fetch(href, { headers: { Accept: "text/markdown,text/plain;q=0.9" } });
      if (!response.ok) throw new Error(`Markdown request failed (${response.status}).`);
      text = await response.text();
    }
    renderMarkdown(content, text || "", options);
    content.focus({ preventScroll: true });
  } catch (error) {
    content.textContent = `Unable to load Markdown document: ${error?.message || error}`;
  }
  return dialog;
}
