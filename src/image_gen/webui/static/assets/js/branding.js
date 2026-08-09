const DEFAULT_PRODUCT_LABEL = "Application";

let current = Object.freeze({
  name: DEFAULT_PRODUCT_LABEL,
  branding: Object.freeze({ mark_asset: "", wordmark_asset: "" }),
});

function textTemplate(value, name = current.name) {
  return String(value || "").replaceAll("{product}", name);
}

function assetCssValue(path) {
  const value = String(path || "").trim();
  if (!value) return "none";
  return `url(${JSON.stringify(value)})`;
}

function applyBrandAssets(root, branding) {
  root.querySelectorAll?.("[data-product-brand-asset]").forEach((node) => {
    const kind = String(node.dataset.productBrandAsset || "").trim().toLowerCase();
    const src = kind === "wordmark" ? branding.wordmark_asset : kind === "mark" ? branding.mark_asset : "";
    if (!src) return;
    if (node instanceof HTMLImageElement) node.src = src;
    else node.style.backgroundImage = assetCssValue(src);
  });
}

function applyTemplateAttributes(root, name) {
  root.querySelectorAll?.("[data-product-name]").forEach((node) => {
    // data-product-name is a content-target marker. Never allow a document
    // container to become one, because assigning textContent to <html> or
    // <body> destroys the mounted application DOM.
    if (node === document.documentElement || node === document.body) return;
    node.textContent = name;
  });
  root.querySelectorAll?.("[data-product-name-template]").forEach((node) => {
    node.textContent = textTemplate(node.dataset.productNameTemplate, name);
  });
  root.querySelectorAll?.("[data-product-aria-template]").forEach((node) => {
    node.setAttribute("aria-label", textTemplate(node.dataset.productAriaTemplate, name));
  });
  root.querySelectorAll?.("[data-product-title-template]").forEach((node) => {
    node.setAttribute("title", textTemplate(node.dataset.productTitleTemplate, name));
  });
  root.querySelectorAll?.("[data-product-alt-template]").forEach((node) => {
    node.setAttribute("alt", textTemplate(node.dataset.productAltTemplate, name));
  });
}

export function configureBranding(application = {}, root = document) {
  const name = String(application?.name || "").trim() || DEFAULT_PRODUCT_LABEL;
  const branding = Object.freeze({
    mark_asset: String(application?.branding?.mark_asset || "").trim(),
    wordmark_asset: String(application?.branding?.wordmark_asset || "").trim(),
  });
  current = Object.freeze({ name, branding });

  window.imageGenProductName = name;
  window.imageGenBranding = Object.freeze({ name, branding: { ...branding } });
  document.documentElement.dataset.productLabel = name;
  document.documentElement.style.setProperty("--product-brand-mark-image", assetCssValue(branding.mark_asset));
  document.documentElement.style.setProperty("--product-brand-wordmark-image", assetCssValue(branding.wordmark_asset));
  document.title = `${name} WebUI`;
  applyTemplateAttributes(root, name);
  applyBrandAssets(root, branding);

  window.dispatchEvent(new CustomEvent("image-gen-branding-ready", {
    detail: { name, branding: { ...branding } },
  }));
  return current;
}

export function productName() {
  return current.name;
}

export function productText(template) {
  return textTemplate(template, current.name);
}

export function productBranding() {
  return { ...current.branding };
}

export function applyProductBranding(root = document) {
  applyTemplateAttributes(root, current.name);
  applyBrandAssets(root, current.branding);
}
