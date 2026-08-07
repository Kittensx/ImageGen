import { loadFragments } from "./fragments.js";

const REGION_BUILDER_SCRIPTS = [
  "/assets/js/region-builder/state.js",
  "/assets/js/region-builder/utils.js",
  "/assets/js/region-builder/features/canvas-core.js",
  "/assets/js/region-builder/features/paint-history.js",
  "/assets/js/region-builder/features/fill-tool.js",
  "/assets/js/region-builder/features/canvas-view.js",
  "/assets/js/region-builder/features/regions.js",
  "/assets/js/region-builder/features/editor.js",
  "/assets/js/region-builder/features/render.js",
  "/assets/js/region-builder/features/presets.js",
  "/assets/js/region-builder/features/prompt-build.js",
  "/assets/js/region-builder/features/prompt-parser.js",
  "/assets/js/region-builder/features/prompt-import.js",
  "/assets/js/region-builder/features/guides.js",
  "/assets/js/region-builder/features/host.js",
  "/assets/js/region-builder/features/drag.js",
  "/assets/js/region-builder/features/host-events.js",
  "/assets/js/region-builder/features/input-events.js",
  "/assets/js/region-builder/init.js",
];

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.dataset.regionBuilderRuntime = "true";
    script.addEventListener("load", resolve, { once: true });
    script.addEventListener("error", () => reject(new Error(`Unable to load ${src}`)), { once: true });
    document.body.append(script);
  });
}

async function bootRegionBuilder() {
  await loadFragments();
  for (const src of REGION_BUILDER_SCRIPTS) await loadScript(src);
}

bootRegionBuilder().catch((error) => {
  console.error("Region Builder startup failed:", error);
  const toast = document.getElementById("toast");
  if (toast) {
    toast.textContent = "Region Builder startup failed. Check the browser console.";
    toast.classList.add("show");
  }
});
