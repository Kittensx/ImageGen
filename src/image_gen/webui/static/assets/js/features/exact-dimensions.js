const EXACT_DIMENSION_INPUT_IDS = ["width", "height"];

function normalizeExactDimensionInput(input) {
  if (!input) return;
  input.setAttribute("step", "1");
  input.step = "1";
  input.dataset.exactPixelStep = "true";
  input.setCustomValidity("");
}

function dispatchDimensionUpdate(input) {
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function bindDimensionSwap(root = document) {
  const button = root.getElementById?.("swapDimensionsButton")
    || root.querySelector?.("#swapDimensionsButton");
  const width = root.getElementById?.("width") || root.querySelector?.("#width");
  const height = root.getElementById?.("height") || root.querySelector?.("#height");
  if (!button || !width || !height || button.dataset.dimensionSwapBound === "true") return;

  button.dataset.dimensionSwapBound = "true";
  button.addEventListener("click", () => {
    const previousWidth = width.value;
    width.value = height.value;
    height.value = previousWidth;

    normalizeExactDimensionInput(width);
    normalizeExactDimensionInput(height);
    dispatchDimensionUpdate(width);
    dispatchDimensionUpdate(height);
  });
}

export function enforceExactDimensionInputs(root = document) {
  EXACT_DIMENSION_INPUT_IDS.forEach((id) => {
    const input = root.getElementById?.(id) || root.querySelector?.(`#${id}`);
    normalizeExactDimensionInput(input);
  });
  bindDimensionSwap(root);
}
