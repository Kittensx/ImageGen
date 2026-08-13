export const WORKSPACE_WIDTH_CLASSES = Object.freeze(["wide", "standard", "compact", "narrow"]);
export const WORKSPACE_WIDTH_THRESHOLDS = Object.freeze({
  narrowMax: 719,
  compactMax: 1049,
  standardMax: 1450,
});
export const WORKSPACE_REPRESENTATIVE_WIDTHS = Object.freeze({
  wide: 1600,
  standard: 1280,
  compact: 960,
  narrow: 680,
});

function normalizedWidthClass(value) {
  const token = String(value || "").trim().toLowerCase();
  return WORKSPACE_WIDTH_CLASSES.includes(token) ? token : "wide";
}

function normalizedSpan(value, fallback = 12) {
  const span = Number(value);
  if (!Number.isFinite(span)) return fallback;
  return Math.max(1, Math.min(12, Math.round(span)));
}

export function classifyWorkspaceWidth(width) {
  const value = Math.max(0, Number(width) || 0);
  if (value > WORKSPACE_WIDTH_THRESHOLDS.standardMax) return "wide";
  if (value > WORKSPACE_WIDTH_THRESHOLDS.compactMax) return "standard";
  if (value > WORKSPACE_WIDTH_THRESHOLDS.narrowMax) return "compact";
  return "narrow";
}

export function responsiveGridSpan(baseSpan, widthClass, { minGridSpan = 1, maxGridSpan = 12 } = {}) {
  const width = normalizedWidthClass(widthClass);
  const minimum = normalizedSpan(minGridSpan, 1);
  const maximum = Math.max(minimum, normalizedSpan(maxGridSpan, 12));
  const base = Math.max(minimum, Math.min(maximum, normalizedSpan(baseSpan, 12)));
  let requested = base;
  if (width === "standard") requested = base >= 8 ? 12 : Math.max(6, base);
  if (width === "compact") requested = base >= 7 ? 12 : Math.max(6, base);
  if (width === "narrow") requested = 12;
  return Math.max(minimum, Math.min(maximum, requested));
}

export function responsivePresentationSpan(effectiveSpan, widthClass, shellState = "expanded", { minGridSpan = 1, maxGridSpan = 12 } = {}) {
  const width = normalizedWidthClass(widthClass);
  const minimum = normalizedSpan(minGridSpan, 1);
  const maximum = Math.max(minimum, normalizedSpan(maxGridSpan, 12));
  const base = Math.max(minimum, Math.min(maximum, normalizedSpan(effectiveSpan, 12)));
  if (String(shellState || "expanded").trim().toLowerCase() !== "side") return base;
  const target = width === "wide" ? 2 : (width === "standard" ? 3 : (width === "compact" ? 6 : 12));
  return Math.max(minimum, Math.min(maximum, target));
}

export function resolveResponsiveVariant(descriptor, widthClass, preferredVariant = "", placementOverrides = {}) {
  const width = normalizedWidthClass(widthClass);
  const supported = [...(descriptor?.supportedVariants || [])].map(String);
  const preferred = String(preferredVariant || descriptor?.defaultVariant || "standard");
  const responsive = descriptor?.responsive && typeof descriptor.responsive === "object" ? descriptor.responsive : {};
  const overrides = placementOverrides && typeof placementOverrides === "object" ? placementOverrides : {};
  const desired = width === "wide"
    ? preferred
    : String(overrides[width] || responsive[width] || preferred || "standard");
  if (supported.includes(desired)) return { variant: desired, resolution: "preferred" };
  if (supported.includes("standard")) return { variant: "standard", resolution: "standard_fallback" };
  if (supported.includes(preferred)) return { variant: preferred, resolution: "preferred_fallback" };
  const defaultVariant = String(descriptor?.defaultVariant || "standard");
  if (supported.includes(defaultVariant)) return { variant: defaultVariant, resolution: "default_fallback" };
  return { variant: supported[0] || desired || "standard", resolution: "first_supported_fallback" };
}

export function responsiveWorkspacePlacements(layout, descriptorLookup, widthClass) {
  const source = Array.isArray(layout?.components) ? layout.components : [];
  return source.map((placement, order) => {
    const descriptor = typeof descriptorLookup === "function" ? descriptorLookup(placement.componentId) : null;
    if (!descriptor) return null;
    const baseSpan = normalizedSpan(placement.span ?? placement.position?.columnSpan, descriptor.defaultGridSpan || 12);
    const resolved = resolveResponsiveVariant(descriptor, widthClass, placement.variant, placement.responsive);
    const responsiveSpan = responsiveGridSpan(baseSpan, widthClass, descriptor);
    const effectiveSpan = responsivePresentationSpan(responsiveSpan, widthClass, placement.shellState, descriptor);
    return {
      ...placement,
      order: Number.isFinite(Number(placement.order)) ? Number(placement.order) : order,
      baseSpan,
      responsiveSpan,
      effectiveSpan,
      baseVariant: placement.variant || descriptor.defaultVariant,
      effectiveVariant: resolved.variant,
      variantResolution: resolved.resolution,
      widthClass: normalizedWidthClass(widthClass),
    };
  }).filter(Boolean);
}

function measuredWidth(root) {
  const rect = root?.getBoundingClientRect?.();
  return Math.max(0, Number(rect?.width || root?.clientWidth || 0));
}

export function bindWorkspaceWidthObserver(root, { onChange = null } = {}) {
  if (!root) return null;
  let lastClass = "";
  let lastWidth = -1;
  const apply = (width = measuredWidth(root)) => {
    const rounded = Math.max(0, Math.round(Number(width) || 0));
    const widthClass = classifyWorkspaceWidth(rounded);
    root.dataset.workspaceWidthClass = widthClass;
    root.dataset.workspaceWidth = String(rounded);
    const classChanged = widthClass !== lastClass;
    const previousWidthClass = lastClass || null;
    lastWidth = rounded;
    if (classChanged) {
      lastClass = widthClass;
      const detail = { width: rounded, widthClass, previousWidthClass };
      onChange?.(detail);
      root.dispatchEvent(new CustomEvent("workspace-width-class-change", { bubbles: true, detail }));
    }
    return { width: lastWidth, widthClass };
  };

  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver((entries) => {
        const entry = entries[entries.length - 1];
        const width = entry?.contentRect?.width ?? measuredWidth(root);
        apply(width);
      })
    : null;
  observer?.observe(root);
  apply();
  return Object.freeze({
    refresh: () => apply(),
    disconnect: () => observer?.disconnect(),
  });
}
