export const HIRES_DIMENSION_PLAN_VERSION = "phase14n12-dimension-plan-v2";
export const HIRES_DIMENSION_MULTIPLE = 8;
export const HIRES_SCALE_TOLERANCE = 1e-6;

export function clampHiresDimension(value, fallback = 512) {
  const parsed = Number(value);
  const fallbackNumber = Number(fallback);
  const candidate = Number.isFinite(parsed) && parsed > 0
    ? parsed
    : (Number.isFinite(fallbackNumber) && fallbackNumber > 0 ? fallbackNumber : 512);
  return Math.max(64, Math.min(16384, Math.round(candidate)));
}

export function normalizeBaseHiresDimension(value, multiple = HIRES_DIMENSION_MULTIPLE) {
  const dimension = clampHiresDimension(value);
  const alignment = Math.max(1, Math.round(Number(multiple) || HIRES_DIMENSION_MULTIPLE));
  if (alignment <= 1) return dimension;
  return Math.max(alignment, Math.round(dimension / alignment) * alignment);
}

export function alignHiresDimension(value, multiple = HIRES_DIMENSION_MULTIPLE) {
  const dimension = clampHiresDimension(value);
  const alignment = Math.max(1, Math.round(Number(multiple) || HIRES_DIMENSION_MULTIPLE));
  if (alignment <= 1) return dimension;
  return Math.min(16384, Math.max(alignment, Math.ceil(dimension / alignment) * alignment));
}

export function normalizeHiresSizeMode(value, enabled = true) {
  const mode = String(value || "scale_from_base").trim().toLowerCase();
  if (enabled && mode === "same_as_base") return "scale_from_base";
  return ["same_as_base", "scale_from_base", "explicit_dimensions"].includes(mode)
    ? mode
    : "scale_from_base";
}

export function planHiresDimensions({
  baseWidth,
  baseHeight,
  mode = "scale_from_base",
  scale = 1.5,
  targetWidth = 0,
  targetHeight = 0,
  dimensionMultiple = HIRES_DIMENSION_MULTIPLE,
  baseDimensionMultiple = HIRES_DIMENSION_MULTIPLE,
  enabled = true,
} = {}) {
  const resolvedBaseMultiple = Math.max(1, Math.round(Number(baseDimensionMultiple) || HIRES_DIMENSION_MULTIPLE));
  const resolvedDimensionMultiple = Math.max(1, Math.round(Number(dimensionMultiple) || resolvedBaseMultiple));
  const resolvedBaseWidth = normalizeBaseHiresDimension(baseWidth || 512, resolvedBaseMultiple);
  const resolvedBaseHeight = normalizeBaseHiresDimension(baseHeight || 512, resolvedBaseMultiple);
  const resolvedMode = normalizeHiresSizeMode(mode, enabled);
  const requestedScale = Math.max(1, Math.min(8, Number(scale) || 1.5));

  let requestedWidth = resolvedBaseWidth;
  let requestedHeight = resolvedBaseHeight;
  if (resolvedMode === "scale_from_base") {
    requestedWidth = clampHiresDimension(resolvedBaseWidth * requestedScale, resolvedBaseWidth);
    requestedHeight = clampHiresDimension(resolvedBaseHeight * requestedScale, resolvedBaseHeight);
  } else if (resolvedMode === "explicit_dimensions") {
    requestedWidth = clampHiresDimension(targetWidth, resolvedBaseWidth);
    requestedHeight = clampHiresDimension(targetHeight, resolvedBaseHeight);
  }

  const internalWidth = alignHiresDimension(requestedWidth, resolvedDimensionMultiple);
  const internalHeight = alignHiresDimension(requestedHeight, resolvedDimensionMultiple);
  const axisScaleWidth = Number((requestedWidth / resolvedBaseWidth).toFixed(6));
  const axisScaleHeight = Number((requestedHeight / resolvedBaseHeight).toFixed(6));
  const uniform = Math.abs(axisScaleWidth - axisScaleHeight) <= HIRES_SCALE_TOLERANCE;
  const uniformScale = uniform ? Number(((axisScaleWidth + axisScaleHeight) / 2).toFixed(6)) : null;
  const baseAspect = resolvedBaseWidth / resolvedBaseHeight;
  const targetAspect = requestedWidth / requestedHeight;
  const aspectRatioChanged = Math.abs(baseAspect - targetAspect) > HIRES_SCALE_TOLERANCE;

  return {
    contract_version: HIRES_DIMENSION_PLAN_VERSION,
    mode: resolvedMode,
    base_width: resolvedBaseWidth,
    base_height: resolvedBaseHeight,
    requested_scale: resolvedMode === "scale_from_base" ? requestedScale : uniformScale,
    requested_width: requestedWidth,
    requested_height: requestedHeight,
    internal_width: internalWidth,
    internal_height: internalHeight,
    final_width: requestedWidth,
    final_height: requestedHeight,
    effective_width: internalWidth,
    effective_height: internalHeight,
    effective_scale_x: axisScaleWidth,
    effective_scale_y: axisScaleHeight,
    axis_scale_width: axisScaleWidth,
    axis_scale_height: axisScaleHeight,
    uniform_scale: uniformScale,
    is_uniform_scale: uniform,
    aspect_ratio_changed: aspectRatioChanged,
    alignment_applied: internalWidth !== requestedWidth || internalHeight !== requestedHeight,
    alignment_correction_required: internalWidth !== requestedWidth || internalHeight !== requestedHeight,
    dimension_multiple: resolvedDimensionMultiple,
    base_dimension_multiple: resolvedBaseMultiple,
  };
}
