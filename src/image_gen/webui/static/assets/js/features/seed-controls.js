import { $ } from "../utils.js";

const MAX_SEED = 2147483647;
const RANGE_PATTERN = /^(?:-1\s*,\s*)?\[\s*(\d+)\s*,\s*(\d+)\s*\]$/;
const FIXED_SEED_PATTERN = /^\d+$/;

let bound = false;
let syncing = false;

function integerValue(input, fallback) {
  const parsed = Number.parseInt(String(input?.value ?? "").trim(), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.min(MAX_SEED, parsed));
}

function normalizedBounds(minimum, maximum) {
  const low = Math.max(0, Math.min(MAX_SEED, Math.trunc(Number(minimum) || 0)));
  const high = Math.max(0, Math.min(MAX_SEED, Math.trunc(Number(maximum) || 0)));
  return low <= high ? [low, high] : [high, low];
}

export function parseSeedRangeText(value) {
  const text = String(value ?? "").trim();
  const match = text.match(RANGE_PATTERN);
  if (!match) return null;
  const [minimum, maximum] = normalizedBounds(Number(match[1]), Number(match[2]));
  return { minimum, maximum };
}

export function looksLikeAdvancedSeedText(value) {
  const text = String(value ?? "").trim();
  return Boolean(text) && (text.includes("[") || text.includes("]") || text.includes(","));
}

export function formatSeedRange(minimum, maximum) {
  const [low, high] = normalizedBounds(minimum, maximum);
  return `-1, [${low},${high}]`;
}

function advancedEnabled() {
  return String($("#seedAdvancedEnabled")?.value || "false") === "true";
}

function setAdvancedEnabled(enabled, { open = false } = {}) {
  const active = Boolean(enabled);
  const state = $("#seedAdvancedEnabled");
  const section = $("#batchSeedStrategy");
  const toggle = $("#seedAdvancedToggle");
  if (state) state.value = active ? "true" : "false";
  if (section) {
    section.hidden = !active;
    if (active && open) section.open = true;
  }
  if (toggle) {
    toggle.setAttribute("aria-pressed", active ? "true" : "false");
    toggle.textContent = active ? "Hide advanced" : "Advanced seed";
  }
}

function setStrategyVisibility() {
  const mode = $("#batchSeedMode")?.value || "random_range";
  const rangeFields = $("#batchSeedRangeFields");
  const noDuplicates = $("#seedNoDuplicatesRow");
  const status = $("#seedStrategyStatus");
  const showRange = mode === "random_range";
  if (rangeFields) rangeFields.hidden = !showRange;
  if (noDuplicates) noDuplicates.hidden = !showRange;
  if (status) {
    if (mode === "random_range") {
      status.textContent = "The Seed field and range controls stay synchronized. Range seeds are sampled without replacement until exhausted.";
    } else if (mode === "random") {
      status.textContent = "Random uses the full supported seed space and keeps the Seed field at -1.";
    } else {
      status.textContent = "Sequential starts from the fixed Seed value and increments for each image.";
    }
  }
}

function currentBounds() {
  return normalizedBounds(
    integerValue($("#seedRangeMin"), 0),
    integerValue($("#seedRangeMax"), MAX_SEED),
  );
}

function setBounds(minimum, maximum) {
  const [low, high] = normalizedBounds(minimum, maximum);
  if ($("#seedRangeMin")) $("#seedRangeMin").value = String(low);
  if ($("#seedRangeMax")) $("#seedRangeMax").value = String(high);
  return [low, high];
}

function fixedSeedFromText(value) {
  const text = String(value ?? "").trim();
  if (!FIXED_SEED_PATTERN.test(text)) return null;
  const parsed = Number.parseInt(text, 10);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > MAX_SEED) return null;
  return parsed;
}

function syncStrategyFromSeed({ autoActivate = true, forceOpen = false } = {}) {
  if (syncing) return;
  const seedInput = $("#seed");
  const mode = $("#batchSeedMode");
  if (!seedInput || !mode) return;
  const raw = String(seedInput.value || "").trim();
  const parsedRange = parseSeedRangeText(raw);
  const advancedText = looksLikeAdvancedSeedText(raw);

  syncing = true;
  try {
    if (parsedRange) {
      if (autoActivate) setAdvancedEnabled(true, { open: true });
      setBounds(parsedRange.minimum, parsedRange.maximum);
      mode.value = "random_range";
      setStrategyVisibility();
      return;
    }

    if (advancedText) {
      if (autoActivate) setAdvancedEnabled(true, { open: true });
      const status = $("#seedStrategyStatus");
      if (status) status.textContent = "Finish the Seed range using a form such as -1, [5000,15000].";
      return;
    }

    if (!advancedEnabled() && !forceOpen) return;
    if (raw === "-1" || raw === "") {
      mode.value = "random";
    } else if (fixedSeedFromText(raw) !== null) {
      mode.value = "sequential";
    }
    setStrategyVisibility();
  } finally {
    syncing = false;
  }
}

function syncSeedFromStrategy() {
  if (syncing) return;
  const seedInput = $("#seed");
  const mode = $("#batchSeedMode");
  if (!seedInput || !mode) return;

  syncing = true;
  try {
    setAdvancedEnabled(true);
    if (mode.value === "random_range") {
      const [low, high] = setBounds(...currentBounds());
      seedInput.value = formatSeedRange(low, high);
    } else if (mode.value === "random") {
      seedInput.value = "-1";
    } else {
      const fixed = fixedSeedFromText(seedInput.value);
      if (fixed === null) {
        const [low] = currentBounds();
        seedInput.value = String(low);
      }
    }
    setStrategyVisibility();
  } finally {
    syncing = false;
  }
}

function activateAdvancedSeed() {
  const raw = String($("#seed")?.value || "").trim();
  const parsedRange = parseSeedRangeText(raw);
  setAdvancedEnabled(true, { open: true });
  if (parsedRange) {
    setBounds(parsedRange.minimum, parsedRange.maximum);
    $("#batchSeedMode").value = "random_range";
  } else if (fixedSeedFromText(raw) !== null) {
    $("#batchSeedMode").value = "sequential";
  } else {
    // Advanced mode prefers bounded randomization. This makes the extra
    // capability visible without changing normal Seed=-1 behavior until the
    // user explicitly opens Advanced Seed.
    $("#batchSeedMode").value = "random_range";
    syncSeedFromStrategy();
  }
  setStrategyVisibility();
}

export function collectSeedValues() {
  const rawSeed = String($("#seed")?.value || "").trim();
  const parsedRange = parseSeedRangeText(rawSeed);
  const [rangeLow, rangeHigh] = parsedRange
    ? [parsedRange.minimum, parsedRange.maximum]
    : currentBounds();
  const mode = advancedEnabled() ? ($("#batchSeedMode")?.value || "random_range") : null;

  if (parsedRange) {
    return {
      seed: formatSeedRange(rangeLow, rangeHigh),
      batch_seed_mode: "random_range",
      seed_range_min: rangeLow,
      seed_range_max: rangeHigh,
      seed_no_duplicates: Boolean($("#seedNoDuplicates")?.checked),
    };
  }

  if (mode === "random_range") {
    return {
      seed: formatSeedRange(rangeLow, rangeHigh),
      batch_seed_mode: "random_range",
      seed_range_min: rangeLow,
      seed_range_max: rangeHigh,
      seed_no_duplicates: Boolean($("#seedNoDuplicates")?.checked),
    };
  }
  if (mode === "random") {
    return {
      seed: "-1",
      batch_seed_mode: "random",
      seed_range_min: 0,
      seed_range_max: MAX_SEED,
      seed_no_duplicates: false,
    };
  }
  if (mode === "sequential") {
    const fixed = fixedSeedFromText(rawSeed);
    return {
      seed: String(fixed ?? rangeLow),
      batch_seed_mode: "sequential",
      seed_range_min: rangeLow,
      seed_range_max: rangeHigh,
      seed_no_duplicates: false,
    };
  }

  // Normal mode leaves strategy unspecified. The backend retains its existing
  // contract: -1 means random and a fixed non-negative seed means sequential.
  return {
    seed: rawSeed,
    batch_seed_mode: null,
    seed_range_min: rangeLow,
    seed_range_max: rangeHigh,
    seed_no_duplicates: Boolean($("#seedNoDuplicates")?.checked),
  };
}

export function syncSeedControlsFromGenerationValues(values = {}) {
  const seedInput = $("#seed");
  const mode = $("#batchSeedMode");
  if (!seedInput || !mode) return;

  const rawSeed = String(values.seed ?? seedInput.value ?? "").trim();
  const parsedRange = parseSeedRangeText(rawSeed);
  const requestedMode = String(values.batch_seed_mode || "").trim();
  if (values.seed_range_min !== undefined || values.seed_range_max !== undefined) {
    setBounds(
      values.seed_range_min ?? integerValue($("#seedRangeMin"), 0),
      values.seed_range_max ?? integerValue($("#seedRangeMax"), MAX_SEED),
    );
  }
  if ($("#seedNoDuplicates")) $("#seedNoDuplicates").checked = values.seed_no_duplicates !== false;

  if (parsedRange || requestedMode === "random_range") {
    const [low, high] = parsedRange
      ? setBounds(parsedRange.minimum, parsedRange.maximum)
      : currentBounds();
    mode.value = "random_range";
    seedInput.value = formatSeedRange(low, high);
    setAdvancedEnabled(true, { open: true });
  } else {
    setAdvancedEnabled(false);
    if (requestedMode === "random") mode.value = "random";
    else if (requestedMode === "sequential") mode.value = "sequential";
    else mode.value = "random_range";
  }
  setStrategyVisibility();
}

export function bindSeedControls({ onChange = null } = {}) {
  if (bound) return;
  bound = true;
  const seedInput = $("#seed");
  const toggle = $("#seedAdvancedToggle");
  const mode = $("#batchSeedMode");
  const rangeMin = $("#seedRangeMin");
  const rangeMax = $("#seedRangeMax");
  const noDuplicates = $("#seedNoDuplicates");
  const section = $("#batchSeedStrategy");
  if (!seedInput || !toggle || !mode || !section) return;

  setAdvancedEnabled(false);
  mode.value = "random_range";
  setStrategyVisibility();

  seedInput.addEventListener("input", () => syncStrategyFromSeed({ autoActivate: true }));
  seedInput.addEventListener("change", () => syncStrategyFromSeed({ autoActivate: true }));

  toggle.addEventListener("click", () => {
    if (advancedEnabled()) {
      setAdvancedEnabled(false);
    } else {
      activateAdvancedSeed();
    }
    if (typeof onChange === "function") onChange();
  });

  mode.addEventListener("change", () => {
    syncSeedFromStrategy();
    if (typeof onChange === "function") onChange();
  });

  [rangeMin, rangeMax].filter(Boolean).forEach((input) => {
    input.addEventListener("input", () => {
      if (mode.value === "random_range" && input.value !== "") syncSeedFromStrategy();
    });
    input.addEventListener("change", () => {
      if (mode.value === "random_range") syncSeedFromStrategy();
      if (typeof onChange === "function") onChange();
    });
  });

  noDuplicates?.addEventListener("change", () => {
    if (typeof onChange === "function") onChange();
  });
}
