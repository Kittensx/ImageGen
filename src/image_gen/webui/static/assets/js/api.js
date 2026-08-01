async function request(path, options = {}) {
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const headers = isFormData
    ? { ...(options.headers || {}) }
    : { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }
  return response.json();
}

async function requestDownload(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // Keep HTTP status text.
    }
    throw new Error(message);
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  const warnings = JSON.parse(response.headers.get("X-IMAGE-GEN-Export-Warnings") || "[]");
  return {
    blob: await response.blob(),
    filename: match?.[1] || "image_gen_queue.igqueue.json",
    warnings,
    jobCount: Number(response.headers.get("X-IMAGE-GEN-Job-Count") || 0),
  };
}

function encodeOutputPath(outputId) {
  return String(outputId || "")
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

export const api = {
  bootstrap: () => request("/api/bootstrap"),
  health: () => request("/api/health"),
  refreshModels: () => request("/api/models/refresh", { method: "POST" }),
  modelRuntimeStatus: () => request("/api/models/runtime-status"),
  activateModel: (modelPath) => request("/api/models/activate", {
    method: "POST",
    body: JSON.stringify({ model_path: modelPath }),
  }),
  activeModel: () => request("/api/models/active"),
  reloadWorkspace: () => request("/api/workspace/reload", { method: "POST" }),
  restartBackend: () => request("/api/system/restart", { method: "POST" }),
  recentOutputs: (filters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters || {}).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      params.set(key, String(value));
    });
    const query = params.toString();
    return request(`/api/recent-outputs${query ? `?${query}` : ""}`);
  },
  clearRecentOutputs: () => request("/api/recent-outputs/clear", { method: "POST" }),
  reloadRecentOutputs: () => request("/api/recent-outputs/reload", { method: "POST" }),
  clearJobCache: () => request("/api/maintenance/job-cache/clear", { method: "POST" }),
  dismissTerminalJobs: () => request("/api/maintenance/queue/dismiss-terminal", { method: "POST" }),
  clearQueuedJobs: (payload = {}) => request("/api/maintenance/queue/clear", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  openOutputFolder: (path = "") => request("/api/outputs/open-folder", {
    method: "POST",
    body: JSON.stringify({ path }),
  }),
  outputDetails: (outputId, detailsUrl = "") => detailsUrl ? request(detailsUrl) : request(`/api/outputs/${encodeOutputPath(outputId)}/details`),
  jobPrimaryOutput: (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}/primary-output`),
  replayPreflight: (values) => request("/api/replay/preflight", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  submitReplay: (preflightToken) => request("/api/replay/submit", {
    method: "POST",
    body: JSON.stringify({ preflight_token: preflightToken }),
  }),
  batchReplayPreflight: (values) => request("/api/replay/batch/preflight", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  submitBatchReplay: (preflightToken, queueValidOnly = false) => request("/api/replay/batch/submit", {
    method: "POST",
    body: JSON.stringify({
      preflight_token: preflightToken,
      queue_valid_only: queueValidOnly,
    }),
  }),
  parseBatchImport: (file, { formatHint = "", defaultsPolicy = "file_only", currentValues = {} } = {}) => {
    const form = new FormData();
    form.append("file", file);
    form.append("format_hint", formatHint);
    form.append("defaults_policy", defaultsPolicy);
    form.append("current_values", JSON.stringify(currentValues || {}));
    return request("/api/batch/import/parse", { method: "POST", body: form });
  },
  preflightBatchImport: (values) => request("/api/batch/import/preflight", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  submitBatchImport: (preflightToken, queueValidOnly = false) => request("/api/batch/import/submit", {
    method: "POST",
    body: JSON.stringify({
      preflight_token: preflightToken,
      queue_valid_only: queueValidOnly,
    }),
  }),
  exportBatch: (values) => requestDownload("/api/batch/export", values),
  variationPreflight: (values) => request("/api/variations/preflight", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  submitVariations: (preflightToken) => request("/api/variations/submit", {
    method: "POST",
    body: JSON.stringify({ preflight_token: preflightToken }),
  }),
  exportVariations: (preflightToken, format = "native") => requestDownload("/api/variations/export", {
    preflight_token: preflightToken,
    format,
    filename_stem: "variation_matrix",
  }),
  saveSession: (values) => request("/api/session", {
    method: "PUT",
    body: JSON.stringify(values),
  }),
  runtimeStartupStatus: () => request("/api/runtime/startup-status"),
  runtimeCommand: () => request("/api/runtime/command"),
  inheritRuntimeStartupProfile: () => request("/api/runtime/inherit-startup-profile", {
    method: "POST",
  }),
  saveSettings: (values) => request("/api/settings", {
    method: "PUT",
    body: JSON.stringify(values),
  }),
  jobs: () => request("/api/jobs"),
  validateScheduler: (values) => request("/api/schedulers/validate", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  submitJob: (values) => request("/api/jobs", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  cancelJob: (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  }),
  promptParsers: () => request("/api/prompt-parsers"),
  promptShortcutProfiles: () => request("/api/prompt-shortcut-profiles"),
  validatePromptShortcutProfile: (values) => request("/api/prompt-shortcut-profiles/validate", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  savePromptShortcutProfile: (values) => request("/api/prompt-shortcut-profiles", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  deletePromptShortcutProfile: (profileId) => request(`/api/prompt-shortcut-profiles/${encodeURIComponent(profileId)}`, {
    method: "DELETE",
  }),
  promptParserPresets: () => request("/api/prompt-parser-presets"),
  savePromptParserPreset: (values) => request("/api/prompt-parser-presets", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  deletePromptParserPreset: (presetId) => request(`/api/prompt-parser-presets/${encodeURIComponent(presetId)}`, {
    method: "DELETE",
  }),
  translatePrompts: (values) => request("/api/prompts/translate", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  preflightPrompts: (values) => request("/api/prompts/preflight", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  promptPresets: () => request("/api/prompt-presets"),
  savePromptPreset: (values) => request("/api/prompt-presets", {
    method: "POST",
    body: JSON.stringify(values),
  }),
  deletePromptPreset: (name) => request(`/api/prompt-presets/${encodeURIComponent(name)}`, {
    method: "DELETE",
  }),
  profiles: (kind, pluginId = "") => {
    const query = pluginId ? `?plugin_id=${encodeURIComponent(pluginId)}` : "";
    return request(`/api/profiles/${encodeURIComponent(kind)}${query}`);
  },
  saveProfile: (kind, values) => request(`/api/profiles/${encodeURIComponent(kind)}`, {
    method: "POST",
    body: JSON.stringify(values),
  }),
  deleteProfile: (kind, name, pluginId = "") => {
    const query = pluginId ? `?plugin_id=${encodeURIComponent(pluginId)}` : "";
    return request(`/api/profiles/${encodeURIComponent(kind)}/${encodeURIComponent(name)}${query}`, {
      method: "DELETE",
    });
  },
};
