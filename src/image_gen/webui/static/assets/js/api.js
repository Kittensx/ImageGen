async function request(path, options = {}) {
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const headers = isFormData
    ? { ...(options.headers || {}) }
    : { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let code = "";
    let detail = null;
    try {
      const payload = await response.json();
      detail = payload.detail ?? null;
      if (detail && typeof detail === "object") {
        code = String(detail.code || "");
        message = String(detail.message || message);
      } else if (detail) {
        message = String(detail);
      }
    } catch {
      // Keep the HTTP status text.
    }
    const error = new Error(message);
    error.code = code;
    error.detail = detail;
    error.status = response.status;
    if (code === "civitai_credentials_required") {
      window.dispatchEvent(new CustomEvent("image-gen-civitai-credential-required", {
        detail: { message },
      }));
    }
    throw error;
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
  profile: () => request("/api/profile"),
  updateProfileSharing: (values = {}) => request("/api/profile/sharing", {
    method: "PATCH",
    body: JSON.stringify(values || {}),
  }),
  discordCommunityStatus: () => request("/api/profile/discord/community"),
  connectDiscordProfile: () => request("/api/profile/discord/connect", { method: "POST" }),
  refreshDiscordPresence: () => request("/api/profile/discord/presence", { method: "POST" }),
  disconnectDiscordProfile: () => request("/api/profile/discord/disconnect", { method: "POST" }),
  bugReports: () => request("/api/bug-reports"),
  syncBugReports: () => request("/api/bug-reports/sync", { method: "POST" }),
  prepareBugIssue: (fingerprint) => request(`/api/bug-reports/${encodeURIComponent(fingerprint)}/issue`, { method: "POST" }),
  revealBugBundle: (fingerprint) => request(`/api/bug-reports/${encodeURIComponent(fingerprint)}/reveal`, { method: "POST" }),
  bugBundleUrl: (fingerprint) => `/api/bug-reports/${encodeURIComponent(fingerprint)}/bundle`,
  refreshModels: () => request("/api/models/refresh", { method: "POST" }),
  upscalers: () => request("/api/upscalers"),
  hiresDimensionPlan: (values = {}) => request("/api/hires/dimension-plan", {
    method: "POST",
    body: JSON.stringify(values || {}),
  }),
  uploadOutpaintPrototypeSource: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/outpaint/prototype/source", { method: "POST", body: form });
  },
  outpaintPrototypePlan: (values = {}) => request("/api/outpaint/prototype/plan", {
    method: "POST",
    body: JSON.stringify(values || {}),
  }),
  refreshUpscalers: (mode = "all", selectedFile = "") => request("/api/upscalers/refresh", {
    method: "POST",
    body: JSON.stringify({ mode, ...(selectedFile ? { selected_file: selectedFile } : {}) }),
  }),
  modelRuntimeStatus: () => request("/api/models/runtime-status"),
  activateModel: (modelPath) => request("/api/models/activate", {
    method: "POST",
    body: JSON.stringify({ model_path: modelPath }),
  }),
  activeModel: () => request("/api/models/active"),
  unloadModel: () => request("/api/models/unload", { method: "POST" }),
  assetCatalog: () => request("/api/assets/catalog"),
  civitaiConnectionStatus: () => request("/api/integrations/civitai"),
  saveCivitaiCredential: (apiKey) => request("/api/integrations/civitai/credential", {
    method: "PUT",
    body: JSON.stringify({ api_key: String(apiKey || "") }),
  }),
  removeCivitaiCredential: () => request("/api/integrations/civitai/credential", { method: "DELETE" }),
  testCivitaiConnection: () => request("/api/integrations/civitai/test", { method: "POST" }),
  enrichAssetsFromCivitai: (assetType, mode = "missing") => request(`/api/civitai/assets/${encodeURIComponent(assetType)}/metadata`, {
    method: "POST",
    body: JSON.stringify({ mode }),
  }),
  enrichAssetFromCivitai: (assetType, assetId, overwrite = false) => request(`/api/civitai/assets/${encodeURIComponent(assetType)}/${encodeURIComponent(assetId)}/metadata`, {
    method: "POST",
    body: JSON.stringify({ overwrite }),
  }),
  refreshAssetCatalog: (assetType = "") => request("/api/assets/refresh", {
    method: "POST",
    body: JSON.stringify(assetType ? { asset_type: assetType } : {}),
  }),
  checkpointAssets: () => request("/api/assets/checkpoints"),
  refreshCheckpointAssets: () => request("/api/assets/checkpoints/refresh", { method: "POST" }),
  checkpointDetails: (assetId) => request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}`),
  saveCheckpointMetadata: (assetId, values) => request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}`, {
    method: "PATCH",
    body: JSON.stringify(values),
  }),
  replaceCheckpointPreview: (assetId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}/preview`, {
      method: "POST",
      body: form,
    });
  },
  loadCheckpointPreviewCandidates: (assetId, limit = 48) => request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}/preview/recent-outputs?limit=${encodeURIComponent(limit)}`),
  replaceCheckpointPreviewFromOutput: (assetId, outputId) => request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}/preview/from-output`, {
    method: "POST",
    body: JSON.stringify({ output_id: outputId }),
  }),
  openCheckpointFolder: (assetId) => request(`/api/assets/checkpoints/${encodeURIComponent(assetId)}/open-folder`, { method: "POST" }),
  vaeAssets: () => request("/api/assets/vaes"),
  refreshVaeAssets: () => request("/api/assets/vaes/refresh", { method: "POST" }),
  vaeDetails: (assetId) => request(`/api/assets/vaes/${encodeURIComponent(assetId)}`),
  saveVaeMetadata: (assetId, values) => request(`/api/assets/vaes/${encodeURIComponent(assetId)}`, {
    method: "PATCH",
    body: JSON.stringify(values),
  }),
  loraAssets: () => request("/api/assets/loras"),
  refreshLoraAssets: () => request("/api/assets/loras/refresh", { method: "POST" }),
  scanLoras: (mode = "missing") => request("/api/assets/loras/scan", {
    method: "POST",
    body: JSON.stringify({ mode }),
  }),
  enrichLorasFromCivitai: (mode = "missing") => api.enrichAssetsFromCivitai("lora", mode),
  enrichLoraFromCivitai: (assetId, overwrite = false) => api.enrichAssetFromCivitai("lora", assetId, overwrite),
  loraDetails: (assetId) => request(`/api/assets/loras/${encodeURIComponent(assetId)}`),
  saveLoraMetadata: (assetId, values) => request(`/api/assets/loras/${encodeURIComponent(assetId)}`, {
    method: "PATCH",
    body: JSON.stringify(values),
  }),
  replaceLoraPreview: (assetId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/assets/loras/${encodeURIComponent(assetId)}/preview`, {
      method: "POST",
      body: form,
    });
  },
  loadLoraPreviewCandidates: (assetId, limit = 48) => request(`/api/assets/loras/${encodeURIComponent(assetId)}/preview/recent-outputs?limit=${encodeURIComponent(limit)}`),
  replaceLoraPreviewFromOutput: (assetId, outputId) => request(`/api/assets/loras/${encodeURIComponent(assetId)}/preview/from-output`, {
    method: "POST",
    body: JSON.stringify({ output_id: outputId }),
  }),
  openLoraFolder: (assetId) => request(`/api/assets/loras/${encodeURIComponent(assetId)}/open-folder`, { method: "POST" }),
  deleteLora: (assetId) => request(`/api/assets/loras/${encodeURIComponent(assetId)}`, { method: "DELETE" }),
  textualInversionAssets: () => request("/api/assets/textual-inversions"),
  refreshTextualInversionAssets: () => request("/api/assets/textual-inversions/refresh", { method: "POST" }),
  textualInversionDetails: (assetId) => request(`/api/assets/textual-inversions/${encodeURIComponent(assetId)}`),
  saveTextualInversionMetadata: (assetId, values) => request(`/api/assets/textual-inversions/${encodeURIComponent(assetId)}`, {
    method: "PATCH",
    body: JSON.stringify(values),
  }),
  replaceTextualInversionPreview: (assetId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/assets/textual-inversions/${encodeURIComponent(assetId)}/preview`, {
      method: "POST",
      body: form,
    });
  },
  openTextualInversionFolder: (assetId) => request(`/api/assets/textual-inversions/${encodeURIComponent(assetId)}/open-folder`, { method: "POST" }),
  defaultAssets: () => request("/api/default-assets"),
  saveDefaultAssets: (values) => request("/api/default-assets", {
    method: "PUT",
    body: JSON.stringify(values),
  }),
  patchDefaultAssets: (values) => request("/api/default-assets", {
    method: "PATCH",
    body: JSON.stringify(values),
  }),
  workspaceLayout: () => request("/api/workspace/layout"),
  saveWorkspaceLayout: (layout) => request("/api/workspace/layout", {
    method: "PATCH",
    body: JSON.stringify({ layout }),
  }),
  resetWorkspaceLayout: () => request("/api/workspace/layout/reset", { method: "POST" }),
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
  pauseQueueAfterCurrent: (jobId = "") => request("/api/queue/pause-after-current", {
    method: "POST",
    body: JSON.stringify(jobId ? { job_id: jobId } : {}),
  }),
  resumeQueue: () => request("/api/queue/resume", { method: "POST" }),
  skipJobImage: (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}/skip`, {
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
