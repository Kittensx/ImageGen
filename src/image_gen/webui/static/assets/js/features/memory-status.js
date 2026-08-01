import { $ } from "../utils.js";

function formatBytes(value) {
  if (value === null || value === undefined || value === "") return "Unavailable";
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return "Unavailable";
  if (number < 1024) return `${Math.round(number)} B`;
  if (number < 1024 ** 2) return `${(number / 1024).toFixed(number < 10 * 1024 ? 1 : 0)} KiB`;
  if (number < 1024 ** 3) return `${(number / 1024 ** 2).toFixed(number < 10 * 1024 ** 2 ? 1 : 0)} MiB`;
  return `${(number / 1024 ** 3).toFixed(2)} GiB`;
}

function finiteNonnegative(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function physicalMemoryValues(cuda) {
  const total = finiteNonnegative(cuda.physical_total_vram_bytes ?? cuda.total_vram_bytes);
  const freeValue = finiteNonnegative(cuda.physical_free_vram_bytes ?? cuda.free_vram_bytes);
  const free = total !== null && freeValue !== null ? Math.min(total, freeValue) : null;
  const explicitUsed = finiteNonnegative(cuda.physical_used_vram_bytes);
  const used = total !== null && free !== null
    ? Math.max(0, Math.min(total, explicitUsed ?? (total - free)))
    : null;
  return { total, free, used };
}

function listText(values, fallback) {
  return Array.isArray(values) && values.length ? values.join(", ") : fallback;
}

function actionLabel(action) {
  const name = String(action?.action || "automatic action").replaceAll("_", " ");
  const component = action?.component_id ? ` · ${action.component_id}` : "";
  const reason = action?.reason ? ` · ${action.reason}` : "";
  return `${name}${component}${reason}`;
}

export function renderMemoryStatus(job) {
  const status = job?.memory_status || {};
  const snapshot = status.latest_snapshot || {};
  const cuda = snapshot.cuda || {};
  const estimate = status.latest_estimate || {};
  const policy = status.effective_policy || status.requested_policy || job?.request?.memory_policy || "auto";
  const physical = physicalMemoryValues(cuda);
  const allocated = finiteNonnegative(cuda.allocated_vram_bytes);
  const reserved = finiteNonnegative(cuda.reserved_vram_bytes);
  const jobPeakAllocated = finiteNonnegative(
    status.job_peak_allocated_vram_bytes
      ?? status.peak_allocated_vram_bytes
      ?? cuda.peak_allocated_vram_bytes,
  );
  const jobPeakReserved = finiteNonnegative(
    status.job_peak_reserved_vram_bytes
      ?? status.peak_reserved_vram_bytes
      ?? cuda.peak_reserved_vram_bytes,
  );
  const overcommit = finiteNonnegative(cuda.allocator_overcommit_bytes)
    ?? (physical.total !== null && reserved !== null ? Math.max(0, reserved - physical.total) : null);
  const oversubscribed = cuda.allocator_oversubscribed === true
    || Boolean(overcommit && overcommit > 0);

  if ($("#memoryPolicyBadge")) {
    $("#memoryPolicyBadge").textContent = String(policy).replaceAll("_", " ");
  }
  if ($("#memoryPhysicalVramUsed")) $("#memoryPhysicalVramUsed").textContent = formatBytes(physical.used);
  if ($("#memoryPhysicalVramFree")) $("#memoryPhysicalVramFree").textContent = formatBytes(physical.free);
  if ($("#memoryPhysicalVramTotal")) $("#memoryPhysicalVramTotal").textContent = formatBytes(physical.total);
  if ($("#memoryTorchAllocated")) $("#memoryTorchAllocated").textContent = formatBytes(allocated);
  if ($("#memoryTorchReserved")) $("#memoryTorchReserved").textContent = formatBytes(reserved);
  if ($("#memoryJobPeakAllocated")) $("#memoryJobPeakAllocated").textContent = formatBytes(jobPeakAllocated);
  if ($("#memoryJobPeakReserved")) $("#memoryJobPeakReserved").textContent = formatBytes(jobPeakReserved);
  if ($("#memoryNextStagePeak")) {
    $("#memoryNextStagePeak").textContent = estimate.safety_adjusted_required_bytes
      ? `${formatBytes(estimate.safety_adjusted_required_bytes)} · ${estimate.stage || "next stage"}`
      : "Waiting";
  }
  const semanticsStatus = $("#memoryVramSemanticsStatus");
  if (semanticsStatus) {
    if (oversubscribed) {
      semanticsStatus.textContent = `PyTorch allocator reservation exceeds physical VRAM${overcommit !== null ? ` by ${formatBytes(overcommit)}` : ""}. Windows shared GPU memory or paging may be active.`;
    } else if (physical.total !== null) {
      semanticsStatus.textContent = `Physical VRAM is measured from CUDA total/free memory. PyTorch allocated and reserved values are shown separately.`;
    } else {
      semanticsStatus.textContent = "Physical VRAM measurement is unavailable. PyTorch allocator values are not treated as physical VRAM usage.";
    }
    semanticsStatus.classList.toggle("warning", oversubscribed);
  }
  if ($("#memoryActiveComponents")) {
    $("#memoryActiveComponents").textContent = listText(status.active_gpu_components, "None reported");
  }
  if ($("#memoryOffloadedComponents")) {
    $("#memoryOffloadedComponents").textContent = listText(status.offloaded_components, "None reported");
  }
  if ($("#memoryActiveStage")) {
    $("#memoryActiveStage").textContent = status.active_stage || job?.status || "Idle";
  }

  const actions = Array.isArray(status.automatic_actions) ? status.automatic_actions.slice(-5) : [];
  const container = $("#memoryAutomaticActions");
  if (container) {
    container.replaceChildren();
    container.classList.toggle("empty-state", actions.length === 0);
    if (!actions.length) {
      container.textContent = job ? "No automatic memory actions yet." : "No active or recent generation.";
    } else {
      actions.forEach((action) => {
        const item = document.createElement("div");
        item.textContent = actionLabel(action);
        container.append(item);
      });
    }
  }

  const previewStatus = $("#memoryPreviewStatus");
  if (previewStatus) {
    const suspended = Boolean(status.preview_image_decode_suspended);
    const reason = String(status.preview_image_decode_suspension_reason || "").trim();
    const source = String(status.preview_image_decode_suspension_source || "").trim().replaceAll("_", " ");
    const released = Boolean(status.preview_decoder_released);
    previewStatus.textContent = suspended
      ? `Image-preview decoding suspended${source ? ` (${source})` : ""}. ${reason || "VRAM was reserved for the active generation stage."} CFG telemetry continues${released ? "; preview decoder released" : ""}.`
      : "Image preview active; CFG telemetry is independent.";
    previewStatus.classList.toggle("warning", suspended);
  }
}

function runtimeText(value, fallback = "Unavailable") {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function setRuntimeText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value;
}

export function renderRuntimeStartupStatus(status) {
  const runtime = status?.runtime || {};
  const profile = runtime.runtime_profile || {};
  const attention = runtime.attention || {};
  const memory = runtime.memory || {};
  const hires = runtime.hires || {};
  const preview = runtime.preview || {};
  const vae = runtime.vae || {};
  const oom = runtime.oom_retry || {};
  const requested = runtimeText(attention.requested_backend, "unavailable");
  const effective = runtimeText(attention.effective_backend, "unverified");
  const provider = attention.provider_verified
    ? runtimeText(attention.verified_kernel_provider)
    : "Unverified";
  const vaeParts = [
    `tiling ${vae.tiling ? "on" : "off"}`,
    `slicing ${vae.slicing ? "on" : "off"}`,
    runtimeText(vae.device, "auto"),
  ];
  const oomText = oom.enabled
    ? `${runtimeText(oom.profile)} · limit ${Number(oom.limit || 0)}`
    : "Disabled";

  setRuntimeText("#runtimeProfileStatus", runtimeText(profile.profile_id, "default"));
  setRuntimeText("#runtimeAttentionStatus", `${requested} → ${effective}`);
  setRuntimeText("#runtimeKernelProviderStatus", provider);
  setRuntimeText("#runtimeMemoryPolicyStatus", runtimeText(memory.policy));
  setRuntimeText("#runtimeHiresMemoryStatus", runtimeText(hires.memory_profile));
  setRuntimeText("#runtimePreviewPolicyStatus", runtimeText(preview.policy));
  setRuntimeText("#runtimeVaeStatus", vaeParts.join(" · "));
  setRuntimeText("#runtimeOomStatus", oomText);

  const restartRequired = Boolean(runtime.restart_required || status?.restart_required);
  const pendingBlocked = Boolean(runtime.pending_change_blocked || status?.pending_change_blocked);
  const restartSettings = runtime.restart_required_settings || [];
  setRuntimeText("#runtimeRestartBadge", pendingBlocked ? "Override blocked" : (restartRequired ? "Restart pending" : "Active"));
  setRuntimeText(
    "#runtimeRestartStatus",
    restartRequired || pendingBlocked
      ? status?.message || `Restart required for: ${restartSettings.join(", ")}.`
      : "Process-start settings are active. Per-job memory settings apply to the next queued job.",
  );
  $("#runtimeRestartStatus")?.classList.toggle("warning", restartRequired || pendingBlocked);
}


async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard access is unavailable.");
}

export function bindRuntimeCommandCopy({ api, notify = () => {} } = {}) {
  const button = $("#copyRuntimeCommandButton");
  const status = $("#runtimeCommandCopyStatus");
  if (!button || !api?.runtimeCommand || button.dataset.bound === "true") return;
  button.dataset.bound = "true";
  button.addEventListener("click", async () => {
    button.disabled = true;
    if (status) status.textContent = "Building canonical runtime command…";
    try {
      const result = await api.runtimeCommand();
      const command = String(result?.set_command || "").trim();
      if (!command) throw new Error("The backend returned an empty runtime command.");
      await copyText(command);
      const mode = result?.mode === "pending" ? "pending restart" : "active";
      if (status) status.textContent = `Copied ${mode} runtime configuration.`;
      notify(`Copied ${mode} COMMANDLINE_ARGS.`);
    } catch (error) {
      if (status) status.textContent = `Copy failed: ${error.message}`;
      notify(`Runtime command copy failed: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  });
}
