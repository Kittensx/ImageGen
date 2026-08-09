const ICON_BY_STATUS = {
  healthy: "/assets/icons/status/green.svg",
  transitioning: "/assets/icons/status/green.svg",
  warning: "/assets/icons/status/amber.svg",
  critical: "/assets/icons/status/red.svg",
  inactive: "/assets/icons/status/gray.svg",
};

const LABEL_BY_STATUS = {
  healthy: "Healthy",
  transitioning: "Working",
  warning: "Caution",
  critical: "Problem",
  inactive: "Offline / inactive",
};

const registry = new Map();

function normalizedStatus(value) {
  const token = String(value || "inactive").trim().toLowerCase();
  return Object.hasOwn(ICON_BY_STATUS, token) ? token : "inactive";
}

function resolveHost(host) {
  if (host instanceof Element) return host;
  return document.querySelector(String(host || ""));
}

function ensureDialog() {
  let dialog = document.getElementById("systemStatusDiagnosticsDialog");
  if (dialog) return dialog;

  dialog = document.createElement("dialog");
  dialog.id = "systemStatusDiagnosticsDialog";
  dialog.className = "system-status-dialog";
  dialog.innerHTML = `
    <form method="dialog" class="system-status-dialog__surface">
      <header class="system-status-dialog__header">
        <div class="system-status-dialog__title-group">
          <img id="systemStatusDialogIcon" class="system-status-dialog__icon" alt="">
          <div>
            <strong id="systemStatusDialogTitle">Subsystem status</strong>
            <span id="systemStatusDialogState">Unavailable</span>
          </div>
        </div>
        <button class="icon-button" value="cancel" aria-label="Close subsystem diagnostics">×</button>
      </header>
      <section class="system-status-dialog__body">
        <p id="systemStatusDialogSummary"></p>
        <p id="systemStatusDialogDetail" class="field-status subtle"></p>
        <dl id="systemStatusDialogFacts" class="system-status-dialog__facts"></dl>
      </section>
      <footer class="system-status-dialog__actions">
        <button id="systemStatusDialogRelatedButton" class="secondary-button" value="none" type="button">Show related UI</button>
        <button class="primary-button" value="cancel">Close</button>
      </footer>
    </form>`;
  document.body.append(dialog);
  return dialog;
}

function factsEntries(facts) {
  if (!facts || typeof facts !== "object") return [];
  return Object.entries(facts).filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== "");
}

function showRelatedTarget(targetSelector) {
  if (!targetSelector) return;
  const target = document.querySelector(targetSelector);
  if (!target) return;
  if (target instanceof HTMLDetailsElement) target.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  if (typeof target.focus === "function") {
    window.setTimeout(() => target.focus({ preventScroll: true }), 250);
  }
}

function openDiagnostics(record) {
  const dialog = ensureDialog();
  const status = normalizedStatus(record.status);
  const icon = dialog.querySelector("#systemStatusDialogIcon");
  const title = dialog.querySelector("#systemStatusDialogTitle");
  const state = dialog.querySelector("#systemStatusDialogState");
  const summary = dialog.querySelector("#systemStatusDialogSummary");
  const detail = dialog.querySelector("#systemStatusDialogDetail");
  const facts = dialog.querySelector("#systemStatusDialogFacts");
  const related = dialog.querySelector("#systemStatusDialogRelatedButton");

  icon.src = ICON_BY_STATUS[status];
  icon.alt = "";
  title.textContent = record.label || "Subsystem status";
  state.textContent = record.stateLabel || LABEL_BY_STATUS[status];
  summary.textContent = record.summary || `${record.label || "Subsystem"}: ${state.textContent}`;
  detail.textContent = record.detail || "No additional diagnostic detail was reported.";

  facts.replaceChildren();
  factsEntries(record.facts).forEach(([key, value]) => {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = String(key).replaceAll("_", " ");
    description.textContent = String(value);
    row.append(term, description);
    facts.append(row);
  });

  const hasTarget = Boolean(record.diagnosticTarget && document.querySelector(record.diagnosticTarget));
  related.hidden = !hasTarget;
  related.onclick = hasTarget
    ? () => {
        dialog.close();
        showRelatedTarget(record.diagnosticTarget);
      }
    : null;

  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function ensureIndicator({ id, host, label, placement = "append" }) {
  let button = document.getElementById(id);
  if (button) return button;
  const container = resolveHost(host);
  if (!container) return null;

  button = document.createElement("button");
  button.id = id;
  button.type = "button";
  button.className = "system-status-indicator";
  button.dataset.status = "inactive";
  button.innerHTML = `<img class="system-status-indicator__icon" src="${ICON_BY_STATUS.inactive}" alt="">`;
  button.addEventListener("click", () => {
    const record = registry.get(id);
    if (record) openDiagnostics(record);
  });
  if (placement === "prepend") container.prepend(button);
  else container.append(button);
  button.title = `${label}: ${LABEL_BY_STATUS.inactive}`;
  button.setAttribute("aria-label", button.title);
  return button;
}

export function setSubsystemStatus({
  id,
  host,
  label,
  status = "inactive",
  stateLabel = "",
  summary = "",
  detail = "",
  facts = null,
  diagnosticTarget = "",
  placement = "append",
  hidden = false,
} = {}) {
  if (!id || !host || !label) return null;
  const button = ensureIndicator({ id, host, label, placement });
  if (!button) return null;
  const normalized = normalizedStatus(status);
  const record = {
    id,
    label,
    status: normalized,
    stateLabel: stateLabel || LABEL_BY_STATUS[normalized],
    summary,
    detail,
    facts,
    diagnosticTarget,
  };
  registry.set(id, record);
  button.hidden = Boolean(hidden);
  button.dataset.status = normalized;
  const icon = button.querySelector(".system-status-indicator__icon");
  if (icon) icon.src = ICON_BY_STATUS[normalized];
  const accessible = `${label}: ${record.stateLabel}. ${summary || detail || "Click for diagnostics."}`.trim();
  button.title = accessible;
  button.setAttribute("aria-label", accessible);
  return button;
}

export function getSubsystemStatus(id) {
  return registry.get(id) || null;
}
