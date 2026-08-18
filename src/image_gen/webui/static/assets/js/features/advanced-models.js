import { api } from "../api.js";
import { $, notify } from "../utils.js";

let catalog = null;
let registryStatus = null;
let registryBrowserData = null;
let selectedRegistryComponentSha = "";

function familyEntry(family) {
  return (catalog?.families || []).find((item) => item.family === family) || null;
}

function option(value, label, { disabled = false, selected = false } = {}) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  node.disabled = disabled;
  node.selected = selected;
  return node;
}

function currentSelections() {
  return Object.fromEntries(
    Array.from(document.querySelectorAll("[data-advanced-component-role]")).map((node) => [node.dataset.advancedComponentRole, node.value]),
  );
}

function selectedBaseFingerprint() {
  const family = $("#advancedModelFamily")?.value || "";
  const entry = familyEntry(family);
  const baseRole = entry?.base_weight_role || "";
  if (!baseRole) return "";
  const select = document.querySelector(`[data-advanced-component-role="${baseRole}"]`);
  const value = String(select?.value || "").trim().toLowerCase();
  if (/^[0-9a-f]{64}$/.test(value)) return value;
  if (value === "auto") {
    const role = (entry.roles || []).find((item) => item.role === baseRole);
    const eligible = (role?.components || []).filter((component) => componentEligible(component, role));
    if (eligible.length === 1) return String(eligible[0].component_sha256 || "");
  }
  return "";
}

function digitalAllowed() {
  return $("#advancedModelAllowDigitalComponents")?.checked !== false;
}

function showUnavailableRegisteredComponents() {
  return Boolean($("#advancedModelShowUnavailableSources")?.checked);
}

function componentHasAccessibleSource(component) {
  return Number(component?.available_source_count || 0) > 0;
}

function renderRegistryStatus(extra = "") {
  const node = $("#advancedModelRegistryStatus");
  if (!node) return;
  const locations = registryStatus?.location_catalog || {};
  const roots = registryStatus?.configured_roots || {};
  const accessible = Number(locations.accessible_location_count || 0);
  const unavailable = Number(locations.unavailable_location_count || 0);
  const unavailableRoots = Number(roots.unavailable_count || 0);
  const parts = [`Registry: ${accessible} accessible location${accessible === 1 ? "" : "s"}`];
  if (unavailable) parts.push(`${unavailable} unavailable/disconnected`);
  if (unavailableRoots) parts.push(`${unavailableRoots} configured root${unavailableRoots === 1 ? "" : "s"} currently unreachable`);
  if (extra) parts.push(extra);
  node.textContent = `${parts.join(" · ")}.`;
}

function sourcePolicyStatusText() {
  return digitalAllowed()
    ? "Physical, standalone, and digital checkpoint sources are all eligible."
    : "Only physical or standalone component sources are eligible. Digital checkpoint components remain visible but cannot be selected.";
}

function phase05StatusForBase(component, role = null) {
  const status = component?.phase05 || {};
  if (status.global_disabled) {
    return { eligible: false, label: "Globally disabled", reason: status.reason || "Globally disabled by component policy." };
  }
  const entry = familyEntry($("#advancedModelFamily")?.value || "");
  const baseRole = entry?.base_weight_role || "";
  if (role?.role === baseRole || role?.base_weight_role) {
    if (status.validation_state === "validation_failed") {
      return { eligible: false, label: "Validation failed", reason: status.reason || "Blocking validation failed." };
    }
    return { eligible: true, label: status.validation_label || "Untested", reason: "" };
  }
  const baseSha = selectedBaseFingerprint();
  if (baseSha) {
    const exclusion = (status.per_base_exclusions || []).find((item) => (
      String(item.base_component_sha256 || "").toLowerCase() === baseSha
      && (!item.component_role || item.component_role === role?.role)
    ));
    if (exclusion) {
      return { eligible: false, label: "Disabled for this base", reason: exclusion.reason || "Disabled for the selected base component." };
    }
    const blocking = (status.blocking_failures_by_base || []).find((item) => (
      !item.base_component_sha256 || String(item.base_component_sha256 || "").toLowerCase() === baseSha
    ));
    if (blocking) {
      return { eligible: false, label: "Validation failed", reason: blocking.error_message || "Blocking validation failed for this base." };
    }
    const passes = (status.validation_passes_by_base || []).filter((item) => (
      !item.base_component_sha256 || String(item.base_component_sha256 || "").toLowerCase() === baseSha
    ));
    if (passes.length) return { eligible: true, label: "Validated", reason: "" };
  }
  return { eligible: true, label: status.validation_label || "Untested", reason: "" };
}

function componentEligible(component, role = null) {
  const sourceEligible = digitalAllowed()
    ? Boolean(component?.selectable_with_digital)
    : Boolean(component?.selectable_without_digital);
  return sourceEligible && phase05StatusForBase(component, role).eligible;
}

function componentStatusLabel(component, role = null) {
  const phase05 = phase05StatusForBase(component, role);
  if (!phase05.eligible) return phase05.label;
  if (!componentEligible(component, role)) {
    if ((component?.source_status || "") === "digital") {
      return "Digital only — enable digital sources";
    }
    if ((component?.source_status || "") === "physical") {
      return "Physical source is not currently loadable for this role";
    }
    if ((component?.source_status || "") === "physical_and_digital") {
      return "No source is currently loadable under this role and policy";
    }
    return "Unavailable";
  }
  const source = String(component?.source_status_label || "Available");
  const validation = phase05.label && phase05.label !== "Untested" ? ` · ${phase05.label}` : "";
  return `${source}${validation}`;
}

function componentOptionLabel(component, role = null) {
  const selectionLabel = String(component?.selection_label || "").trim();
  if (selectionLabel && componentEligible(component, role)) {
    const phase05 = phase05StatusForBase(component, role);
    return phase05.label === "Validated" ? `${selectionLabel} · Validated` : selectionLabel;
  }
  const base = String(selectionLabel || component?.display_name || component?.component_sha256 || "component");
  return `${base} [${componentStatusLabel(component, role)}]`;
}

function setWholeModelControlsDisabled(disabled) {
  ["#modelPath", "#startupDefaultsTrigger", "#vaePath", "#vaeFetchCivitaiButton"].forEach((selector) => {
    const node = $(selector);
    if (node) node.disabled = Boolean(disabled);
  });
  const status = $("#advancedModelsStatus");
  if (status) {
    status.textContent = disabled
      ? "Advanced Models is authoritative. The checkpoint and standalone VAE selectors above are ignored."
      : "Whole-checkpoint generation is active. Enable Advanced Models to compose compatible registered components.";
  }
}

function renderT5Placement() {
  const select = document.querySelector('[data-advanced-component-role="text_encoder_3"]');
  const row = $("#advancedModelT5DeviceBlock");
  if (!row) return;
  const enabled = Boolean(select && String(select.value || "").trim() && select.value !== "off");
  row.hidden = !enabled;
  row.classList.toggle("is-hidden", !enabled);
}

function roleHelpText(role, components, eligibleCount) {
  if (role.required) {
    if (eligibleCount === 1) return "Auto resolves the only currently eligible unique fingerprint.";
    if (eligibleCount > 1) return "When multiple currently eligible unique fingerprints exist, choose one explicitly.";
    if (components.length > 0 && !digitalAllowed()) return "No current-policy candidates. Enable digital sources or add a physical/standalone component.";
    return "No compatible components are currently available for this required role.";
  }
  if (components.length > 0 && eligibleCount === 0 && !digitalAllowed()) {
    return "Optional components are hidden by current policy. Enable digital sources to use digital-only entries.";
  }
  return "Optional components are never enabled by Auto; Off is the default.";
}

function renderRoles(saved = {}) {
  const family = $("#advancedModelFamily")?.value || "";
  const target = $("#advancedModelComponents");
  const policyStatus = $("#advancedModelSourcePolicyStatus");
  if (policyStatus) policyStatus.textContent = sourcePolicyStatusText();
  if (!target) return;
  target.replaceChildren();
  const entry = familyEntry(family);
  if (!entry) {
    const message = document.createElement("small");
    message.className = "field-status subtle";
    message.textContent = "No scanned components are available for this model family. Update the component registry first.";
    target.append(message);
    renderT5Placement();
    return;
  }

  (entry.roles || []).forEach((role) => {
    const block = document.createElement("label");
    block.className = "field-block advanced-model-component-field";
    const heading = document.createElement("span");
    heading.textContent = `${role.label}${role.required ? " (required)" : " (optional)"}`;
    const select = document.createElement("select");
    select.dataset.advancedComponentRole = role.role;
    select.setAttribute("aria-label", role.label);
    const registeredComponents = role.components || [];
    const components = showUnavailableRegisteredComponents()
      ? registeredComponents
      : registeredComponents.filter((component) => componentHasAccessibleSource(component));
    const singleCatalogComponent = components.length === 1;
    const eligible = components.filter((component) => componentEligible(component, role));
    const savedValue = String(saved?.[role.role] ?? "").trim().toLowerCase();

    if (role.required) {
      if (eligible.length === 1) {
        select.append(option("auto", `Auto — ${componentOptionLabel(eligible[0], role)}`));
      } else if (eligible.length > 1) {
        select.append(option("", `Choose ${role.label} — ${eligible.length} eligible components`, { disabled: true, selected: !savedValue }));
      } else if (components.length > 0 && !digitalAllowed()) {
        select.append(option("", `No ${role.label} components allowed by current source policy`, { disabled: true, selected: true }));
      } else {
        select.append(option("", `No compatible ${role.label} components`, { disabled: true, selected: true }));
      }
    } else {
      select.append(option("off", "Off"));
    }

    components.forEach((component) => {
      select.append(option(component.component_sha256, componentOptionLabel(component, role), {
        disabled: !componentEligible(component, role),
      }));
    });

    if (savedValue && Array.from(select.options).some((item) => item.value === savedValue && !item.disabled)) {
      select.value = savedValue;
    } else if (role.required && eligible.length === 1) {
      select.value = "auto";
    } else if (!role.required) {
      select.value = "off";
    }

    const help = document.createElement("small");
    help.className = "field-status subtle";
    help.textContent = roleHelpText(role, components, eligible.length);
    help.dataset.singleCatalogComponent = singleCatalogComponent ? "true" : "false";

    block.append(heading, select, help);
    target.append(block);
    select.addEventListener("change", () => {
      const familyConfig = familyEntry($("#advancedModelFamily")?.value || "");
      if (role.role === familyConfig?.base_weight_role) {
        const savedSelections = currentSelections();
        renderRoles(savedSelections);
      }
      renderT5Placement();
      window.dispatchEvent(new CustomEvent("image-gen-advanced-models-changed"));
    });
  });
  renderT5Placement();
}

function populateFamilies(savedFamily = "", savedComponents = {}) {
  const select = $("#advancedModelFamily");
  if (!select) return;
  select.replaceChildren();
  (catalog?.families || [])
    .filter((family) => family.constructible !== false)
    .forEach((family) => select.append(option(family.family, family.label)));
  if (savedFamily && Array.from(select.options).some((item) => item.value === savedFamily)) {
    select.value = savedFamily;
  }
  renderRoles(savedComponents);
}

function registryBadge(text) {
  const node = document.createElement("span");
  node.className = "component-registry-badge";
  node.textContent = String(text || "");
  return node;
}

function registryPath(text) {
  const node = document.createElement("div");
  node.className = "component-registry-path";
  node.textContent = String(text || "");
  return node;
}

function renderRegistryBrowserLists() {
  const data = registryBrowserData || { models: [], components: [], summary: {} };
  const modelsTarget = $("#componentRegistryModels");
  const componentsTarget = $("#componentRegistryComponents");
  if (!modelsTarget || !componentsTarget) return;
  modelsTarget.replaceChildren();
  componentsTarget.replaceChildren();

  (data.models || []).forEach((model) => {
    const card = document.createElement("article");
    card.className = "component-registry-card";
    const heading = document.createElement("div");
    heading.className = "component-registry-card-heading";
    const name = document.createElement("strong");
    name.textContent = model.filename || "Registered checkpoint";
    heading.append(name, registryBadge(model.location_state || (model.accessible ? "available" : "unavailable")));
    const badges = document.createElement("div");
    badges.className = "component-registry-badges";
    if (model.architecture) badges.append(registryBadge(model.architecture));
    if (model.sha256) badges.append(registryBadge(`SHA ${String(model.sha256).slice(0, 12)}`));
    badges.append(registryBadge(`${(model.components || []).length} components`));
    card.append(heading, badges, registryPath(model.path || ""));
    modelsTarget.append(card);
  });

  (data.components || []).forEach((component) => {
    const card = document.createElement("article");
    card.className = `component-registry-card is-selectable${component.component_sha256 === selectedRegistryComponentSha ? " is-selected" : ""}`;
    card.tabIndex = 0;
    const heading = document.createElement("div");
    heading.className = "component-registry-card-heading";
    const hash = document.createElement("strong");
    hash.textContent = component.short_hash || String(component.component_sha256 || "").slice(0, 12);
    heading.append(hash, registryBadge(component.accessible_source_count ? "Accessible" : "Unavailable"));
    const badges = document.createElement("div");
    badges.className = "component-registry-badges";
    (component.families || []).forEach((value) => badges.append(registryBadge(value)));
    (component.roles || []).forEach((value) => badges.append(registryBadge(value)));
    const phase05 = component.policy || {};
    if (phase05.global_disabled) badges.append(registryBadge("Globally disabled"));
    else if (phase05.validation_state === "validation_failed") badges.append(registryBadge("Validation failed"));
    else if (phase05.validation_state === "validated") badges.append(registryBadge("Validated"));
    else badges.append(registryBadge("Untested"));
    badges.append(registryBadge(`${component.registered_source_count || 0} sources`));
    if (component.relationship_evidence_count) badges.append(registryBadge(`${component.relationship_evidence_count} relationships`));
    const source = (component.sources || [])[0];
    card.append(heading, badges, registryPath(source?.path || source?.filename || "No registered source path"));
    const choose = () => {
      selectedRegistryComponentSha = String(component.component_sha256 || "");
      renderRegistryBrowserLists();
      void renderRegistryComponentDetail(component);
    };
    card.addEventListener("click", choose);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        choose();
      }
    });
    componentsTarget.append(card);
  });

  if (!data.models?.length) modelsTarget.append(registryPath("No registered model locations match the current filters."));
  if (!data.components?.length) componentsTarget.append(registryPath("No fingerprinted components match the current filters."));
  if ($("#componentRegistryModelCount")) $("#componentRegistryModelCount").textContent = String(data.models?.length || 0);
  if ($("#componentRegistryComponentCount")) $("#componentRegistryComponentCount").textContent = String(data.components?.length || 0);
  const summary = data.summary || {};
  if ($("#componentRegistrySummary")) {
    $("#componentRegistrySummary").textContent = `${summary.accessible_location_count || 0} accessible registered locations · ${summary.unavailable_location_count || 0} unavailable/disconnected · ${data.components?.length || 0} component identities shown.`;
  }
}

function detailSection(title, content) {
  const section = document.createElement("section");
  section.className = "component-registry-detail-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  if (content instanceof Node) section.append(content);
  else section.append(registryPath(content));
  return section;
}

function jsonBlock(value) {
  const pre = document.createElement("pre");
  pre.className = "component-registry-json";
  pre.textContent = JSON.stringify(value ?? {}, null, 2);
  return pre;
}

async function renderRegistryComponentDetail(component) {
  const target = $("#componentRegistryDetail");
  if (!target || !component?.component_sha256) return;
  target.replaceChildren(registryPath("Loading component evidence…"));
  try {
    const evidence = await api.componentRegistryEvidence(component.component_sha256);
    target.replaceChildren();
    const identity = document.createElement("div");
    identity.append(
      registryPath(`Fingerprint: ${component.component_sha256}`),
      registryPath(`Families: ${(component.families || []).join(", ") || "unclassified"}`),
      registryPath(`Roles: ${(component.roles || []).join(", ") || "unclassified"}`),
    );
    target.append(detailSection("Identity", identity));

    const sources = document.createElement("div");
    (component.sources || []).forEach((source) => {
      const row = document.createElement("div");
      row.className = "component-registry-card";
      row.append(
        registryPath(source.path || source.filename || ""),
        registryPath(`${source.source_form || "unknown source"} · ${source.location_state || source.availability_state || "unknown state"}`),
      );
      sources.append(row);
    });
    if (!(component.sources || []).length) sources.append(registryPath("No source occurrences recorded."));
    target.append(detailSection("Known locations", sources));

    const baseSha = selectedBaseFingerprint();
    const primaryRole = String((component.roles || [])[0] || "");
    const globalDisabled = (evidence.policies || []).some((item) => item.policy_scope === "global" && item.policy_action === "disable");
    const baseDisabled = Boolean(baseSha) && (evidence.policies || []).some((item) => (
      item.policy_scope === "base"
      && String(item.base_component_sha256 || "").toLowerCase() === baseSha.toLowerCase()
      && (!item.component_role || item.component_role === primaryRole)
    ));
    const actions = document.createElement("div");
    actions.className = "component-registry-policy-actions";
    const globalButton = document.createElement("button");
    globalButton.type = "button";
    globalButton.className = "secondary-button compact-button";
    globalButton.textContent = globalDisabled ? "Re-enable globally" : "Disable globally";
    globalButton.addEventListener("click", async () => {
      try {
        if (globalDisabled) {
          await api.clearComponentPolicy({ component_sha256: component.component_sha256, policy_scope: "global" });
        } else {
          await api.setComponentPolicy({ component_sha256: component.component_sha256, policy_scope: "global", component_role: primaryRole, reason: "Disabled from the Model / Component Registry browser." });
        }
        await refreshRegistryBrowserData();
        const updated = (registryBrowserData?.components || []).find((item) => item.component_sha256 === component.component_sha256);
        if (updated) await renderRegistryComponentDetail(updated);
        catalog = await api.advancedModelComponents();
        populateFamilies($("#advancedModelFamily")?.value || "", currentSelections());
      } catch (error) {
        notify(`Component policy update failed: ${error.message}`, "error");
      }
    });
    actions.append(globalButton);
    if (baseSha && baseSha !== component.component_sha256) {
      const baseButton = document.createElement("button");
      baseButton.type = "button";
      baseButton.className = "secondary-button compact-button";
      baseButton.textContent = baseDisabled ? "Re-enable for selected base" : "Exclude for selected base";
      baseButton.addEventListener("click", async () => {
        try {
          if (baseDisabled) {
            await api.clearComponentPolicy({
              component_sha256: component.component_sha256,
              policy_scope: "base",
              base_component_sha256: baseSha,
              component_role: primaryRole,
            });
          } else {
            await api.setComponentPolicy({
              component_sha256: component.component_sha256,
              policy_scope: "base",
              base_component_sha256: baseSha,
              component_role: primaryRole,
              reason: "Excluded for the selected base from the Model / Component Registry browser.",
            });
          }
          catalog = await api.advancedModelComponents();
          populateFamilies($("#advancedModelFamily")?.value || "", currentSelections());
          await refreshRegistryBrowserData();
        } catch (error) {
          notify(`Per-base component policy update failed: ${error.message}`, "error");
        }
      });
      actions.append(baseButton);
    }
    const clearValidationButton = document.createElement("button");
    clearValidationButton.type = "button";
    clearValidationButton.className = "secondary-button compact-button";
    clearValidationButton.textContent = "Clear validation evidence";
    clearValidationButton.disabled = !(evidence.validations || []).length;
    clearValidationButton.addEventListener("click", async () => {
      try {
        await api.clearComponentValidation({ component_sha256: component.component_sha256 });
        await refreshRegistryBrowserData();
        const updated = (registryBrowserData?.components || []).find((item) => item.component_sha256 === component.component_sha256);
        if (updated) await renderRegistryComponentDetail(updated);
      } catch (error) {
        notify(`Validation evidence could not be cleared: ${error.message}`, "error");
      }
    });
    actions.append(clearValidationButton);
    target.append(detailSection("User policy", actions));
    target.append(detailSection("Policy records", jsonBlock(evidence.policies || [])));
    target.append(detailSection("Compatibility validation", jsonBlock(evidence.validations || [])));
    target.append(detailSection("Analytical / provenance relationships", jsonBlock(evidence.relationships || [])));
  } catch (error) {
    target.replaceChildren(registryPath(`Unable to load component evidence: ${error.message}`));
  }
}

function registryBrowserFilters() {
  return {
    family: $("#componentRegistryFamilyFilter")?.value || "",
    role: $("#componentRegistryRoleFilter")?.value || "",
    accessibleOnly: Boolean($("#componentRegistryAccessibleOnly")?.checked),
    search: $("#componentRegistrySearch")?.value || "",
  };
}

async function refreshRegistryBrowserData() {
  registryBrowserData = await api.componentRegistryBrowser(registryBrowserFilters());
  const familyFilter = $("#componentRegistryFamilyFilter");
  const roleFilter = $("#componentRegistryRoleFilter");
  if (familyFilter && familyFilter.options.length <= 1) {
    (catalog?.families || []).forEach((item) => familyFilter.append(option(item.family, item.label)));
  }
  if (roleFilter && roleFilter.options.length <= 1) {
    const roles = new Map();
    (catalog?.families || []).forEach((family) => (family.roles || []).forEach((role) => roles.set(role.role, role.label)));
    [...roles.entries()].sort((a, b) => a[1].localeCompare(b[1])).forEach(([value, label]) => roleFilter.append(option(value, label)));
  }
  renderRegistryBrowserLists();
}

async function openRegistryBrowser() {
  const dialog = $("#componentRegistryDialog");
  if (!dialog) return;
  selectedRegistryComponentSha = "";
  if (!dialog.open) dialog.showModal();
  try {
    await refreshRegistryBrowserData();
  } catch (error) {
    notify(`Component registry browser could not be loaded: ${error.message}`, "error");
  }
}

export async function bindAdvancedModels({ values = {}, saveSessionSoon = null } = {}) {
  const enabled = $("#advancedModelsEnabled");
  const family = $("#advancedModelFamily");
  const allowDigital = $("#advancedModelAllowDigitalComponents");
  const device = $("#advancedModelT5Device");
  const showUnavailable = $("#advancedModelShowUnavailableSources");
  const scanLibrary = $("#advancedModelScanLibraryButton");
  const registryBrowser = $("#advancedModelRegistryBrowserButton");
  if (!enabled || !family) return;

  enabled.checked = Boolean(values.advanced_models_enabled);
  if (allowDigital) allowDigital.checked = values.advanced_model_allow_digital_components !== false;
  if (device) device.value = String(values.advanced_model_t5_device || "cpu");

  try {
    [catalog, registryStatus] = await Promise.all([
      api.advancedModelComponents(),
      api.advancedModelRegistryStatus(false),
    ]);
    renderRegistryStatus();
    populateFamilies(String(values.advanced_model_family || ""), values.advanced_model_components || {});
  } catch (error) {
    catalog = { families: [] };
    registryStatus = null;
    populateFamilies();
    renderRegistryStatus("status unavailable");
    notify(`Advanced Models component catalog could not be loaded: ${error.message}`, "error");
  }

  const syncEnabled = () => {
    const on = Boolean(enabled.checked);
    $("#advancedModelsEditor").hidden = !on;
    setWholeModelControlsDisabled(on);
    saveSessionSoon?.();
  };
  enabled.addEventListener("change", syncEnabled);
  family.addEventListener("change", () => {
    renderRoles({});
    saveSessionSoon?.();
  });
  allowDigital?.addEventListener("change", () => {
    renderRoles(currentSelections());
    saveSessionSoon?.();
  });
  showUnavailable?.addEventListener("change", () => {
    renderRoles(currentSelections());
  });
  scanLibrary?.addEventListener("click", async () => {
    const savedFamily = String(family.value || "");
    const savedComponents = currentSelections();
    scanLibrary.disabled = true;
    const previousLabel = scanLibrary.textContent;
    scanLibrary.textContent = "Scanning…";
    renderRegistryStatus("scanning reachable model roots");
    try {
      const result = await api.refreshAdvancedModelRegistry({ strength: "structural" });
      catalog = result.component_catalog || await api.advancedModelComponents();
      registryStatus = {
        configured_roots: result.configured_roots || {},
        location_catalog: result.location_catalog || {},
      };
      populateFamilies(savedFamily, savedComponents);
      const relinked = Number(result?.derived?.reconciliation?.relinked_count || 0);
      const touched = Number(result?.result_count || 0);
      renderRegistryStatus(`${touched} scanned/updated · ${relinked} exact-SHA relink${relinked === 1 ? "" : "s"}`);
      notify(`Model library scan complete: ${touched} registry result${touched === 1 ? "" : "s"}, ${relinked} relink${relinked === 1 ? "" : "s"}.`, "success");
    } catch (error) {
      renderRegistryStatus("last scan failed");
      notify(`Model library scan failed: ${error.message}`, "error");
    } finally {
      scanLibrary.disabled = false;
      scanLibrary.textContent = previousLabel;
    }
  });
  registryBrowser?.addEventListener("click", () => void openRegistryBrowser());
  $("#componentRegistryCloseButton")?.addEventListener("click", () => $("#componentRegistryDialog")?.close());
  $("#componentRegistryRefreshButton")?.addEventListener("click", async () => {
    const button = $("#componentRegistryRefreshButton");
    if (!button) return;
    button.disabled = true;
    const previous = button.textContent;
    button.textContent = "Scanning…";
    try {
      const result = await api.refreshAdvancedModelRegistry({ strength: "structural" });
      catalog = result.component_catalog || await api.advancedModelComponents();
      registryStatus = {
        configured_roots: result.configured_roots || {},
        location_catalog: result.location_catalog || {},
      };
      renderRegistryStatus("registry refreshed");
      populateFamilies(String(family.value || ""), currentSelections());
      await refreshRegistryBrowserData();
      notify("Model / Component Registry refreshed.", "success");
    } catch (error) {
      notify(`Registry refresh failed: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      button.textContent = previous;
    }
  });
  ["#componentRegistryFamilyFilter", "#componentRegistryRoleFilter", "#componentRegistryAccessibleOnly"].forEach((selector) => {
    $(selector)?.addEventListener("change", () => void refreshRegistryBrowserData());
  });
  let registrySearchTimer = 0;
  $("#componentRegistrySearch")?.addEventListener("input", () => {
    window.clearTimeout(registrySearchTimer);
    registrySearchTimer = window.setTimeout(() => void refreshRegistryBrowserData(), 250);
  });
  device?.addEventListener("change", () => saveSessionSoon?.());
  window.addEventListener("image-gen-advanced-models-changed", () => saveSessionSoon?.());
  window.addEventListener("image-gen-generation-values-applied", (event) => {
    const next = event.detail?.values || {};
    enabled.checked = Boolean(next.advanced_models_enabled);
    if (allowDigital) allowDigital.checked = next.advanced_model_allow_digital_components !== false;
    if (device) device.value = String(next.advanced_model_t5_device || "cpu");
    populateFamilies(String(next.advanced_model_family || ""), next.advanced_model_components || {});
    const on = Boolean(enabled.checked);
    $("#advancedModelsEditor").hidden = !on;
    setWholeModelControlsDisabled(on);
  });
  syncEnabled();
}
