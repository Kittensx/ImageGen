import { api } from "../api.js?v=profile2";
import { productName } from "../branding.js?v=brand1";
import { $, notify } from "../utils.js";
import { setActionIcon } from "../components/action-icons.js?v=0.1.1";

let currentProfile = null;
let saving = false;

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = String(value ?? "");
}

function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" }).format(date);
}

function profileAge(value) {
  if (!value) return "Profile start date unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Profile start date unavailable";
  const days = Math.max(0, Math.floor((Date.now() - date.getTime()) / 86400000));
  if (days < 1) return "Profile started today";
  if (days === 1) return "Profile started 1 day ago";
  return `Profile started ${days.toLocaleString()} days ago`;
}

function profileStartLabel(source) {
  if (source === "install_artifact") return "Installed";
  if (source === "environment_marker") return "Environment created";
  return "Profile started";
}

function percent(value) {
  const numeric = Number(value || 0);
  return `${Math.round(Math.max(0, Math.min(1, numeric)) * 100)}%`;
}

function setCheckbox(selector, value) {
  const node = $(selector);
  if (node) node.checked = Boolean(value);
}

function renderDiscordCommunity(status = {}) {
  const container = $("#homeDiscordCommunityStatus");
  const countNode = $("#homeDiscordOnlineCount");
  const privacyNode = $("#homeDiscordOnlinePrivacy");
  const numericCount = Number(status?.online_count);
  const hasCount = Boolean(status?.ok) && Number.isInteger(numericCount) && numericCount >= 0;
  const stale = hasCount && Boolean(status?.stale || status?.state === "stale");

  container?.classList.toggle("is-online", hasCount && !stale);
  container?.classList.toggle("is-stale", stale);

  if (hasCount) {
    const label = numericCount === 1 ? "member online" : "members online";
    if (countNode) countNode.textContent = `${numericCount.toLocaleString()} ${label}`;
    if (privacyNode) {
      privacyNode.textContent = stale
        ? "Showing the last available Discord count. Member names are not displayed."
        : "Live Discord server count. Member names are not displayed.";
    }
    return;
  }

  if (countNode) countNode.textContent = "Community status unavailable";
  if (privacyNode) privacyNode.textContent = "The Discord server can still be opened below; member names are never displayed here.";
}

async function refreshDiscordCommunity() {
  try {
    renderDiscordCommunity(await api.discordCommunityStatus());
  } catch (_error) {
    renderDiscordCommunity({ ok: false, state: "unavailable" });
  }
}

function renderDiscord(profile) {
  const discord = profile?.discord || {};
  const sharing = profile?.sharing || {};
  const capabilities = profile?.discord_capabilities || {};
  const activity = profile?.discord_activity_preview || {};

  setText("#homeProfileDiscordState", discord.linked ? "Discord linked" : "Not connected");
  setText(
    "#homeProfileDiscordServerState",
    discord.linked
      ? `Linked to this local ${productName()} profile`
      : "Connect Discord to link community sharing",
  );

  setCheckbox("#homeDiscordPresenceEnabled", sharing.discord_rich_presence_enabled);
  setCheckbox("#homeDiscordIntroEnabled", sharing.discord_intro_card_enabled);
  setCheckbox("#homeShareInstallDate", sharing.share_install_date);
  setCheckbox("#homeShareImageCount", sharing.share_image_count);
  setCheckbox("#homeShareBugStats", sharing.share_bug_stats);

  const capabilityBadge = $("#homeDiscordCapabilityBadge");
  if (capabilityBadge) {
    capabilityBadge.classList.toggle("is-ready", Boolean(capabilities.rich_presence_ready));
    capabilityBadge.classList.toggle("is-pending", !capabilities.rich_presence_ready);
    capabilityBadge.textContent = capabilities.rich_presence_ready ? "Presence ready" : "Setup pending";
  }

  const capabilityText = $("#homeDiscordCapabilityText");
  if (capabilityText) {
    if (capabilities.rich_presence_ready) {
      capabilityText.textContent = discord.linked
        ? "Your Discord account is linked locally through the Social SDK. Rich Presence is controlled by your opt-in below; posting into server channels would still require a separate bot integration."
        : `Discord Social SDK support is ready. Connect your account to link this local ${productName()} profile and enable opt-in Rich Presence.`;
    } else if (capabilities.application_id_configured) {
      capabilityText.textContent = "The Discord application is configured; the Discord Social SDK presence helper still needs to be installed. Your sharing choices are already stored locally.";
    } else {
      capabilityText.textContent = "Discord account linking is not configured yet. Your sharing choices are stored locally until the Social SDK integration is available.";
    }
  }

  setText("#homeDiscordActivityState", activity.state || "Aggregate stats only");

  const refresh = $("#homeDiscordPresenceRefresh");
  if (refresh) {
    refresh.disabled = !(capabilities.rich_presence_ready && sharing.discord_rich_presence_enabled);
    setActionIcon(refresh, "refresh", { label: "Refresh Discord Presence", title: "Refresh Discord Presence", replace: true });
  }

  const serverButton = $("#homeDiscordServerButton");
  if (serverButton) {
    const url = capabilities.community_url || "";
    serverButton.disabled = !url;
    const label = capabilities.invite_configured ? `Join ${productName()} Server` : `Open ${productName()} Server`;
    const title = capabilities.invite_configured
      ? `Open the ${productName()} community invite in Discord.`
      : `Open the ${productName()} server in Discord. A public invite will replace this link when configured.`;
    setActionIcon(serverButton, "external-link", { label, title, replace: true });
  }

  const connect = $("#homeDiscordConnectButton");
  if (connect) {
    const ready = Boolean(capabilities.account_linking_ready);
    connect.disabled = !discord.linked && !ready;
    const label = discord.linked ? "Disconnect Discord" : "Connect Discord";
    const title = discord.linked
      ? `Disconnect this local ${productName()} profile from Discord.`
      : ready
        ? `Authorize ${productName()} through Discord. The WebUI stores only your public Discord identity, not OAuth tokens.`
        : capabilities.application_id_configured
          ? `Install the ${productName()} Discord native helper to enable account linking.`
          : `Configure the ${productName()} Discord application ID to enable account linking.`;
    setActionIcon(connect, "discord", { label, title, replace: true });
  }
}

function linkedProfileGreeting(profile) {
  const discord = profile?.discord || {};
  if (!discord.linked) return "";
  const name = String(discord.display_name || discord.username || "").trim();
  return name ? `Hello, ${name}.` : "";
}

function renderWelcomeGreeting(profile) {
  const greeting = $("#homeWelcomeGreeting");
  if (!greeting) return;
  const text = linkedProfileGreeting(profile);
  greeting.textContent = text;
  greeting.hidden = !text;
}

function renderProfile(profile) {
  currentProfile = profile || {};
  const usage = currentProfile.usage || {};
  const bugs = currentProfile.bugs || {};
  renderWelcomeGreeting(currentProfile);

  setText("#homeProfileInstalledLabel", profileStartLabel(currentProfile.install_date_source));
  setText("#homeProfileInstalledAt", formatDate(currentProfile.installed_at));
  setText("#homeProfileInstalledFor", profileAge(currentProfile.installed_at));
  setText("#homeProfileVersion", currentProfile.last_seen_version || "Unknown");
  setText("#homeProfileFirstVersion", `First seen version ${currentProfile.first_seen_version || "unknown"}`);
  setText("#homeProfileImagesCreated", Number(usage.images_generated || 0).toLocaleString());
  setText("#homeBugReportedCount", bugs.reported || 0);
  setText("#homeBugOpenCount", bugs.open || 0);
  setText("#homeBugResolvedCount", bugs.resolved || 0);
  setText("#homeBugPendingCount", bugs.pending || 0);
  setText("#homeBugResolutionRate", percent(bugs.resolution_rate));

  const installBadge = $("#homeProfileInstallBadge");
  if (installBadge) {
    installBadge.textContent = "Local profile";
    installBadge.classList.add("is-active");
  }

  renderDiscord(currentProfile);
}

function sharingValues() {
  return {
    discord_rich_presence_enabled: Boolean($("#homeDiscordPresenceEnabled")?.checked),
    discord_intro_card_enabled: Boolean($("#homeDiscordIntroEnabled")?.checked),
    share_install_date: Boolean($("#homeShareInstallDate")?.checked),
    share_image_count: Boolean($("#homeShareImageCount")?.checked),
    share_bug_stats: Boolean($("#homeShareBugStats")?.checked),
  };
}

async function saveSharing() {
  if (saving) return;
  saving = true;
  try {
    const profile = await api.updateProfileSharing(sharingValues());
    renderProfile(profile);
    const presence = profile?.presence_publish || {};
    if (profile?.presence_diagnostic_created) {
      window.dispatchEvent(new CustomEvent("image-gen-bug-report-refresh", {
        detail: { source: "discord_presence", stage: profile.presence_diagnostic_stage || "discord_presence_refresh" },
      }));
    }
    if (sharingValues().discord_rich_presence_enabled && presence.state === "discord_application_required") {
      notify(`Discord sharing preference saved. Rich Presence will activate after the ${productName()} Discord application is configured.`);
    } else if (sharingValues().discord_rich_presence_enabled && ["presence_helper_required", "native_helper_required", "helper_required"].includes(presence.state)) {
      notify("Discord sharing preference saved. Rich Presence will activate after the Social SDK presence helper is installed.");
    } else if (sharingValues().discord_rich_presence_enabled && !presence.published && profile?.presence_diagnostic_created) {
      const detail = String(presence.message || "").trim();
      notify(`Discord sharing was saved, but presence failed: ${presence.state || "unavailable"}.${detail ? ` ${detail}` : ""} A diagnostic was added to Bug Reports.`, "error");
    }
  } catch (error) {
    notify(`Unable to save profile sharing preferences: ${error.message}`, "error");
  } finally {
    saving = false;
  }
}

async function refreshProfile() {
  try {
    const profile = await api.profile();
    renderProfile(profile);
    return profile;
  } catch (error) {
    setText("#homeProfileInstalledFor", `Profile unavailable: ${error.message}`);
    return null;
  }
}

async function toggleDiscordConnection() {
  const button = $("#homeDiscordConnectButton");
  const linked = Boolean(currentProfile?.discord?.linked);
  if (button) {
    button.disabled = true;
    button.classList.add("is-working");
    setActionIcon(button, "discord", {
      label: linked ? "Disconnecting Discord" : "Waiting for Discord",
      title: linked ? "Disconnecting Discord…" : "Waiting for Discord…",
      replace: true,
    });
  }
  try {
    const profile = linked
      ? await api.disconnectDiscordProfile()
      : await api.connectDiscordProfile();
    renderProfile(profile);
    if (linked) {
      notify(`Discord disconnected from this ${productName()} profile.`);
    } else {
      notify(`Discord account connected to ${productName()}.`);
    }
  } catch (error) {
    notify(`Unable to ${linked ? "disconnect" : "connect"} Discord: ${error.message}`, "error");
    await refreshProfile();
  } finally {
    button?.classList.remove("is-working");
    renderDiscord(currentProfile || {});
  }
}

function openDiscordServer() {
  const url = String(currentProfile?.discord_capabilities?.community_url || "");
  if (!url.startsWith("https://discord.com/") && !url.startsWith("https://discord.gg/")) {
    notify(`The ${productName()} Discord server link is not configured yet.`, "error");
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

async function refreshPresence() {
  const button = $("#homeDiscordPresenceRefresh");
  if (button) {
    button.disabled = true;
    button.classList.add("is-working");
  }
  try {
    const result = await api.refreshDiscordPresence();
    renderProfile(result?.profile || currentProfile || {});
    const presence = result?.presence || {};
    const state = presence.state || "unavailable";
    const detail = String(presence.message || "").trim();
    if (result?.diagnostic_created) {
      window.dispatchEvent(new CustomEvent("image-gen-bug-report-refresh", {
        detail: { source: "discord_presence", stage: result.diagnostic_stage || "discord_presence_refresh" },
      }));
    }
    notify(
      presence.published
        ? "Discord Rich Presence refreshed."
        : `Discord presence was not published: ${state}.${detail ? ` ${detail}` : ""}${result?.diagnostic_created ? " A diagnostic was added to Bug Reports." : ""}`,
      presence.published ? undefined : "error",
    );
  } catch (error) {
    notify(`Unable to refresh Discord presence: ${error.message}`, "error");
  } finally {
    button?.classList.remove("is-working");
    renderDiscord(currentProfile || {});
  }
}

export function bindImageGenProfile() {
  [
    "#homeDiscordPresenceEnabled",
    "#homeDiscordIntroEnabled",
    "#homeShareInstallDate",
    "#homeShareImageCount",
    "#homeShareBugStats",
  ].forEach((selector) => $(selector)?.addEventListener("change", saveSharing));

  $("#homeDiscordConnectButton")?.addEventListener("click", toggleDiscordConnection);
  $("#homeDiscordServerButton")?.addEventListener("click", openDiscordServer);
  $("#homeDiscordPresenceRefresh")?.addEventListener("click", refreshPresence);
  window.addEventListener("image-gen-profile-refresh", refreshProfile);
  refreshProfile();
  refreshDiscordCommunity();
  window.setInterval(() => {
    if (!document.hidden) refreshDiscordCommunity();
  }, 60000);

  return { refresh: refreshProfile, render: renderProfile, refreshDiscordCommunity };
}
