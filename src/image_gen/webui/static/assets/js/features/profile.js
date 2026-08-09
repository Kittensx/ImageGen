import { api } from "../api.js?v=profile2";
import { $, notify } from "../utils.js";

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

function installedFor(value) {
  if (!value) return "Install date unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Install date unavailable";
  const days = Math.max(0, Math.floor((Date.now() - date.getTime()) / 86400000));
  if (days < 1) return "Installed today";
  if (days === 1) return "Installed 1 day ago";
  return `Installed ${days.toLocaleString()} days ago`;
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

  setText("#homeProfileDiscordState", discord.linked ? (discord.display_name || "Discord linked") : "Not connected");
  setText(
    "#homeProfileDiscordServerState",
    discord.server_member
      ? `Connected${discord.server_name ? ` · ${discord.server_name}` : ""}`
      : "Server membership not linked",
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
        ? "Discord sharing is available. Rich Presence is controlled by your opt-in below; server profile posting remains bot-backed."
        : "Rich Presence support is configured. Account/server linking will be completed by the ImageGen Discord integration.";
    } else if (capabilities.application_id_configured) {
      capabilityText.textContent = "The Discord application is configured; the Discord Social SDK presence helper still needs to be installed. Your sharing choices are already stored locally.";
    } else {
      capabilityText.textContent = "Discord account linking and the server bot are not configured yet. Your choices below are stored locally until those integrations are available.";
    }
  }

  setText("#homeDiscordActivityState", activity.state || "Aggregate stats only");

  const refresh = $("#homeDiscordPresenceRefresh");
  if (refresh) refresh.disabled = !(capabilities.rich_presence_ready && sharing.discord_rich_presence_enabled);

  const serverButton = $("#homeDiscordServerButton");
  if (serverButton) {
    const url = capabilities.community_url || "";
    serverButton.disabled = !url;
    serverButton.textContent = capabilities.invite_configured ? "Join ImageGen Server" : "Open ImageGen Server";
    serverButton.title = capabilities.invite_configured
      ? "Open the ImageGen community invite in Discord."
      : "Open the ImageGen server in Discord. A public invite will replace this link when configured.";
  }

  const connect = $("#homeDiscordConnectButton");
  if (connect) {
    const ready = Boolean(capabilities.account_linking_ready);
    connect.disabled = !discord.linked && !ready;
    connect.textContent = discord.linked ? "Disconnect Discord" : "Connect Discord";
    connect.title = discord.linked
      ? "Disconnect this local ImageGen profile from Discord."
      : ready
        ? "Authorize ImageGen through Discord. The WebUI stores only your public Discord identity, not OAuth tokens."
        : capabilities.application_id_configured
          ? "Install the ImageGen Discord native helper to enable account linking."
          : "Configure the ImageGen Discord application ID to enable account linking.";
  }
}

function renderProfile(profile) {
  currentProfile = profile || {};
  const usage = currentProfile.usage || {};
  const bugs = currentProfile.bugs || {};

  setText("#homeProfileInstalledAt", formatDate(currentProfile.installed_at));
  setText("#homeProfileInstalledFor", installedFor(currentProfile.installed_at));
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
    if (sharingValues().discord_rich_presence_enabled && presence.state === "discord_application_required") {
      notify("Discord sharing preference saved. Rich Presence will activate after the ImageGen Discord application is configured.");
    } else if (sharingValues().discord_rich_presence_enabled && presence.state === "presence_helper_required") {
      notify("Discord sharing preference saved. Rich Presence will activate after the Social SDK presence helper is installed.");
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
    button.textContent = linked ? "Disconnecting…" : "Waiting for Discord…";
  }
  try {
    const profile = linked
      ? await api.disconnectDiscordProfile()
      : await api.connectDiscordProfile();
    renderProfile(profile);
    if (linked) {
      notify("Discord disconnected from this ImageGen profile.");
    } else {
      const name = profile?.discord?.display_name || profile?.discord?.username || "Discord account";
      notify(`${name} connected to ImageGen.`);
    }
  } catch (error) {
    notify(`Unable to ${linked ? "disconnect" : "connect"} Discord: ${error.message}`, "error");
    await refreshProfile();
  }
}

function openDiscordServer() {
  const url = String(currentProfile?.discord_capabilities?.community_url || "");
  if (!url.startsWith("https://discord.com/") && !url.startsWith("https://discord.gg/")) {
    notify("The ImageGen Discord server link is not configured yet.", "error");
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

async function refreshPresence() {
  const button = $("#homeDiscordPresenceRefresh");
  if (button) button.disabled = true;
  try {
    const result = await api.refreshDiscordPresence();
    renderProfile(result?.profile || currentProfile || {});
    const presence = result?.presence || {};
    notify(presence.published ? "Discord Rich Presence refreshed." : `Discord presence was not published: ${presence.state || "unavailable"}.`);
  } catch (error) {
    notify(`Unable to refresh Discord presence: ${error.message}`, "error");
  } finally {
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
