#define DISCORDPP_IMPLEMENTATION
#include "discordpp.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

namespace {
constexpr const char* kHelperVersion = "1.0.0";

std::string read_stdin() {
  std::ostringstream stream;
  stream << std::cin.rdbuf();
  return stream.str();
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u00";
          const char* hex = "0123456789abcdef";
          out << hex[(ch >> 4) & 0x0F] << hex[ch & 0x0F];
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  return out.str();
}

std::optional<std::string> json_string(const std::string& json, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) return std::nullopt;
  std::string value = match[1].str();
  value = std::regex_replace(value, std::regex("\\\\n"), "\n");
  value = std::regex_replace(value, std::regex("\\\\r"), "\r");
  value = std::regex_replace(value, std::regex("\\\\t"), "\t");
  value = std::regex_replace(value, std::regex("\\\\\""), "\"");
  value = std::regex_replace(value, std::regex("\\\\\\\\"), "\\");
  return value;
}

uint64_t application_id_from(const std::string& json) {
  const auto value = json_string(json, "application_id");
  if (!value || value->empty()) return 0;
  try {
    return static_cast<uint64_t>(std::stoull(*value));
  } catch (...) {
    return 0;
  }
}

void write_error(const std::string& state, const std::string& message) {
  std::cout << "{\"ok\":false,\"state\":\"" << json_escape(state)
            << "\",\"message\":\"" << json_escape(message) << "\"}" << std::endl;
}

bool wait_until(const std::atomic<bool>& done, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    discordpp::RunCallbacks();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return done.load();
}

int status_command() {
  auto client = std::make_shared<discordpp::Client>();
  (void)client;
  std::cout << "{\"ok\":true,\"state\":\"ready\",\"sdk_available\":true,\"helper_version\":\""
            << kHelperVersion << "\"}" << std::endl;
  return 0;
}

int link_account(const std::string& input) {
  const uint64_t application_id = application_id_from(input);
  if (application_id == 0) {
    write_error("invalid_application_id", "Discord application ID is missing or invalid.");
    return 2;
  }

  auto client = std::make_shared<discordpp::Client>();
  client->SetApplicationId(application_id);

  std::atomic<bool> done{false};
  std::atomic<bool> connected{false};
  std::string failure;

  client->SetStatusChangedCallback([&](auto status, auto error, auto details) {
    if (status == discordpp::Client::Status::Ready) {
      connected = true;
      done = true;
    } else if (error != discordpp::Client::Error::None) {
      failure = discordpp::Client::ErrorToString(error) + " (" + std::to_string(details) + ")";
      done = true;
    }
  });

  auto verifier = client->CreateAuthorizationCodeVerifier();
  discordpp::AuthorizationArgs args{};
  args.SetClientId(application_id);
  args.SetScopes(discordpp::Client::GetDefaultPresenceScopes());
  args.SetCodeChallenge(verifier.Challenge());

  client->Authorize(args, [&, client, verifier, application_id](auto result, auto code, auto redirect_uri) {
    if (!result.Successful()) {
      failure = result.ToString();
      done = true;
      return;
    }
    client->GetToken(application_id, code, verifier.Verifier(), redirect_uri,
      [&, client](auto token_result, auto access_token, auto /*refresh_token*/, auto, auto, auto) {
        if (!token_result.Successful()) {
          failure = token_result.ToString();
          done = true;
          return;
        }
        client->UpdateToken(discordpp::AuthorizationTokenType::Bearer, access_token,
          [&, client](auto update_result) {
            if (!update_result.Successful()) {
              failure = update_result.ToString();
              done = true;
              return;
            }
            client->Connect();
          });
      });
  });

  if (!wait_until(done, std::chrono::seconds(170))) {
    client->AbortAuthorize();
    write_error("timeout", "Discord account linking timed out.");
    return 3;
  }
  if (!connected.load()) {
    write_error("authorization_failed", failure.empty() ? "Discord authorization failed." : failure);
    return 4;
  }

  const auto user = client->GetCurrentUserV2();
  if (!user.has_value()) {
    write_error("user_unavailable", "Discord connected but did not expose the current user.");
    return 5;
  }

  const auto avatar = user->AvatarUrl(discordpp::UserHandle::AvatarType::Gif, discordpp::UserHandle::AvatarType::Png);
  std::cout << "{\"ok\":true,\"state\":\"linked\",\"user\":{"
            << "\"id\":\"" << user->Id() << "\","
            << "\"username\":\"" << json_escape(user->Username()) << "\","
            << "\"display_name\":\"" << json_escape(user->DisplayName()) << "\","
            << "\"avatar_url\":\"" << json_escape(avatar) << "\"}}" << std::endl;
  return 0;
}

int set_activity(const std::string& input) {
  const uint64_t application_id = application_id_from(input);
  if (application_id == 0) {
    write_error("invalid_application_id", "Discord application ID is missing or invalid.");
    return 2;
  }

  auto client = std::make_shared<discordpp::Client>();
  client->SetApplicationId(application_id);
  discordpp::Activity activity{};
  activity.SetType(discordpp::ActivityTypes::Playing);

  const auto details = json_string(input, "details");
  const auto state = json_string(input, "state");
  const auto large_image = json_string(input, "large_image_key");
  const auto large_text = json_string(input, "large_image_text");
  if (details && details->size() >= 2) activity.SetDetails(*details);
  if (state && state->size() >= 2) activity.SetState(*state);
  if ((large_image && !large_image->empty()) || (large_text && large_text->size() >= 2)) {
    discordpp::ActivityAssets assets{};
    if (large_image && !large_image->empty()) assets.SetLargeImage(*large_image);
    if (large_text && large_text->size() >= 2) assets.SetLargeText(*large_text);
    activity.SetAssets(assets);
  }

  std::atomic<bool> done{false};
  std::string failure;
  client->UpdateRichPresence(std::move(activity), [&](auto result) {
    if (!result.Successful()) failure = result.ToString();
    done = true;
  });
  if (!wait_until(done, std::chrono::seconds(8))) {
    write_error("timeout", "Discord Rich Presence update timed out.");
    return 3;
  }
  if (!failure.empty()) {
    write_error("presence_failed", failure);
    return 4;
  }
  std::cout << "{\"ok\":true,\"state\":\"published\"}" << std::endl;
  return 0;
}

int clear_activity(const std::string& input) {
  const uint64_t application_id = application_id_from(input);
  if (application_id == 0) {
    write_error("invalid_application_id", "Discord application ID is missing or invalid.");
    return 2;
  }
  auto client = std::make_shared<discordpp::Client>();
  client->SetApplicationId(application_id);
  client->ClearRichPresence();
  for (int i = 0; i < 20; ++i) {
    discordpp::RunCallbacks();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "{\"ok\":true,\"state\":\"cleared\"}" << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    write_error("usage", "Expected --status, --link-account, --set-activity, or --clear-activity.");
    return 64;
  }
  const std::string command = argv[1];
  if (command == "--status") return status_command();
  const std::string input = read_stdin();
  if (command == "--link-account") return link_account(input);
  if (command == "--set-activity") return set_activity(input);
  if (command == "--clear-activity") return clear_activity(input);
  write_error("usage", "Unknown Discord helper command.");
  return 64;
}
