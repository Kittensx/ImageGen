param(
  [Parameter(Mandatory = $true)]
  [string]$SdkRoot,
  [ValidateSet("Debug", "Release", "RelWithDebInfo")]
  [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SourceDir = Join-Path $ProjectRoot "src\image_gen\discord_native\helper"
$BuildDir = Join-Path $ProjectRoot "artifacts\build\discord-helper"
$OutputDir = Join-Path $ProjectRoot "app\discord"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

cmake -S $SourceDir -B $BuildDir -DDISCORD_SDK_ROOT="$SdkRoot"
cmake --build $BuildDir --config $Configuration

$Executable = Get-ChildItem -Path $BuildDir -Recurse -Filter "imagegen_discord_helper.exe" | Select-Object -First 1
$DiscordDll = Get-ChildItem -Path $BuildDir -Recurse -Filter "discord_partner_sdk.dll" | Select-Object -First 1
if (-not $Executable) { throw "imagegen_discord_helper.exe was not produced." }
if (-not $DiscordDll) { throw "discord_partner_sdk.dll was not copied next to the helper." }

Copy-Item -Force $Executable.FullName (Join-Path $OutputDir "imagegen_discord_helper.exe")
Copy-Item -Force $DiscordDll.FullName (Join-Path $OutputDir "discord_partner_sdk.dll")
Write-Host "Discord helper installed to $OutputDir"
