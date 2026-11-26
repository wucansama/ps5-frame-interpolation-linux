// Settings file: ~/.config/moonglide/config.toml (or --config PATH / MOONGLIDE_CONFIG).
// A small TOML subset: top-level `key = value` lines are the active settings, and
// `[preset."Name"]` sections hold named presets with the same keys. Keys are the command-line
// option names without "--" (e.g. `local-structure = 0.5`, `hud = ["rect:...", ...]`), so a
// file is turned into arguments and goes through the one option parser; arguments given on
// the command line override the file.
#pragma once
#include <map>
#include <string>
#include <vector>

using Settings = std::map<std::string, std::string>;   // key -> value (arrays joined with '\n')

struct ConfigFile {
  std::string path;
  Settings active;
  std::map<std::string, Settings> presets;
};

std::string defaultConfigPath();
// Missing file: false, `out` empty. Parse problems are reported and skipped.
bool loadConfig(const std::string& path, ConfigFile& out);
// Writes atomically (temp file + rename); creates the directory.
bool saveConfig(const ConfigFile& cfg);

// Settings -> command-line arguments ("--key", "value", "--no-nr", "--hud", ...). Keys in
// `skip` are left out (the command line sets them).
std::vector<std::string> settingsToArgs(const Settings& s, const std::vector<std::string>& skip = {});
// The settings key a command-line flag sets ("--no-nr" -> "nr", "--hud" -> "hud"); empty if
// the flag isn't a setting.
std::string settingKeyOfFlag(const std::string& flag);
