// Background helpers for the Source tab: Remote Play console discovery / pairing (they run the
// moonglide-chiaki-tap CLI, JSON lines on stdout) and V4L2 capture device / mode enumeration. All of
// it runs on worker threads; the UI only reads snapshots, so rendering never blocks.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using JsonFlat = std::map<std::string, std::string>;   // a flat JSON object: every value as text

// Parses one flat JSON object (strings, numbers, true/false/null; nested values are skipped).
bool parseJsonFlat(const std::string& line, JsonFlat& out);

// Runs argv (no shell), returns its stdout lines. exitCode -1: could not start / timed out.
// Nothing about the arguments is logged.
std::vector<std::string> runForLines(const std::vector<std::string>& argv, int timeoutMs, int* exitCode = nullptr,
                                     const std::string* stdinData = nullptr);
// xdg-open URL, detached.
void openInBrowser(const std::string& url);
// The moonglide-chiaki-tap binary: `configured` or ~/.local/share/moonglide/bin/moonglide-chiaki-tap.
std::string tapBinary(const std::string& configured);

class RpSetup {
 public:
  struct Console {
    std::string host, name, type, state, version, hostId;
    bool registered = false;
  };
  struct Snapshot {
    bool scanning = false;
    bool scanned = false;
    std::string scanError;
    std::vector<Console> consoles;
    bool loginBusy = false;
    std::string loginUrl;     // set when the sign-in page address came back (take it with takeLoginUrl)
    std::string loginError;
    bool accountBusy = false;
    std::string accountError;
    bool pairBusy = false;
    bool pairDone = false;
    bool pairOk = false;
    std::string pairMessage;
    std::string pairedHost;   // the address that was paired
  };
  RpSetup();
  ~RpSetup();
  Snapshot snapshot() const;
  void scan(const std::string& tap);
  void loginUrl(const std::string& tap);
  // Returns the sign-in url once (and clears it).
  std::string takeLoginUrl();
  // After the callback: ok -> account id + online id, else the error.
  void accountFromUrl(const std::string& tap, const std::string& url);
  // The result of accountFromUrl, taken once.
  bool takeAccount(std::string& accountId, std::string& onlineId);
  void pair(const std::string& tap, const std::string& host, const std::string& accountId, const std::string& pin, bool ps4);

 private:
  struct State;
  std::shared_ptr<State> st_;
};

class CaptureEnum {
 public:
  struct Device {
    std::string path, card;
  };
  struct Format {
    std::string key;                                  // mjpeg | yuyv | nv12
    std::vector<std::pair<uint32_t, uint32_t>> sizes; // descending area
    std::map<std::pair<uint32_t, uint32_t>, std::vector<int>> fps;   // per size, descending
  };
  struct Modes {
    bool done = false, ok = false;
    std::vector<Format> formats;
  };
  CaptureEnum();
  ~CaptureEnum();
  void refreshDevices();
  bool devicesReady() const;
  std::vector<Device> devices() const;
  // Starts the enumeration of `path` once (cached); returns what is known now.
  Modes modes(const std::string& path);
  void forget(const std::string& path);

 private:
  struct State;
  std::shared_ptr<State> st_;
};
