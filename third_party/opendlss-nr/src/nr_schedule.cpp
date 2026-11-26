#include "nr_schedule.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "nr_graph.h"

namespace nr {
namespace {

// The channels of the six levels, and the levels themselves: 0 is full resolution, 5 is the
// ViT bottleneck. `tools/nr_schedule.py` calls this table LEVELS.
int levelOf(uint32_t channels) {
  switch (channels) {
    case 32: return 0;
    case 64: return 1;
    case 128: return 2;
    case 256: return 3;
    case 512: return 4;
    case 1024: return 5;
  }
  throw std::runtime_error("no level for " + std::to_string(channels) + " channels");
}

const char* kindName(StageKind kind) {
  switch (kind) {
    case StageKind::Pre: return "pre";
    case StageKind::Level: return "level";
    case StageKind::Split: return "split";
    case StageKind::Vit: return "vit";
    case StageKind::Projection: return "projection";
    case StageKind::Post: return "post";
  }
  return "?";
}

struct Classified {
  std::string kind;   // "pre", "block", "encoder-transition", "decoder-transition", "split", ...
  uint32_t channels = 0;
  bool isSplit() const { return kind == "split" || kind == "split-512x1024"; }
};

bool sameSignature(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) { return a == b; }

std::vector<uint32_t> concat4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }

// The expected signature of every kind the port lays out. Mirrors `signatures()` in
// tools/nr_schedule.py; the formulas are the port's own `fusedLayout` so a layout change
// makes the classifier refuse the model instead of recording a graph from a stale table.
std::vector<std::pair<std::vector<uint32_t>, Classified>> signatureTable() {
  std::vector<std::pair<std::vector<uint32_t>, Classified>> table;
  auto add = [&](std::vector<uint32_t> lengths, const char* kind, uint32_t channels) {
    table.push_back({std::move(lengths), Classified{kind, channels}});
  };

  add({preFusedLayout().endWithoutPadding + 16}, "pre", 32);
  for (uint32_t c : {32u, 64u, 128u, 256u}) add({fusedLayout(c).endWithoutPadding + 16}, "block", c);
  // Encoder transition: the stage's last block also carries the 2C x C upsample matrix.
  for (uint32_t c : {32u, 64u, 128u, 256u})
    add({fusedLayout(c).endWithoutPadding + 2 * c * c + (c == 32 ? 16u : 0u)}, "encoder-transition", c);
  // Decoder transition: the stage's first block carries the 2C -> C upsample; the 32-channel
  // one is 16 bytes wider for the narrow-buffer padding.
  for (auto [c, extra] : std::vector<std::pair<uint32_t, uint32_t>>{{32, 80}, {64, 112}, {128, 240}, {256, 496}})
    add({fusedLayout(c).endWithoutPadding + 2 * c * c + extra}, "decoder-transition", c);
  const std::vector<uint32_t> split = concat4(524288, 263168, 917568, 263168);
  add(split, "split", 512);
  {
    std::vector<uint32_t> wider = split;
    wider.push_back(524304);   // also carries the 512 -> 1024 matrix
    add(std::move(wider), "split-512x1024", 512);
  }
  add({4194320, 4196352, 3145856, 2, 1050624}, "vit", 1024);
  add({525312}, "projection", 512);   // block 39: the 1024 -> 512 matrix plus its skip scale
  add({postFusedLayout().endWithoutPadding}, "post", 32);
  return table;
}

Classified classify(const std::vector<uint32_t>& signature) {
  static const auto table = signatureTable();
  for (const auto& [lengths, classified] : table)
    if (sameSignature(lengths, signature)) return classified;
  std::string joined;
  for (uint32_t length : signature) joined += (joined.empty() ? "" : ",") + std::to_string(length);
  throw std::runtime_error("unrecognised block signature (" + std::to_string(signature.size()) + " layers: " + joined + ")");
}

}  // namespace

const Stage* Schedule::find(const std::string& name) const {
  for (const Stage& entry : stages)
    if (entry.name == name) return &entry;
  return nullptr;
}

const Stage& Schedule::stage(const std::string& name) const {
  const Stage* found = find(name);
  if (!found) throw std::runtime_error("the model's schedule has no stage " + name);
  return *found;
}

const Stage& Schedule::ofKind(StageKind kind, uint32_t channels) const {
  for (const Stage& entry : stages)
    if (entry.kind == kind && entry.channels == channels) return entry;
  throw std::runtime_error("the model's schedule has no " + std::string(kindName(kind)) + "-" +
                           std::to_string(channels) + " stage");
}

uint32_t Schedule::vitFirst() const { return (uint32_t)ofKind(StageKind::Vit, 1024).first; }
uint32_t Schedule::vitLast() const { return (uint32_t)ofKind(StageKind::Vit, 1024).last; }

Schedule deriveSchedule(const std::vector<BlockSignature>& blocks) {
  if (blocks.empty()) throw std::runtime_error("the model has no blocks");
  std::vector<Classified> kinds;
  kinds.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    try {
      kinds.push_back(classify(blocks[i].layerBytes));
    } catch (const std::exception& error) {
      throw std::runtime_error("block " + std::to_string(i) + ": " + error.what());
    }
  }

  Schedule schedule;
  schedule.blockCount = (uint32_t)blocks.size();
  bool encoder = true;
  size_t index = 0;
  while (index < blocks.size()) {
    const int block = (int)index;
    const Classified here = kinds[index];
    if (here.kind == "pre") {
      schedule.stages.push_back({"pre", 32, block, block, levelOf(32), StageKind::Pre, true, "pre"});
      ++index;
      continue;
    }
    if (here.kind == "post") {
      schedule.stages.push_back({"post", 32, block, block, levelOf(32), StageKind::Post, false, "post"});
      ++index;
      continue;
    }
    if (here.kind == "projection") {
      schedule.stages.push_back({"transition-38-39", 512, block, block, levelOf(512), StageKind::Projection, false,
                                 "projection"});
      encoder = false;
      ++index;
      continue;
    }
    if (here.kind == "vit") {
      const int first = block;
      while (index < blocks.size() && kinds[index].kind == "vit") ++index;
      schedule.stages.push_back({"vit", 1024, first, (int)index - 1, 5, StageKind::Vit, false, "vit"});
      continue;
    }
    if (here.isSplit()) {
      const int first = block;
      while (index < blocks.size() && kinds[index].isSplit()) ++index;
      const std::string lastKind = kinds[index - 1].kind;
      const bool encoderSide = encoder && lastKind == "split-512x1024";
      schedule.stages.push_back({std::string(encoderSide ? "encoder" : "decoder") + "-512", 512, first,
                                 (int)index - 1, levelOf(512), StageKind::Split, encoderSide,
                                 lastKind == "split-512x1024" ? "split-512x1024" : "split"});
      continue;
    }
    // A U-Net level: blocks up to and including the transition block. On the decoder side the
    // transition block opens the stage and the count is one fewer per level.
    const uint32_t channels = here.channels;
    const int first = block;
    while (index < blocks.size()) {
      const Classified& next = kinds[index];
      if (next.channels != channels || (next.kind != "block" && next.kind != "encoder-transition" &&
                                        next.kind != "decoder-transition"))
        break;
      const std::string kindHere = next.kind;
      ++index;
      if (kindHere == "encoder-transition")
        break;   // the encoder stage owns its transition; a decoder's opens it and keeps going
    }
    const int last = (int)index - 1;
    const std::string transition = kinds[index - 1].kind;
    if (kinds[first].kind == "decoder-transition") encoder = false;
    schedule.stages.push_back({std::string(encoder ? "encoder" : "decoder") + "-" + std::to_string(channels), channels,
                               first, last, levelOf(channels), StageKind::Level, encoder, transition});
  }
  return schedule;
}

namespace {
std::string readText(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::stringstream stream;
  stream << file.rdbuf();
  return stream.str();
}
}  // namespace

Schedule readSchedule(const std::string& directory, uint32_t* blockCountOut) {
  json::Value manifest = json::parse(readText(directory + "/manifest.json"));
  const uint32_t blockCount = (uint32_t)manifest["totals"]["blockCount"].integer();
  // The manifest names a layer "layer0", "layer1", ...; the reference sorts on that string, so this
  // does too (sorting on an integer parse of it silently collapses every layer to 0).
  std::vector<std::vector<std::pair<std::string, uint32_t>>> layers(blockCount);
  for (const json::Value& entry : manifest["tensors"].array) {
    if (entry["parameter"].str() != "layer") continue;
    const auto block = (size_t)entry["block"].integer();
    if (block >= layers.size()) continue;   // not a block body tensor
    layers[block].push_back({entry["layer"].str(), (uint32_t)entry["byteLength"].integer()});
  }
  std::vector<BlockSignature> blocks(blockCount);
  for (size_t block = 0; block < layers.size(); ++block) {
    std::sort(layers[block].begin(), layers[block].end());
    for (const auto& [layer, length] : layers[block]) blocks[block].layerBytes.push_back(length);
  }
  if (blockCountOut) *blockCountOut = blockCount;
  return deriveSchedule(blocks);
}

std::string renderSchedule(const Schedule& schedule) {
  std::string out = std::to_string(schedule.blockCount) + " blocks\n";
  char line[160];
  snprintf(line, sizeof(line), "%-20s%6s%7s%6s%7s  %s\n", "stage", "ch", "first", "last", "level", "kind");
  out += line;
  for (const Stage& entry : schedule.stages) {
    snprintf(line, sizeof(line), "%-20s%6u%7d%6d%7d  %s\n", entry.name.c_str(), entry.channels, entry.first,
             entry.last, entry.level, entry.variant.c_str());
    out += line;
  }
  return out;
}

}  // namespace nr
