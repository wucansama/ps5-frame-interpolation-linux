// The NR graph's stage schedule, derived from the model instead of hard-coded.
//
// The graph used to walk six literal block ranges (`{5,8}`, `9..14`, `15..22`, `23..30`,
// the ViT chain `31..38` and its `block < 38` chain constant, `40..47`) plus four decoder
// stages keyed on `48`, `56`, `62`, `66`. Those numbers are the 71-block model's block
// counts per U-Net level, not the architecture: every block's tensor is laid out by
// `fusedLayout(channels)` and the kinds repeat. `tools/nr_schedule.py` is the reference
// implementation of that classification; this is the port's mirror of it, so the recorder
// walks a table derived from whatever model it was handed.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace nr {

enum class StageKind { Pre, Level, Split, Vit, Projection, Post };

struct Stage {
  std::string name;     // "pre", "encoder-32", ..., "vit", "transition-38-39", "decoder-32", "post"
  uint32_t channels = 0;
  int first = 0, last = 0;
  int level = 0;        // Geometry::levels index; 5 is the ViT bottleneck, 6 is un-pooled
  StageKind kind = StageKind::Level;
  bool encoder = true;  // direction of a U-Net level stage
  std::string variant;  // the classified kind: "block", "split", "vit", ...
};

// One block's body signature: the byte lengths of its `layer` tensors in layer order.
struct BlockSignature {
  std::vector<uint32_t> layerBytes;
};

struct Schedule {
  uint32_t blockCount = 0;
  std::vector<Stage> stages;

  const Stage& stage(const std::string& name) const;   // throws when the model has no such stage
  const Stage* find(const std::string& name) const;
  const Stage& ofKind(StageKind kind, uint32_t channels) const;  // the single stage of that kind/channels
  uint32_t vitLast() const;                                          // last ViT block (the chain constant)
  uint32_t vitFirst() const;
};

// Classify one block by its tensors' byte lengths and walk the blocks into stage order:
// encoder levels, bottleneck, decoder levels, post. Throws when a block's signature is not
// one the port can lay out - a wrong schedule must fail at startup, not record the wrong graph.
Schedule deriveSchedule(const std::vector<BlockSignature>& blocks);

// Device-free: read `directory/manifest.json` and return the block signatures it describes.
Schedule readSchedule(const std::string& directory, uint32_t* blockCountOut = nullptr);

// The table as text, the same columns `tools/nr_schedule.py` prints, so the two can be diffed.
std::string renderSchedule(const Schedule& schedule);

}  // namespace nr
