// The badge: a few lines of 5x7 bitmap text, drawn by present.comp. The host fills a small
// host-visible buffer of character codes; the font is a 128 x 8-row table of bit rows.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t kOverlayColumns = 64;
constexpr uint32_t kOverlayLines = 12;

// 128 glyphs x 8 uint32 rows (bits 4..0 = pixels left..right, rows 0..6).
std::vector<uint32_t> overlayFont();
// Packs lines into kOverlayLines x kOverlayColumns codes (space-padded, upper-cased).
void overlayPack(const std::vector<std::string>& lines, uint32_t* out);
