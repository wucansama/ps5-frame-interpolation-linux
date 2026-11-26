// MJPEG -> RGBA8 with libjpeg-turbo (SIMD). UVC MJPEG often omits the Huffman tables;
// libjpeg-turbo >= 2.0 falls back to the standard ones, which is what those streams assume.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

class JpegDecoder {
 public:
  JpegDecoder();
  ~JpegDecoder();
  // Decodes into `rgba` (width*height*4, rows of `width`). Returns false (and sets error())
  // on a corrupt frame or a size mismatch.
  bool decode(const uint8_t* data, size_t bytes, uint8_t* rgba, uint32_t width, uint32_t height);
  const std::string& error() const { return error_; }
  bool fast = true;   // IFAST DCT + plain upsampling: ~30% faster, invisible after NR

 private:
  struct Impl;
  Impl* impl_;
  std::string error_;
};
