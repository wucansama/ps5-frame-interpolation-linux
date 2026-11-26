#include "jpeg_decode.h"

#include <csetjmp>
#include <algorithm>
#include <cstdio>
// clang-format off
#include <jpeglib.h>
#include <jerror.h>
// clang-format on

namespace {
struct JpegError {
  jpeg_error_mgr mgr;   // first member: libjpeg hands back a pointer to it
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};
void onError(j_common_ptr cinfo) {
  auto* e = reinterpret_cast<JpegError*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, e->message);
  longjmp(e->jump, 1);
}
}  // namespace

struct JpegDecoder::Impl {
  jpeg_decompress_struct cinfo{};
  JpegError err{};
};
static void onMessage(j_common_ptr, int) {}   // corrupt-data warnings: frames still decode

JpegDecoder::JpegDecoder() : impl_(new Impl) {
  impl_->cinfo.err = jpeg_std_error(&impl_->err.mgr);
  impl_->err.mgr.error_exit = onError;
  impl_->err.mgr.emit_message = onMessage;
  jpeg_create_decompress(&impl_->cinfo);
}

JpegDecoder::~JpegDecoder() {
  jpeg_destroy_decompress(&impl_->cinfo);
  delete impl_;
}

bool JpegDecoder::decode(const uint8_t* data, size_t bytes, uint8_t* rgba, uint32_t width, uint32_t height) {
  jpeg_decompress_struct& c = impl_->cinfo;
  if (setjmp(impl_->err.jump)) {
    error_ = impl_->err.message;
    jpeg_abort_decompress(&c);
    return false;
  }
  jpeg_mem_src(&c, data, (unsigned long)bytes);
  jpeg_read_header(&c, TRUE);
  c.out_color_space = JCS_EXT_RGBA;
  if (fast) {
    c.dct_method = JDCT_IFAST;
    c.do_fancy_upsampling = FALSE;
  }
  jpeg_start_decompress(&c);
  if (c.output_width != width || c.output_height != height) {
    error_ = "jpeg is " + std::to_string(c.output_width) + "x" + std::to_string(c.output_height);
    jpeg_abort_decompress(&c);
    return false;
  }
  const size_t stride = (size_t)width * 4;
  JSAMPROW rows[16];
  while (c.output_scanline < c.output_height) {
    const uint32_t n = std::min<uint32_t>(16, c.output_height - c.output_scanline);
    for (uint32_t i = 0; i < n; ++i) rows[i] = rgba + (size_t)(c.output_scanline + i) * stride;
    jpeg_read_scanlines(&c, rows, n);
  }
  jpeg_finish_decompress(&c);
  return true;
}
