// MacroSilicon MS2109 (USB 534d:2109) HDMI dongle: live MJPEG encoder quality.
//
// The firmware loads its q~75 quantisation tables into the encoder on every stream start; the
// HID "factory" interface can overwrite them while streaming: XDATA 0xF526 = quant RAM index
// (write 0), 0xF527 = data port (auto-increment), 128 bytes (64 luma + 64 chroma, zigzag, the
// ROM's chroma layout). RAM only, never the EEPROM. Report format as ms-tools/mshal: feature
// report 0, [0xb6, addr hi, addr lo, value] (XDATA write), echoed back on the get.
// q90 is stable at 30 fps on USB2; q95 overflows the link.
#pragma once
#include <string>

// The hidraw node on the same USB device as this V4L2 node, if it is an MS2109 ("" if not).
std::string ms2109Hidraw(const std::string& videoDevice);
// quality 1..100 (IJG scaling), or 0 for the firmware's own q~75 table. Throws on failure.
void ms2109SetJpegQuality(const std::string& hidraw, int quality);
