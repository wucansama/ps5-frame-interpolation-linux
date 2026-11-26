// Remote Play frame tap: the shared-memory protocol between moonglide-chiaki-tap (a separate,
// AGPL process built on chiaki-ng's libchiaki, tools/chiaki-tap/) and moonglide (MIT). The tap
// writes every compressed video frame the PS5 sends; moonglide decodes it (NVDEC), runs its own
// pipeline and tells the tap how long to hold the sound back so it meets the picture.
//
// Modelled on SmoovPlay's frame tap, layout version 1 (docs/frame-tap.md in
// github.com/smoovplay/smoovplay, MIT), ported to Linux:
//   - POSIX shared memory: shm_open(name) -> /dev/shm/<name>, kRpTapSize bytes.
//   - "Set the event" is a futex: the writer bumps `futex` and FUTEX_WAKEs it.
//   - Times are CLOCK_MONOTONIC nanoseconds (qpc_freq = 1e9).
//   - The stream's codec header (VPS/SPS/PPS, sent once by the PS5 as the stream's first
//     packet) is kept in the header too, so a reader that starts mid-stream can decode.
//
// This header is MIT; it carries only the layout, no chiaki-ng code.
#pragma once
#include <stdint.h>

#define RP_TAP_DEFAULT_NAME "moonglide-remoteplay"
#define RP_TAP_MAGIC 0x50544643u   /* 'CFTP' */
#define RP_TAP_VERSION 1u
#define RP_TAP_HEADER_SIZE 4096u
#define RP_TAP_SLOT_HEADER 64u
#define RP_TAP_SLOT_COUNT 32u
#define RP_TAP_SLOT_SIZE (2u * 1024u * 1024u)
#define RP_TAP_SIZE (RP_TAP_HEADER_SIZE + RP_TAP_SLOT_COUNT * RP_TAP_SLOT_SIZE)
#define RP_TAP_CODEC_HEADER_MAX 3000u

enum RpTapCodec { RP_TAP_H264 = 0, RP_TAP_H265 = 1, RP_TAP_H265_HDR = 2 };
enum RpTapFeature {
  RP_TAP_FEATURE_AUDIO_DELAY = 2u,   // honours audio_delay_ms
};
enum RpTapState {
  RP_TAP_IDLE = 0,         // tap running, no session
  RP_TAP_WAKING = 1,       // waking the console from rest mode
  RP_TAP_CONNECTING = 2,
  RP_TAP_STREAMING = 3,
  RP_TAP_QUIT = 4,         // session ended; `status` says why
  RP_TAP_STANDBY = 5,      // the console is in rest mode; the tap waits for wake_request (or for it to wake)
};
enum RpTapSlotFlag {
  RP_TAP_SLOT_TRUNCATED = 1u,   // did not fit the slot
};

// Camera-stick onset (input-lag measurement while playing): the tap watches the stick the console
// gets (the physical pads). When the right stick goes from neutral
// (|v| < 0.1 for at least 300 ms) to deflected (|v| > 0.5) it records the moment it sent that state
// to the console. The app compares it with the first frame in which the picture starts to pan.
// Single writer (the tap): x, y, onset_ns first, then `seq` is bumped with a release store.
#define RP_TAP_STICK_OFFSET 320u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RpTapStick {
  uint64_t onset_ns;   // 0  CLOCK_MONOTONIC of the state that deflected the stick
  uint32_t seq;        // 8  bumped with every onset
  int16_t x, y;        // 12 the stick then (-32768..32767, y positive = down)
} RpTapStick;

// Header: the first 4096 bytes. Offsets 0..95 match SmoovPlay's v1 header.
typedef struct RpTapHeader {
  uint32_t magic;              // 0   RP_TAP_MAGIC while a stream is on; 0 while (re)written / no stream
  uint32_t version;            // 4   RP_TAP_VERSION
  uint32_t slot_count;         // 8
  uint32_t slot_size;          // 12
  uint32_t codec;              // 16  RpTapCodec
  uint32_t width;              // 20
  uint32_t height;             // 24
  uint32_t max_fps;            // 28
  uint64_t write_seq;          // 32  frames written so far; bumped after the slot is complete
  uint32_t idr_request;        // 40  reader sets 1; writer clears it and asks the PS5 for a keyframe
  uint32_t session;            // 44  bumped at every stream start
  uint64_t qpc_freq;           // 48  1e9 (CLOCK_MONOTONIC ns)
  uint64_t reader_heartbeat;   // 56  reader: CLOCK_MONOTONIC ns, refreshed while it reads
  uint32_t features;           // 64  RpTapFeature bits
  uint32_t rtt_us;             // 68  writer: network round trip measured at session start (0: unknown)
  uint64_t input_ns;           // 72  writer: CLOCK_MONOTONIC of the last controller button press sent to the console
  uint32_t input_seq;          // 80  writer: bumped with every such press (input-lag measurement)
  uint32_t reserved84;         // 84
  uint32_t audio_delay_ms;     // 88  reader: hold the sound back this long (0-500); ignored once the
                               //     heartbeat is over 2 s old
  uint32_t colour;             // 92  (decoded frames only in SmoovPlay; 0)
  // Linux additions:
  uint32_t futex;              // 96  bumped after every frame and state change, FUTEX_WAKE'd
  uint32_t state;              // 100 RpTapState
  uint32_t writer_pid;         // 104
  uint32_t audio_out_ms;       // 108 writer: the sound device's own latency (informational)
  uint32_t audio_held_ms;      // 112 writer: how much it is holding the sound back right now
  uint32_t frames_lost_total;  // 116 writer: frames the network lost this session
  uint32_t bitrate_kbps;       // 120 the session's requested bitrate
  uint32_t wake_request;       // 124 reader sets 1; writer clears it and wakes the console from rest mode
  char status[128];            // 128 a line for humans ("streaming", the quit reason, ...)
  uint8_t pad256[1024 - 256];
  uint32_t codec_header_size;  // 1024 bytes in codec_header (0: none yet)
  uint32_t codec_header_session; // 1028 the `session` it belongs to
  uint8_t codec_header[RP_TAP_CODEC_HEADER_MAX];   // 1032 Annex B VPS/SPS/PPS
} RpTapHeader;

// Slot i at RP_TAP_HEADER_SIZE + i * slot_size; frame n goes in slot n % slot_count.
typedef struct RpTapSlot {
  uint64_t seq;            // 0  the frame's number; all ones while the slot is rewritten
  uint64_t arrival_ns;     // 8  CLOCK_MONOTONIC when the frame was complete on this PC
  uint32_t size;           // 16 bytes of data
  int32_t frames_lost;     // 20 frames lost just before this one
  uint32_t recovered;      // 24 1: first frame after a loss that repairs the picture
  uint32_t flags;          // 28 RpTapSlotFlag
  uint8_t reserved[32];    // 32
  // 64: data[size], Annex B
} RpTapSlot;

#ifdef __cplusplus
}
static_assert(sizeof(RpTapHeader) <= RP_TAP_HEADER_SIZE, "rp tap header too big");
static_assert(sizeof(RpTapSlot) == RP_TAP_SLOT_HEADER, "rp tap slot header");
static_assert(sizeof(RpTapStick) == 16 && RP_TAP_STICK_OFFSET + sizeof(RpTapStick) <= 1024, "rp tap stick block");
static_assert(__builtin_offsetof(RpTapHeader, input_ns) == 72, "rp tap layout");
static_assert(__builtin_offsetof(RpTapHeader, audio_delay_ms) == 88, "rp tap layout");
static_assert(__builtin_offsetof(RpTapHeader, futex) == 96, "rp tap layout");
static_assert(__builtin_offsetof(RpTapHeader, codec_header_size) == 1024, "rp tap layout");
#endif
