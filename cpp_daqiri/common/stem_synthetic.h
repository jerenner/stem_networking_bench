/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "stem_packet.h"

#ifdef __CUDACC__
#define STEM_SYNTH_HD __host__ __device__
#else
#define STEM_SYNTH_HD
#endif

namespace stem {

enum class SyntheticPattern : uint32_t { kRamp, kWalkingDot };

STEM_SYNTH_HD inline void synthetic_le16(uint8_t* dst, uint16_t value) {
  dst[0] = value & 255u;
  dst[1] = value >> 8;
}
STEM_SYNTH_HD inline void synthetic_be16(uint8_t* dst, uint16_t value) {
  dst[0] = value >> 8;
  dst[1] = value & 255u;
}

struct SyntheticLayout {
  uint32_t source_mask = 255;
  bool legacy_payload = false;
  uint32_t receiver = 0;
  SyntheticPattern pattern = SyntheticPattern::kRamp;

  STEM_SYNTH_HD uint32_t source_count() const {
    uint32_t n = 0;
    for (uint32_t s = 0; s < 8; ++s) { n += (source_mask >> s) & 1u; }
    return n;
  }
  STEM_SYNTH_HD uint32_t rows_per_source() const {
    return legacy_payload ? ROWS_PER_SOURCE : TILE_PACKETS_PER_SOURCE;
  }
  STEM_SYNTH_HD uint32_t packets_per_frame() const {
    return source_count() * rows_per_source();
  }
  STEM_SYNTH_HD uint32_t cycle_packets() const {
    return FRAMES_PER_WRAP * packets_per_frame();
  }
  STEM_SYNTH_HD uint32_t payload_bytes() const {
    return legacy_payload ? STEM_PAYLOAD_SIZE : TILE_PAYLOAD_BYTES;
  }
  STEM_SYNTH_HD uint32_t packet_bytes() const {
    return L4_HEADER_SIZE + STEM_HEADER_SIZE + payload_bytes();
  }
  STEM_SYNTH_HD uint32_t stride_bytes() const {
    return (packet_bytes() + 63u) & ~63u;
  }
  STEM_SYNTH_HD uint32_t source_id(uint32_t ordinal) const {
    for (uint32_t s = 0; s < 8; ++s) {
      if ((source_mask >> s) & 1u) {
        if (ordinal == 0) { return s; }
        --ordinal;
      }
    }
    return 0xffffu;
  }
  STEM_SYNTH_HD uint32_t frame(uint32_t packet) const {
    return packet / packets_per_frame();
  }
  STEM_SYNTH_HD uint32_t row_offset(uint32_t packet) const {
    return (packet % packets_per_frame()) / source_count();
  }
  STEM_SYNTH_HD uint32_t ordinal(uint32_t packet) const {
    return packet % source_count();
  }
  STEM_SYNTH_HD uint32_t tile(uint32_t packet) const {
    return ordinal(packet) * TILE_PACKETS_PER_SOURCE + row_offset(packet);
  }
  STEM_SYNTH_HD void write_header(uint8_t* p, uint32_t packet) const {
    for (uint32_t i = 0; i < L4_HEADER_SIZE + STEM_HEADER_SIZE; ++i) { p[i] = 0; }
    p[0] = 2; p[5] = static_cast<uint8_t>(receiver);
    p[6] = 2; p[11] = 1;
    synthetic_be16(p + 12, 0x0800);  // Ethernet II / IPv4 / UDP, no VLAN/options.
    p[14] = 0x45;
    synthetic_be16(p + 16, packet_bytes() - 14);
    p[22] = 64; p[23] = 17;
    p[26] = 10; p[29] = 1;
    p[30] = 10; p[32] = 1; p[33] = static_cast<uint8_t>(receiver + 1);
    uint32_t checksum = 0;
    for (uint32_t i = 14; i < 34; i += 2) { checksum += (p[i] << 8) | p[i + 1]; }
    while (checksum >> 16) { checksum = (checksum & 65535u) + (checksum >> 16); }
    synthetic_be16(p + 24, static_cast<uint16_t>(~checksum));
    synthetic_be16(p + 34, 23130);
    synthetic_be16(p + 36, 23130);
    synthetic_be16(p + 38, packet_bytes() - 34);
    uint8_t* header = p + L4_HEADER_SIZE;
    synthetic_le16(header, 0x5a5a);
    synthetic_le16(header + 2, frame(packet));
    synthetic_le16(header + 4, frame(packet) * ROWS_PER_SOURCE + row_offset(packet));
    synthetic_le16(header + 6, source_id(ordinal(packet)));
    synthetic_le16(header + 8, 0xa5a5);
    for (uint32_t i = 10; i < STEM_HEADER_SIZE; i += 2) {
      synthetic_le16(header + i, frame(packet));
    }
  }
  STEM_SYNTH_HD uint16_t sample(uint32_t frame_index, uint32_t tile_index,
                               uint32_t sample_index) const {
    if (pattern == SyntheticPattern::kWalkingDot) {
      const uint32_t dot =
          (frame_index * 31u + tile_index * 7u + receiver * 13u) % TILE_SAMPLES;
      return sample_index == dot ? 20000u : static_cast<uint16_t>(100u + receiver);
    }
    return static_cast<uint16_t>(64u +
        ((receiver * 977u + frame_index * 37u + tile_index * 17u +
          sample_index * 5u) & 4095u));
  }
};

struct SyntheticConfig {
  SyntheticLayout layout;
  bool maximum_rate = false;
  double target_gbps = 100.0;
  double duration_seconds = 30.0;
  uint64_t buckets_per_receiver = 0;  // Optional deterministic stop condition.
  uint32_t packets_per_burst = 16384;
  uint32_t frames_per_tensor = 128;
  uint32_t gpu_device = 0;
  bool validate_output = false;
  double report_interval_seconds = 1.0;

  void validate() const {
    if (layout.source_mask == 0 || layout.source_mask > 255) {
      throw std::runtime_error("synthetic source mask must be in [1, 255]");
    }
    // Existing relative_frame is uint8 and wrapping uses a 128-frame cycle.
    if (frames_per_tensor == 0 || frames_per_tensor > FRAMES_PER_WRAP) {
      throw std::runtime_error("synthetic frames_per_tensor must be in [1, 128]");
    }
    if (packets_per_burst == 0 ||
        packets_per_burst > 32u * layout.packets_per_frame()) {
      throw std::runtime_error(
          "synthetic packets_per_burst must be positive and span at most 32 frames");
    }
    if (!std::isfinite(target_gbps) || target_gbps <= 0 || target_gbps > 1000000) {
      throw std::runtime_error("synthetic target_gbps must be finite and in (0, 1000000]");
    }
    if (!std::isfinite(duration_seconds) ||
        (duration_seconds < 0 && duration_seconds != -1) ||
        duration_seconds > 604800) {
      throw std::runtime_error("synthetic duration_seconds must be -1 or in [0, 604800]");
    }
    if (buckets_per_receiver > 1000000000ULL) {
      throw std::runtime_error("synthetic buckets_per_receiver exceeds 1000000000");
    }
    if (!std::isfinite(report_interval_seconds) || report_interval_seconds < 0.01) {
      throw std::runtime_error("synthetic report_interval_seconds must be at least 0.01");
    }
    if (gpu_device != 0) {
      throw std::runtime_error(
          "synthetic currently uses CUDA device 0; select the physical GPU with CUDA_VISIBLE_DEVICES");
    }
  }

  double scheduled_seconds(uint64_t packets) const {
    return static_cast<double>(packets) * layout.packet_bytes() * 8.0 /
           (target_gbps * 1e9);
  }
  // The producer clock never resets when a consumer falls behind. This is
  // virtual schedule debt, not an allocated queue or a NIC loss counter.
  double lag_seconds(uint64_t packets, double elapsed) const {
    return maximum_rate ? 0.0 : std::max(0.0, elapsed - scheduled_seconds(packets));
  }
};

}  // namespace stem
#undef STEM_SYNTH_HD
