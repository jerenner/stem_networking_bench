/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "stem_packet.h"

namespace stem {

struct ArrivalOrderPosition {
  uint64_t frame;
  uint16_t tile;
  uint16_t source;
  uint16_t row_offset;
};

// Diagnostic positions depend only on packets observed on this receiver.
// Source IDs are virtual; the mask controls the amount of work per frame.
class ArrivalOrderSequence {
 public:
  explicit ArrivalOrderSequence(uint32_t source_mask) {
    if (source_mask == 0 || (source_mask & ~0xffu) != 0) {
      throw std::runtime_error("arrival-order source mask must be in [1, 255]");
    }
    for (uint16_t source = 0; source < NUM_SOURCES_MAX; ++source) {
      if ((source_mask >> source) & 1u) { sources_[source_count_++] = source; }
    }
    packets_per_frame_ = source_count_ * TILE_PACKETS_PER_SOURCE;
  }

  uint32_t packets_per_frame() const { return packets_per_frame_; }

  ArrivalOrderPosition next() {
    const uint64_t frame = packet_count_ / packets_per_frame_;
    const auto tile = static_cast<uint16_t>(packet_count_ % packets_per_frame_);
    ++packet_count_;
    return {frame, tile, sources_[tile / TILE_PACKETS_PER_SOURCE],
            static_cast<uint16_t>(tile % TILE_PACKETS_PER_SOURCE)};
  }

 private:
  std::array<uint16_t, NUM_SOURCES_MAX> sources_{};
  uint32_t source_count_ = 0;
  uint32_t packets_per_frame_ = 0;
  uint64_t packet_count_ = 0;
};

}  // namespace stem
