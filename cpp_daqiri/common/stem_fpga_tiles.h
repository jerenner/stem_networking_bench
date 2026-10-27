/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>

#include "stem_packet.h"

#ifdef __CUDACC__
#define STEM_FPGA_HD __host__ __device__
#else
#define STEM_FPGA_HD
#endif

namespace stem {

constexpr uint32_t FPGA_LINKS_PER_HALF = 30;
constexpr uint32_t FPGA_TILES_PER_LINK = 16;

struct FpgaTileHeader {
  uint16_t frame_number = 0;
  uint8_t tile_id = 0;
  uint8_t tile_slot = 0;
  bool valid = false;
};

STEM_FPGA_HD inline bool valid_fpga_tile_id(uint16_t id) {
  return id <= 255 && (id & 0x7fu) < FPGA_LINKS_PER_HALF;
}

STEM_FPGA_HD inline FpgaTileHeader decode_fpga_tile_header(const uint8_t* data) {
  FpgaTileHeader header;
  if (data == nullptr) { return header; }
  header.tile_id = data[2];
  header.tile_slot = data[3];
  header.frame_number = static_cast<uint16_t>(data[4]) |
                        (static_cast<uint16_t>(data[5]) << 8);
  header.valid = data[0] == 0x5a && data[1] == 0x5a &&
                 data[8] == 0xa5 && data[9] == 0xa5 &&
                 valid_fpga_tile_id(header.tile_id) &&
                 header.tile_slot < FPGA_TILES_PER_LINK;
  return header;
}

// Provisional physical map: 6 ZLP links then 24 CoreLoss links per half.
// ZLP slots default to four vertical tiles per readout, then the next readout.
// The alternative enumerates the four readouts before advancing vertically.
STEM_FPGA_HD inline uint16_t fpga_tile_index(uint16_t id, uint16_t slot,
                                           bool zlp_row_first = false) {
  if (!valid_fpga_tile_id(id) || slot >= FPGA_TILES_PER_LINK) {
    return static_cast<uint16_t>(FULL_FRAME_TILE_PACKETS);
  }
  const uint32_t half = (id >> 7) & 1u;
  const uint32_t link = id & 0x7fu;
  if (link < 6) {
    const uint32_t read = zlp_row_first ? slot % 4 : slot / 4;
    const uint32_t row = zlp_row_first ? slot / 4 : slot % 4;
    return static_cast<uint16_t>((half * 4 + row) * 24 + read * 6 + link);
  }
  return static_cast<uint16_t>(192 + (half * 16 + slot) * 24 + link - 6);
}

// Nearest-cycle unwrapping tolerates packet reordering and 65535 -> 0.
// Gaps of 32768 firmware frames or more require restarting/re-synchronizing.
class FpgaFrameCounter {
 public:
  int64_t unwrap(uint16_t value) {
    if (!initialized_) {
      reference_ = value;
      initialized_ = true;
      return reference_;
    }
    int32_t delta = static_cast<int32_t>(
        (static_cast<uint32_t>(value) -
         static_cast<uint32_t>(reference_ & 0xffff)) & 0xffffu);
    if (delta >= 32768) { delta -= 65536; }
    const int64_t candidate = reference_ + delta;
    if (candidate > reference_) { reference_ = candidate; }
    return candidate;
  }

 private:
  bool initialized_ = false;
  int64_t reference_ = 0;
};

}  // namespace stem

#undef STEM_FPGA_HD
