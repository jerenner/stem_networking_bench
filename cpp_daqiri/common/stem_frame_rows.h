/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>

#include "stem_packet.h"

#ifdef __CUDACC__
#define STEM_ROW_HD __host__ __device__
#else
#define STEM_ROW_HD
#endif

namespace stem {

constexpr uint32_t DETECTOR_REFERENCE_ROWS = 32;  // Pattern + meta + 30 BLR.
constexpr uint32_t DETECTOR_HALF_ROWS = FRAME_HEIGHT / 2;
constexpr uint32_t DETECTOR_ROW_LAYOUT_VERSION = 2;
static_assert(FRAME_HEIGHT == 1024, "Detector row flips require the 1024-row layout");

struct FrameRowFlips {
  bool flip_top_imaging_rows = false;
  bool flip_bottom_reference_rows = false;
  // Version 1 only reversed rows 992..1023. Version 2 relocates the entire
  // bottom stream, moving its leading references to the detector edge.
  uint32_t layout_version = DETECTOR_ROW_LAYOUT_VERSION;

  STEM_ROW_HD constexpr bool enabled() const {
    return flip_top_imaging_rows || flip_bottom_reference_rows;
  }

  constexpr bool operator==(const FrameRowFlips& other) const {
    return flip_top_imaging_rows == other.flip_top_imaging_rows &&
           flip_bottom_reference_rows == other.flip_bottom_reference_rows &&
           (!flip_bottom_reference_rows || layout_version == other.layout_version);
  }
  constexpr bool operator!=(const FrameRowFlips& other) const {
    return !(*this == other);
  }
};

// Native FPGA tile headers identify this fixed detector layout. These are
// layout/provenance bits, not user-selectable receiver corrections.
constexpr FrameRowFlips fpga_detector_row_flips() { return {true, true}; }

// Global stream-to-sensor permutation, not a local flip within each tile.
// Both boards send pattern, meta, BLR, then centre-to-edge imaging rows.
STEM_ROW_HD inline uint32_t orient_detector_row(uint32_t row,
                                               FrameRowFlips flips) {
  if (flips.flip_top_imaging_rows && row >= DETECTOR_REFERENCE_ROWS &&
      row < DETECTOR_HALF_ROWS) {
    return DETECTOR_REFERENCE_ROWS + DETECTOR_HALF_ROWS - 1 - row;
  }
  if (flips.flip_bottom_reference_rows &&
      row >= DETECTOR_HALF_ROWS && row < FRAME_HEIGHT) {
    if (row < DETECTOR_HALF_ROWS + DETECTOR_REFERENCE_ROWS) {
      return FRAME_HEIGHT + DETECTOR_HALF_ROWS - 1 - row;
    }
    return row - DETECTOR_REFERENCE_ROWS;
  }
  return row;
}

}  // namespace stem

#undef STEM_ROW_HD
