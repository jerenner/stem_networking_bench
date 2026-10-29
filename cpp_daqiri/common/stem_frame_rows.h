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
static_assert(FRAME_HEIGHT == 1024, "Detector row flips require the 1024-row layout");

struct FrameRowFlips {
  bool flip_top_imaging_rows = false;
  bool flip_bottom_reference_rows = false;

  STEM_ROW_HD constexpr bool enabled() const {
    return flip_top_imaging_rows || flip_bottom_reference_rows;
  }

  constexpr bool operator==(const FrameRowFlips& other) const {
    return flip_top_imaging_rows == other.flip_top_imaging_rows &&
           flip_bottom_reference_rows == other.flip_bottom_reference_rows;
  }
  constexpr bool operator!=(const FrameRowFlips& other) const {
    return !(*this == other);
  }
};

// Native FPGA tile headers identify this fixed detector layout. These are
// layout/provenance bits, not user-selectable receiver corrections.
constexpr FrameRowFlips fpga_detector_row_flips() { return {true, true}; }

// Global row permutation, not a local flip within each tile. The ZLP tile
// crossing row 32 is deliberately split between fixed references and imaging.
STEM_ROW_HD inline uint32_t orient_detector_row(uint32_t row,
                                               FrameRowFlips flips) {
  if (flips.flip_top_imaging_rows && row >= DETECTOR_REFERENCE_ROWS &&
      row < DETECTOR_HALF_ROWS) {
    return DETECTOR_REFERENCE_ROWS + DETECTOR_HALF_ROWS - 1 - row;
  }
  if (flips.flip_bottom_reference_rows &&
      row >= FRAME_HEIGHT - DETECTOR_REFERENCE_ROWS && row < FRAME_HEIGHT) {
    return 2 * FRAME_HEIGHT - DETECTOR_REFERENCE_ROWS - 1 - row;
  }
  return row;
}

}  // namespace stem

#undef STEM_ROW_HD
