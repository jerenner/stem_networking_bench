/* SPDX-License-Identifier: Apache-2.0 */
#include "stem_frame_rows.h"

#include <array>
#include <iostream>
#include <stdexcept>

void require(bool condition) {
  if (!condition) { throw std::runtime_error("detector row permutation mismatch"); }
}

int main() {
  constexpr auto native = stem::fpga_detector_row_flips();
  static_assert(native.flip_top_imaging_rows && native.flip_bottom_reference_rows,
                "Native detector placement must always include both reversals");
  require(stem::orient_detector_row(32, native) == 511);
  require(stem::orient_detector_row(992, native) == 1023);
  for (bool top : {false, true}) {
    for (bool bottom : {false, true}) {
      const stem::FrameRowFlips flags{top, bottom};
      require(flags.enabled() == (top || bottom));
      std::array<unsigned, 1024> visits{};
      for (unsigned row = 0; row < visits.size(); ++row) {
        const unsigned expected = top && row >= 32 && row <= 511 ? 543 - row
            : bottom && row >= 992 ? 2015 - row : row;
        const auto actual = stem::orient_detector_row(row, flags);
        require(actual == expected);
        require(actual < visits.size());
        visits[actual]++;
        require(stem::orient_detector_row(actual, flags) == row);
        require((actual < 512) == (row < 512));
      }
      for (const auto count : visits) { require(count == 1); }
      require(stem::orient_detector_row(1024, flags) == 1024);
    }
  }
  std::cout << "PASS all row-flip combinations, boundaries, bijection and inverse\n";
}
