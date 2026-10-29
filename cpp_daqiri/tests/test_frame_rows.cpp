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
                "Native detector placement must always orient both streams");
  static_assert(native.layout_version == 2, "Native detector layout must be version 2");
  require(stem::orient_detector_row(32, native) == 511);
  require(stem::orient_detector_row(512, native) == 1023);
  require(stem::orient_detector_row(513, native) == 1022);
  require(stem::orient_detector_row(543, native) == 992);
  require(stem::orient_detector_row(544, native) == 512);
  require(stem::orient_detector_row(1023, native) == 991);
  require(native != stem::FrameRowFlips{true, true, 1});
  require(stem::FrameRowFlips{true, false, 1} == stem::FrameRowFlips{true, false, 2});
  for (bool top : {false, true}) {
    for (bool bottom : {false, true}) {
      const stem::FrameRowFlips flags{top, bottom};
      require(flags.enabled() == (top || bottom));
      std::array<unsigned, 1024> visits{};
      for (unsigned row = 0; row < visits.size(); ++row) {
        const unsigned expected = top && row >= 32 && row <= 511 ? 543 - row
            : bottom && row >= 512 ? (row < 544 ? 1535 - row : row - 32) : row;
        const auto actual = stem::orient_detector_row(row, flags);
        require(actual == expected);
        require(actual < visits.size());
        visits[actual]++;
        const unsigned inverse = top && actual >= 32 && actual < 512 ? 543 - actual
            : bottom && actual >= 512 ? (actual >= 992 ? 1535 - actual : actual + 32) : actual;
        require(inverse == row);
        require((actual < 512) == (row < 512));
      }
      for (const auto count : visits) { require(count == 1); }
      require(stem::orient_detector_row(1024, flags) == 1024);
    }
  }
  std::cout << "PASS all row-flip combinations, boundaries, bijection and inverse\n";
}
