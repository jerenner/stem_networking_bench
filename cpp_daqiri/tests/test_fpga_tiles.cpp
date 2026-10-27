/* SPDX-License-Identifier: Apache-2.0 */
#include "stem_fpga_tiles.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}

void check_header() {
  std::array<uint8_t, 64> bytes{};
  bytes.fill(0x37);
  bytes[0] = bytes[1] = 0x5a;
  bytes[2] = 157;
  bytes[3] = 15;
  bytes[4] = 0xe1;
  bytes[5] = 0x56;
  bytes[8] = bytes[9] = 0xa5;
  const auto good = stem::decode_fpga_tile_header(bytes.data());
  require(good.valid && good.tile_id == 157 && good.tile_slot == 15 &&
              good.frame_number == 0x56e1,
          "FPGA fields, half bit and little-endian counter");
  require(!stem::decode_fpga_tile_header(nullptr).valid, "null header accepted");
  for (unsigned offset : {0u, 1u, 8u, 9u}) {
    auto bad = bytes;
    bad[offset] ^= 1;
    require(!stem::decode_fpga_tile_header(bad.data()).valid, "invalid marker accepted");
  }
  for (uint16_t id = 0; id <= 255; ++id) {
    auto test = bytes;
    test[2] = id;
    require(stem::decode_fpga_tile_header(test.data()).valid ==
                (id < 30 || (id >= 128 && id < 158)),
            "FPGA ID ranges");
  }
  for (unsigned slot = 0; slot < 256; ++slot) {
    auto test = bytes;
    test[3] = slot;
    require(stem::decode_fpga_tile_header(test.data()).valid == (slot < 16),
            "slot upper bits must not be silently masked");
  }
}

void check_map() {
  for (bool row_first : {false, true}) {
    std::array<bool, 960> covered{};
    unsigned zlp = 0, core = 0;
    for (unsigned half = 0; half < 2; ++half) {
      for (unsigned link = 0; link < 30; ++link) {
        for (unsigned slot = 0; slot < 16; ++slot) {
          const auto tile = stem::fpga_tile_index(half * 128 + link, slot, row_first);
          require(tile < covered.size() && !covered[tile], "mapping overlap or bounds");
          covered[tile] = true;
          unsigned row, col;
          if (link < 6) {
            ++zlp;
            row = (tile / 24) * 128;
            col = (tile % 24) * 32;
            const unsigned read = row_first ? slot % 4 : slot / 4;
            const unsigned vertical = row_first ? slot / 4 : slot % 4;
            require(row == half * 512 + vertical * 128 &&
                        col == read * 192 + link * 32,
                    "ZLP placement/readout/half");
          } else {
            ++core;
            row = ((tile - 192) / 24) * 32;
            col = 768 + ((tile - 192) % 24) * 128;
            require(row == half * 512 + slot * 32 &&
                        col == 768 + (link - 6) * 128,
                    "CoreLoss placement/half");
          }
          require((row < 512) == (half == 0), "packet crossed sensor halves");
        }
      }
    }
    for (bool present : covered) { require(present, "map left holes"); }
    require(zlp == 192 && core == 768, "full image tile counts");
  }
  for (unsigned bad : {30u, 127u, 158u, 255u, 256u, 65535u}) {
    require(stem::fpga_tile_index(bad, 0) == 960, "invalid ID mapped");
  }
  require(stem::fpga_tile_index(0, 16) == 960, "invalid slot mapped");
  require(stem::fpga_tile_index(0, 0) == 0 &&
              stem::fpga_tile_index(5, 15) == 95 &&
              stem::fpga_tile_index(128, 0) == 96 &&
              stem::fpga_tile_index(133, 15) == 191 &&
              stem::fpga_tile_index(6, 0) == 192 &&
              stem::fpga_tile_index(29, 15) == 575 &&
              stem::fpga_tile_index(134, 0) == 576 &&
              stem::fpga_tile_index(157, 15) == 959,
          "fixed spatial anchors; sparse selection must not compact them");
}

void check_counter() {
  stem::FpgaFrameCounter counter;
  require(counter.unwrap(65534) == 65534, "start at arbitrary firmware counter");
  require(counter.unwrap(65535) == 65535, "advance before wrap");
  require(counter.unwrap(0) == 65536, "16-bit wrap");
  require(counter.unwrap(65534) == 65534, "reordering across wrap");
  require(counter.unwrap(1) == 65537, "late packet must not reset reference");
  for (int64_t frame = 65538; frame < 140000; ++frame) {
    require(counter.unwrap(static_cast<uint16_t>(frame)) == frame,
            "counter progression through repeated wraps");
  }
  stem::FpgaFrameCounter zero;
  require(zero.unwrap(0) == 0 && zero.unwrap(65535) == -1 && zero.unwrap(1) == 1,
          "late packet before initial zero is not a future frame");
}
}  // namespace

int main() {
  try {
    check_header();
    check_map();
    check_counter();
    std::cout << "PASS: FPGA header, both halves, 960-tile maps, ZLP slot orders, "
                 "invalid fields and reordered 16-bit wraps\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
