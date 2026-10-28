#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-emulate actual CUDA kernel bodies; not a CUDA build or timing test.

Only thread/block indexing is emulated. Scalar algorithms are extracted from
stem_kernels.cu so these checks cannot silently test a second implementation.
Requires a C++17 host compiler, no GPU or Python packages.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


COMMON = Path(__file__).resolve().parents[1] / "common"


def kernel_body(source, name):
    match = re.search(
        rf"(?:template\s*<[^>]+>\s*)?__(?:global|device)__[^{{;]*\b{name}\s*\(",
        source,
    )
    if match is None:
        raise AssertionError(f"kernel definition not found: {name}")
    start = source.index("{", match.start())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return (source[match.start():end].replace("__global__", "")
            .replace("__device__", "").replace("__forceinline__", "inline"))


HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "stem_kernels.h"

struct Index { uint32_t x = 0; };
Index blockIdx, threadIdx, blockDim{256}, gridDim{1};
template <typename F> void launch(uint32_t blocks, F function) {
  gridDim.x = blocks;
  for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x) {
    for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) { function(); }
  }
}
void require(bool condition) {
  if (!condition) { throw std::runtime_error("emulated sample kernel mismatch"); }
}
namespace stem {
// KERNELS
}

int main() {
  using namespace stem;
  constexpr uint32_t pixels = FRAME_HEIGHT * FRAME_WIDTH;
  constexpr uint32_t headers = 106;
  // Exercise all spatial tile shapes, both halves, offset headers, and the
  // legacy duplicate-prefix option without touching header bytes.
  for (uint32_t tile : {0u, 168u, 192u, 959u}) {
    for (bool duplicate : {false, true}) {
      std::vector<uint16_t> packet(headers / 2 + TILE_SAMPLES, 0x5a5a);
      for (uint32_t i = 0; i < TILE_SAMPLES; ++i) {
        packet[headers / 2 + i] = static_cast<uint16_t>(i | 0xa000);
      }
      uint8_t* ptr = reinterpret_cast<uint8_t*>(packet.data());
      PacketPlacement placement{0, static_cast<uint16_t>(tile), 1, 1};
      std::vector<int16_t> signed_output(2 * pixels, 0);
      std::vector<uint16_t> unsigned_output(2 * pixels, 0);
      uint32_t row, col, h, w;
      require(tile_geometry(tile, row, col, h, w));
      const uint16_t payload_bytes = duplicate ? 7680 : 8192;
      launch(1, [&] {
        stem_gather_tile_packets_by_placement_kernel<true>(
            &ptr, &placement, reinterpret_cast<uint8_t*>(signed_output.data()),
            payload_bytes, headers, 1, 2, FRAME_HEIGHT, FRAME_WIDTH, duplicate);
        stem_gather_tile_packets_by_placement_kernel<false>(
            &ptr, &placement, reinterpret_cast<uint8_t*>(unsigned_output.data()),
            payload_bytes, headers, 1, 2, FRAME_HEIGHT, FRAME_WIDTH, duplicate);
      });
      for (uint32_t i = 0; i < TILE_SAMPLES; ++i) {
        const auto src = i >= payload_bytes / 2 ? i - payload_bytes / 2 : i;
        const auto index = pixels + (row + i / w) * FRAME_WIDTH + col + i % w;
        require(signed_output[index] == (src < 2048 ? int(src) : int(src) - 4096));
        require(unsigned_output[index] == static_cast<uint16_t>(src | 0xa000));
      }
      require(std::count_if(signed_output.begin(), signed_output.end(),
                           [](int16_t x) { return x != 0; }) ==
              (duplicate ? TILE_SAMPLES - 2 : TILE_SAMPLES - 1));
      require(packet.front() == 0x5a5a);
    }
  }

  constexpr uint32_t frames = 3, height = 8, width = 4, size = height * width;
  std::vector<int16_t> input(frames * size);
  std::vector<float> float_input(input.size()), dark(size), valid(size, 1);
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<int16_t>(int(i % 17) - 8);
    float_input[i] = input[i];
  }
  for (uint32_t i = 0; i < size; ++i) { dark[i] = float(i % 5) / 2; }
  valid[7] = 0;
  // Every toggle combination, including negative BLR edge references.
  for (bool subtract_dark : {false, true}) {
    for (bool apply_blr : {false, true}) {
      std::vector<float> baseline(frames * 2 * 2), baseline_float(baseline.size());
      launch(1, [&] {
        stem_compute_blr_baseline_kernel(input.data(), dark.data(), baseline.data(),
            frames, height, width, 1, 2, 2, 2, 1, 2, subtract_dark);
        stem_compute_blr_baseline_kernel(float_input.data(), dark.data(), baseline_float.data(),
            frames, height, width, 1, 2, 2, 2, 1, 2, subtract_dark);
      });
      require(baseline == baseline_float);
      std::vector<float> corrected(input.size()), mean(size);
      launch(1, [&] {
        stem_correct_with_blr_and_mean_kernel(input.data(), dark.data(), baseline.data(),
            corrected.data(), mean.data(), frames, height, width, 2, 2, 2, 1, 2,
            subtract_dark, apply_blr, true);
      });
      for (uint32_t pixel = 0; pixel < size; ++pixel) {
        const auto row = pixel / width, col = pixel % width;
        float sum = 0;
        for (uint32_t f = 0; f < frames; ++f) {
          const float expected = input[f * size + pixel] -
              (subtract_dark ? dark[pixel] : 0) -
              (apply_blr ? baseline[f * 4 + (row / 4) * 2 + col / 2] : 0);
          require(corrected[f * size + pixel] == expected);
          sum += expected;
        }
        require(mean[pixel] == sum / frames);
      }
      const auto before_mask = corrected;
      launch(1, [&] {
        stem_apply_dynamic_and_valid_pixel_mask_float_kernel(corrected.data(),
            mean.data(), valid.data(), frames, height, width, 3, 1, 5, 1, true, true, true);
      });
      for (uint32_t pixel = 0; pixel < size; ++pixel) {
        const auto row = pixel / width, col = pixel % width;
        bool zero = valid[pixel] == 0;
        const uint32_t lo = row < 4 ? 1 : 4, hi = row < 4 ? 4 : 7;
        if (!zero && row >= lo && row < hi) {
          std::vector<float> window;
          for (uint32_t r = std::max(row > 0 ? row - 1 : 0, lo);
               r < std::min(row + 2, hi); ++r) { window.push_back(mean[r * width + col]); }
          std::sort(window.begin(), window.end());
          zero = std::fabs(mean[pixel] - window[window.size() / 2]) > 5;
        }
        for (uint32_t f = 0; f < frames; ++f) {
          require(corrected[f * size + pixel] ==
                  (zero ? 0 : before_mask[f * size + pixel]));
        }
      }
    }
  }
  // Raw and dark-subtracted live views must not reinterpret negative samples.
  for (bool subtract_dark : {false, true}) {
    for (bool threshold : {false, true}) {
      std::vector<float> representative(size), sums(size);
      launch(1, [&] {
        stem_extract_representative_and_sum_kernel(input.data(), dark.data(),
            representative.data(), sums.data(), frames, size, width, 1,
            subtract_dark, threshold, 2, -2, 1);
      });
      for (uint32_t pixel = 0; pixel < size; ++pixel) {
        float sum = 0;
        for (uint32_t f = 0; f < frames; ++f) {
          float value = input[f * size + pixel] - (subtract_dark ? dark[pixel] : 0);
          if (threshold && value <= (pixel % width < 2 ? -2 : 1)) { value = 0; }
          if (f == 1) { require(representative[pixel] == value); }
          sum += value;
        }
        require(sums[pixel] == sum);
      }
    }
  }
  std::cout << "PASS actual gather/BLR/correction/mean/mask/preview CUDA bodies (CPU emulation)\n";
}
'''


def main():
    source = (COMMON / "stem_kernels.cu").read_text()
    names = ["tile_geometry", "stem_gather_tile_packets_by_placement_kernel",
             "stem_compute_blr_baseline_kernel", "stem_correct_with_blr_and_mean_kernel",
             "stem_median_from_small_window", "stem_apply_dynamic_and_valid_pixel_mask_float_kernel",
             "stem_extract_representative_and_sum_kernel"]
    bodies = "\n".join(kernel_body(source, name) for name in names)
    with tempfile.TemporaryDirectory(prefix="stem_sample_kernels_") as tmp:
        path = Path(tmp)
        (path / "cuda_runtime.h").write_text("#pragma once\nusing cudaStream_t = void*;\n")
        (path / "test.cpp").write_text(HARNESS.replace("// KERNELS", bodies))
        command = shlex.split(os.environ.get("CXX", "c++"))
        command += ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-O1",
                    "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                    "-I", tmp, "-I", str(COMMON), str(path / "test.cpp"),
                    "-o", str(path / "test")]
        subprocess.run(command, check=True)
        subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    main()
