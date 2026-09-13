/* SPDX-License-Identifier: Apache-2.0 */
#include "stem_synthetic_source.h"

#include <stdexcept>
#include <string>

namespace stem {
namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("synthetic CUDA: ") + cudaGetErrorString(result));
  }
}

__global__ void generate_synthetic_packets(uint8_t* data, SyntheticLayout layout) {
  const uint32_t packet = blockIdx.x;
  uint8_t* p = data + static_cast<uint64_t>(packet) * layout.stride_bytes();
  if (threadIdx.x == 0) { layout.write_header(p, packet); }
  auto* payload = reinterpret_cast<uint16_t*>(p + L4_HEADER_SIZE + STEM_HEADER_SIZE);
  for (uint32_t i = threadIdx.x; i < layout.payload_bytes() / 2; i += blockDim.x) {
    payload[i] = layout.sample(layout.frame(packet), layout.tile(packet), i);
  }
}

__global__ void validate_synthetic_kernel(
    const uint16_t* input, uint64_t first_frame, uint64_t values,
    SyntheticLayout layout, unsigned long long* mismatches) {
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= values) { return; }
  const uint32_t pixel = i % (FRAME_HEIGHT * FRAME_WIDTH);
  const uint32_t row = pixel / FRAME_WIDTH;
  const uint32_t col = pixel % FRAME_WIDTH;
  uint32_t tile, sample;
  if (col < TILE_ZLP_COLUMNS) {
    tile = (row / 128) * 24 + col / 32;
    sample = (row % 128) * 32 + col % 32;
  } else {
    tile = 192 + (row / 32) * 24 + (col - TILE_ZLP_COLUMNS) / 128;
    sample = (row % 32) * 128 + (col - TILE_ZLP_COLUMNS) % 128;
  }
  if (layout.legacy_payload && sample >= FRAME_WIDTH) { sample -= FRAME_WIDTH; }
  const uint32_t frame = (first_frame + i / (FRAME_HEIGHT * FRAME_WIDTH)) % FRAMES_PER_WRAP;
  const uint16_t expected = tile < layout.source_count() * TILE_PACKETS_PER_SOURCE
      ? layout.sample(frame, tile, sample) : 0;
  if (input[i] != expected) { atomicAdd(mismatches, 1ULL); }
}
}  // namespace

SyntheticPacketPool::SyntheticPacketPool(const SyntheticLayout& layout) {
  bytes_ = static_cast<uint64_t>(layout.cycle_packets()) * layout.stride_bytes();
  check(cudaMalloc(&device_, bytes_));
  try {
    pointers_.resize(layout.cycle_packets());
    for (uint32_t i = 0; i < pointers_.size(); ++i) {
      pointers_[i] = device_ + static_cast<uint64_t>(i) * layout.stride_bytes();
    }
    generate_synthetic_packets<<<layout.cycle_packets(), 256>>>(device_, layout);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  } catch (...) {
    cudaFree(device_);
    throw;
  }
}

SyntheticPacketPool::~SyntheticPacketPool() { if (device_) { cudaFree(device_); } }

void validate_synthetic_frames(const uint16_t* frames, uint64_t first_frame,
                               uint32_t count, const SyntheticLayout& layout,
                               unsigned long long* mismatches, cudaStream_t stream) {
  const uint64_t values = static_cast<uint64_t>(count) * FRAME_HEIGHT * FRAME_WIDTH;
  validate_synthetic_kernel<<<static_cast<uint32_t>((values + 255) / 256), 256, 0, stream>>>(
      frames, first_frame, values, layout, mismatches);
  check(cudaGetLastError());
}
}  // namespace stem
