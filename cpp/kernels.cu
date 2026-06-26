/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <stdio.h>
#include "kernels.cuh"
#include "matx.h"

__global__ void populate_packets(uint8_t** gpu_bufs, uint16_t pkt_len, uint16_t offset) {
  int pkt = blockIdx.x;

  for (int samp = threadIdx.x; samp < pkt_len / 4; samp += blockDim.x) {
    auto p = reinterpret_cast<uint32_t*>(gpu_bufs[pkt] + offset + samp * sizeof(uint32_t));
    *p = (samp << 16) | (samp & 0xff);
  }
}

/**
 * @brief Populate each packet with a monotonically-increasing sequence
 *
 * @param gpu_bufs GPU packet pointer list from advanced_network "gpu_pkts"
 * @param pkt_len Length of each packet in bytes. Must be a multiple of 4
 * @param num_pkts Number of packets
 * @param offset Offset into packet to start
 * @param stream CUDA stream
 */
void populate_packets(uint8_t** gpu_bufs, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                      cudaStream_t stream) {
  populate_packets<<<num_pkts, 256, 0, stream>>>(gpu_bufs, pkt_len, offset);
}

// Must be divisible by 4 bytes in this kernel!
__global__ void copy_headers(uint8_t** gpu_bufs, void* header, uint16_t hdr_size) {
  if (gpu_bufs == nullptr) return;

  int pkt = blockIdx.x;

  for (int samp = threadIdx.x; samp < hdr_size / 4; samp += blockDim.x) {
    auto p = reinterpret_cast<uint32_t*>(gpu_bufs[pkt] + samp * sizeof(uint32_t));
    *p = *(reinterpret_cast<uint32_t*>(header) + samp);
  }
}

__global__ void print_packets(uint8_t** gpu_bufs) {
  uint8_t* p = gpu_bufs[0];
  for (int i = 0; i < 64; i++) { printf("%02X ", p[i]); }

  printf("\n");
}

void copy_headers(uint8_t** gpu_bufs, void* header, uint16_t hdr_size, uint32_t num_pkts,
                  cudaStream_t stream) {
  copy_headers<<<num_pkts, 32, 0, stream>>>(gpu_bufs, header, hdr_size);
}




// New kernel for transmitting packets
__global__ void populate_packets_from_frame_kernel(uint8_t* frame_buf, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset) {
  int pkt_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int block_dim = blockDim.x;

  if (pkt_idx >= num_pkts) return;

  uint8_t* pkt_start = frame_buf + (pkt_idx * pkt_len) + offset;

  for (int i = thread_idx; i < pkt_len; i += block_dim) {
      pkt_start[i] = (uint8_t)(pkt_idx + i);
  }
}

void populate_packets_from_frame(uint8_t* frame_buf, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                                 cudaStream_t stream) {
  populate_packets_from_frame_kernel<<<num_pkts, 256, 0, stream>>>(frame_buf, pkt_len, num_pkts, offset);
}

__device__ __forceinline__ int32_t source_id_to_global_row(uint32_t source_id, uint32_t row_offset) {
  // Layout rows from the center of the frame outward:
  // - source IDs 0..3 fill rows 511..0 in interleaved groups of 4
  //   ordered as (3, 2, 1, 0) when read from top to bottom.
  // - source IDs 4..7 fill rows 512..1023 in interleaved groups of 4
  //   ordered as (4, 5, 6, 7) when read from top to bottom.
  if (source_id < 4) { return 511 - static_cast<int32_t>(row_offset * 4 + source_id); }
  if (source_id < 8) { return 512 + static_cast<int32_t>(row_offset * 4 + (source_id - 4)); }
  return -1;
}

__global__ void gather_packets_kernel(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, uint64_t base_absolute_row) {

  int pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  uint8_t* src = src_ptrs[pkt_idx];
  
  // Extract 16-bit row number from custom header (which wraps every 16384 rows)
  uint16_t row_number = ((uint16_t)src[5] << 8) | (uint16_t)src[4];
  
  uint32_t source_id = ((uint16_t)src[7] << 8) | (uint16_t)src[6];
  
  // row_number wraps every 16384 rows, giving 128 frames (since each frame is 128 rows per source)
  uint32_t frame_idx = row_number / 128;
  uint32_t row_offset = row_number % 128;
  
  int32_t global_row = source_id_to_global_row(source_id, row_offset);
  if (global_row < 0) return; // Ignore invalid source_id
  
  // Map the modulo-128 frame counter into the current output batch window.
  // The previous "nearest signed distance" logic dropped the second half of a
  // full 128-frame tensor because frames 65..127 were wrapped to negative
  // offsets. Here we keep a forward modulo offset from the batch start and
  // discard only packets that lie outside the configured batch length.
  const uint32_t base_frame_mod = static_cast<uint32_t>((base_absolute_row / 1024) % 128);
  const uint32_t batch_frames = max_rows / 1024;
  const uint32_t relative_frame = (frame_idx + 128 - base_frame_mod) % 128;

  if (relative_frame >= batch_frames) return;

  int64_t target_row_1d = static_cast<int64_t>(relative_frame) * 1024 + global_row;

  if (target_row_1d < 0 || target_row_1d >= max_rows) return;

  uint8_t* payload_src = src + header_len;
  uint8_t* dst = dst_base + target_row_1d * payload_len;

  // Since Eth+IP+UDP header is 42 bytes, the starting addr is usually only 2-byte aligned.
  // Using uint4 (16-byte) or uint32_t (4-byte) causes a CUDA Misaligned Address exception.
  // uint16_t is safe.
  uint16_t* src16 = (uint16_t*)payload_src;
  uint16_t* dst16 = (uint16_t*)dst;
  int unroll_len = payload_len / sizeof(uint16_t);
  for (int i = threadIdx.x; i < unroll_len; i += blockDim.x) {
    dst16[i] = src16[i];
  }
}

void gather_packets(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, uint64_t base_absolute_row, cudaStream_t stream) {
  gather_packets_kernel<<<num_pkts, 256, 0, stream>>>(src_ptrs, dst_base, payload_len, header_len, num_pkts, max_rows, base_absolute_row);
}

__global__ void extract_packet_headers_kernel(uint8_t** src_ptrs,
                                              PacketHeaderInfo* headers,
                                              uint32_t num_pkts) {
  const uint32_t pkt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pkt_idx >= num_pkts) return;

  PacketHeaderInfo header{};
  header.source_id = 0xFFFF;
  header.global_row = -1;
  header.epoch_us = 0;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src != nullptr) {
    header.row_number = ((uint16_t)src[5] << 8) | (uint16_t)src[4];
    header.source_id = ((uint16_t)src[7] << 8) | (uint16_t)src[6];
    header.frame_index = header.row_number / 128;
    header.row_offset = header.row_number % 128;
    header.global_row = static_cast<int16_t>(
        source_id_to_global_row(header.source_id, header.row_offset));
    for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
      header.epoch_us |= static_cast<uint64_t>(src[16 + i]) << (i * 8);
    }
  }

  headers[pkt_idx] = header;
}

void extract_packet_headers(uint8_t** src_ptrs,
                            PacketHeaderInfo* headers,
                            uint32_t num_pkts,
                            cudaStream_t stream) {
  const uint32_t threads = 256;
  const uint32_t blocks = (num_pkts + threads - 1) / threads;
  extract_packet_headers_kernel<<<blocks, threads, 0, stream>>>(src_ptrs, headers, num_pkts);
}

__global__ void gather_packets_by_placement_kernel(uint8_t** src_ptrs,
                                                   const PacketPlacement* placements,
                                                   uint8_t* dst_base,
                                                   uint16_t payload_len,
                                                   uint16_t header_len,
                                                   uint32_t num_pkts,
                                                   uint32_t max_rows) {
  const uint32_t pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  const PacketPlacement placement = placements[pkt_idx];
  if (!placement.valid || placement.global_row < 0) return;

  const int64_t target_row_1d =
      static_cast<int64_t>(placement.relative_frame) * 1024 + placement.global_row;
  if (target_row_1d < 0 || target_row_1d >= max_rows) return;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) return;

  uint8_t* payload_src = src + header_len;
  uint8_t* dst = dst_base + target_row_1d * payload_len;

  const uint16_t* src16 = reinterpret_cast<const uint16_t*>(payload_src);
  uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
  const int unroll_len = payload_len / sizeof(uint16_t);
  for (int i = threadIdx.x; i < unroll_len; i += blockDim.x) {
    dst16[i] = src16[i];
  }
}

void gather_packets_by_placement(uint8_t** src_ptrs,
                                 const PacketPlacement* placements,
                                 uint8_t* dst_base,
                                 uint16_t payload_len,
                                 uint16_t header_len,
                                 uint32_t num_pkts,
                                 uint32_t max_rows,
                                 cudaStream_t stream) {
  gather_packets_by_placement_kernel<<<num_pkts, 256, 0, stream>>>(
      src_ptrs, placements, dst_base, payload_len, header_len, num_pkts, max_rows);
}

__device__ __forceinline__ bool tile_geometry(uint32_t tile_index,
                                              uint32_t& row_start,
                                              uint32_t& col_start,
                                              uint32_t& tile_height,
                                              uint32_t& tile_width) {
  constexpr uint32_t kFrameHeight = 1024;
  constexpr uint32_t kFrameWidth = 3840;
  constexpr uint32_t kZlpColumns = 192 * 4;
  constexpr uint32_t kZlpTileWidth = 32;
  constexpr uint32_t kZlpTileHeight = 128;
  constexpr uint32_t kCoreTileWidth = 128;
  constexpr uint32_t kCoreTileHeight = 32;
  constexpr uint32_t kZlpTileCols = kZlpColumns / kZlpTileWidth;
  constexpr uint32_t kZlpTileRows = kFrameHeight / kZlpTileHeight;
  constexpr uint32_t kZlpTiles = kZlpTileCols * kZlpTileRows;
  constexpr uint32_t kCoreColumns = kFrameWidth - kZlpColumns;
  constexpr uint32_t kCoreTileCols = kCoreColumns / kCoreTileWidth;
  constexpr uint32_t kCoreTileRows = kFrameHeight / kCoreTileHeight;
  constexpr uint32_t kCoreTiles = kCoreTileCols * kCoreTileRows;

  if (tile_index < kZlpTiles) {
    const uint32_t tile_row = tile_index / kZlpTileCols;
    const uint32_t tile_col = tile_index % kZlpTileCols;
    row_start = tile_row * kZlpTileHeight;
    col_start = tile_col * kZlpTileWidth;
    tile_height = kZlpTileHeight;
    tile_width = kZlpTileWidth;
    return true;
  }

  const uint32_t core_index = tile_index - kZlpTiles;
  if (core_index >= kCoreTiles) return false;

  const uint32_t tile_row = core_index / kCoreTileCols;
  const uint32_t tile_col = core_index % kCoreTileCols;
  row_start = tile_row * kCoreTileHeight;
  col_start = kZlpColumns + tile_col * kCoreTileWidth;
  tile_height = kCoreTileHeight;
  tile_width = kCoreTileWidth;
  return true;
}

__global__ void gather_tile_packets_by_placement_kernel(
    uint8_t** src_ptrs,
    const PacketPlacement* placements,
    uint8_t* dst_base,
    uint16_t available_payload_len,
    uint16_t header_len,
    uint32_t num_pkts,
    uint32_t frames,
    uint32_t frame_height,
    uint32_t frame_width,
    bool duplicate_prefix_to_simulate_tile_payload) {
  constexpr uint32_t kTileSamples = 4096;

  const uint32_t pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  const PacketPlacement placement = placements[pkt_idx];
  if (!placement.valid || placement.relative_frame >= frames) return;

  uint32_t row_start = 0;
  uint32_t col_start = 0;
  uint32_t tile_height = 0;
  uint32_t tile_width = 0;
  if (!tile_geometry(placement.tile_index, row_start, col_start, tile_height, tile_width)) return;
  if (row_start + tile_height > frame_height || col_start + tile_width > frame_width) return;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) return;

  const uint16_t* payload = reinterpret_cast<const uint16_t*>(src + header_len);
  uint16_t* output = reinterpret_cast<uint16_t*>(dst_base);
  const uint32_t available_samples = available_payload_len / sizeof(uint16_t);
  const uint64_t frame_offset =
      static_cast<uint64_t>(placement.relative_frame) * frame_height * frame_width;

  for (uint32_t sample_idx = threadIdx.x; sample_idx < kTileSamples; sample_idx += blockDim.x) {
    uint32_t src_sample_idx = sample_idx;
    if (duplicate_prefix_to_simulate_tile_payload && src_sample_idx >= available_samples) {
      src_sample_idx -= available_samples;
    }
    if (src_sample_idx >= available_samples) continue;

    const uint32_t local_row = sample_idx / tile_width;
    const uint32_t local_col = sample_idx - local_row * tile_width;
    if (local_row >= tile_height) continue;

    const uint64_t dst_idx =
        frame_offset +
        static_cast<uint64_t>(row_start + local_row) * frame_width +
        (col_start + local_col);
    output[dst_idx] = payload[src_sample_idx];
  }
}

void gather_tile_packets_by_placement(uint8_t** src_ptrs,
                                      const PacketPlacement* placements,
                                      uint8_t* dst_base,
                                      uint16_t available_payload_len,
                                      uint16_t header_len,
                                      uint32_t num_pkts,
                                      uint32_t frames,
                                      uint32_t frame_height,
                                      uint32_t frame_width,
                                      bool duplicate_prefix_to_simulate_tile_payload,
                                      cudaStream_t stream) {
  gather_tile_packets_by_placement_kernel<<<num_pkts, 256, 0, stream>>>(
      src_ptrs,
      placements,
      dst_base,
      available_payload_len,
      header_len,
      num_pkts,
      frames,
      frame_height,
      frame_width,
      duplicate_prefix_to_simulate_tile_payload);
}

__global__ void summarize_packets_kernel(uint8_t** src_ptrs,
                                         PacketDebugSummary* summaries,
                                         uint16_t payload_len,
                                         uint16_t header_len,
                                         uint32_t num_pkts) {
  const uint32_t pkt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pkt_idx >= num_pkts) return;

  PacketDebugSummary summary{};
  summary.source_id = 0xFFFF;
  summary.global_row = -1;
  summary.first_nonzero_col = 0xFFFF;
  summary.second_nonzero_col = 0xFFFF;
  summary.max_value_col = 0xFFFF;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) {
    summaries[pkt_idx] = summary;
    return;
  }

  summary.row_number = ((uint16_t)src[5] << 8) | (uint16_t)src[4];
  summary.source_id = ((uint16_t)src[7] << 8) | (uint16_t)src[6];
  summary.frame_index = summary.row_number / 128;
  const uint32_t row_offset = summary.row_number % 128;
  summary.global_row = static_cast<int16_t>(source_id_to_global_row(summary.source_id, row_offset));

  const uint16_t* payload = reinterpret_cast<const uint16_t*>(src + header_len);
  const uint32_t sample_count = payload_len / sizeof(uint16_t);

  for (uint32_t col = 0; col < sample_count; ++col) {
    const uint16_t value = payload[col];
    if (value == 0) { continue; }

    if (summary.nonzero_count == 0) {
      summary.first_nonzero_col = static_cast<uint16_t>(col);
      summary.first_nonzero_value = value;
    } else if (summary.nonzero_count == 1) {
      summary.second_nonzero_col = static_cast<uint16_t>(col);
      summary.second_nonzero_value = value;
    }

    if (summary.max_value_col == 0xFFFF || value > summary.max_value) {
      summary.max_value_col = static_cast<uint16_t>(col);
      summary.max_value = value;
    }

    if (summary.nonzero_count < 0xFFFF) { summary.nonzero_count++; }
  }

  summaries[pkt_idx] = summary;
}

void summarize_packets(uint8_t** src_ptrs,
                       PacketDebugSummary* summaries,
                       uint16_t payload_len,
                       uint16_t header_len,
                       uint32_t num_pkts,
                       cudaStream_t stream) {
  const uint32_t threads = 256;
  const uint32_t blocks = (num_pkts + threads - 1) / threads;
  summarize_packets_kernel<<<blocks, threads, 0, stream>>>(
      src_ptrs, summaries, payload_len, header_len, num_pkts);
}

template <typename InputT>
__global__ void compute_blr_baseline_kernel(const InputT* input,
                                            const float* dark_frame,
                                            float* baseline,
                                            uint32_t frames,
                                            uint32_t height,
                                            uint32_t width,
                                            uint32_t blr_rows,
                                            uint32_t zlp_width,
                                            uint32_t zlp_group_columns,
                                            uint32_t core_group_columns,
                                            uint32_t zlp_bins,
                                            uint32_t bins_per_half,
                                            bool subtract_dark) {
  const uint64_t baseline_values = static_cast<uint64_t>(frames) * 2 * bins_per_half;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;

  for (uint64_t baseline_idx =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       baseline_idx < baseline_values;
       baseline_idx += stride) {
    const uint32_t bin = baseline_idx % bins_per_half;
    const uint64_t frame_half = baseline_idx / bins_per_half;
    const uint32_t half = frame_half % 2;
    const uint32_t frame = frame_half / 2;

    const bool is_zlp = bin < zlp_bins;
    const uint32_t group_columns =
        is_zlp ? zlp_group_columns : core_group_columns;
    const uint32_t group_start =
        is_zlp
            ? bin * zlp_group_columns
            : zlp_width + (bin - zlp_bins) * core_group_columns;
    const uint32_t row_start = half == 0 ? 0 : height - blr_rows;

    float sum = 0.0f;
    for (uint32_t row_offset = 0; row_offset < blr_rows; ++row_offset) {
      const uint64_t pixel_base =
          static_cast<uint64_t>(row_start + row_offset) * width + group_start;
      const uint64_t input_base =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_base;
      for (uint32_t col_offset = 0; col_offset < group_columns; ++col_offset) {
        const uint64_t pixel_idx = pixel_base + col_offset;
        const float dark_value = subtract_dark ? dark_frame[pixel_idx] : 0.0f;
        sum += static_cast<float>(input[input_base + col_offset]) - dark_value;
      }
    }

    baseline[baseline_idx] =
        sum / static_cast<float>(blr_rows * group_columns);
  }
}

template <typename InputT>
void launch_compute_blr_baseline(const InputT* input,
                                 const float* dark_frame,
                                 float* baseline,
                                 uint32_t frames,
                                 uint32_t height,
                                 uint32_t width,
                                 uint32_t blr_rows,
                                 uint32_t zlp_width,
                                 uint32_t zlp_group_columns,
                                 uint32_t core_group_columns,
                                 bool subtract_dark,
                                 cudaStream_t stream) {
  if (frames == 0 || height == 0 || width == 0 || blr_rows == 0) { return; }

  const uint32_t zlp_bins = zlp_width / zlp_group_columns;
  const uint32_t core_bins = (width - zlp_width) / core_group_columns;
  const uint32_t bins_per_half = zlp_bins + core_bins;
  const uint64_t baseline_values = static_cast<uint64_t>(frames) * 2 * bins_per_half;

  const uint32_t threads = 256;
  const uint64_t required_blocks = (baseline_values + threads - 1) / threads;
  const uint32_t blocks =
      static_cast<uint32_t>(required_blocks > 65535 ? 65535 : required_blocks);

  compute_blr_baseline_kernel<InputT><<<blocks, threads, 0, stream>>>(
      input,
      dark_frame,
      baseline,
      frames,
      height,
      width,
      blr_rows,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      zlp_bins,
      bins_per_half,
      subtract_dark);
}

void compute_blr_baseline(const uint16_t* input,
                          const float* dark_frame,
                          float* baseline,
                          uint32_t frames,
                          uint32_t height,
                          uint32_t width,
                          uint32_t blr_rows,
                          uint32_t zlp_width,
                          uint32_t zlp_group_columns,
                          uint32_t core_group_columns,
                          bool subtract_dark,
                          cudaStream_t stream) {
  launch_compute_blr_baseline(
      input,
      dark_frame,
      baseline,
      frames,
      height,
      width,
      blr_rows,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      subtract_dark,
      stream);
}

void compute_blr_baseline(const float* input,
                          const float* dark_frame,
                          float* baseline,
                          uint32_t frames,
                          uint32_t height,
                          uint32_t width,
                          uint32_t blr_rows,
                          uint32_t zlp_width,
                          uint32_t zlp_group_columns,
                          uint32_t core_group_columns,
                          bool subtract_dark,
                          cudaStream_t stream) {
  launch_compute_blr_baseline(
      input,
      dark_frame,
      baseline,
      frames,
      height,
      width,
      blr_rows,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      subtract_dark,
      stream);
}

template <typename InputT>
__global__ void correct_with_blr_and_mean_kernel(
    const InputT* input,
    const float* dark_frame,
    const float* blr_baseline,
    float* output,
    float* batch_mean,
    uint32_t frames,
    uint32_t height,
    uint32_t width,
    uint32_t zlp_width,
    uint32_t zlp_group_columns,
    uint32_t core_group_columns,
    uint32_t zlp_bins,
    uint32_t bins_per_half,
    bool subtract_dark,
    bool apply_blr,
    bool compute_batch_mean) {
  const uint32_t frame_pixels = height * width;
  const uint32_t pixel_stride = blockDim.x * gridDim.x;
  const uint32_t half_height = height / 2;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels;
       pixel_idx += pixel_stride) {
    const uint32_t row = pixel_idx / width;
    const uint32_t col = pixel_idx - row * width;
    const uint32_t half = row < half_height ? 0 : 1;
    uint32_t bin = 0;
    if (apply_blr) {
      bin =
          col < zlp_width
              ? col / zlp_group_columns
              : zlp_bins + (col - zlp_width) / core_group_columns;
    }
    const float dark_value = subtract_dark ? dark_frame[pixel_idx] : 0.0f;

    float sum = 0.0f;
    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx = static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      float value = static_cast<float>(input[idx]) - dark_value;
      if (apply_blr) {
        const uint64_t baseline_idx =
            (static_cast<uint64_t>(frame) * 2 + half) * bins_per_half + bin;
        value -= blr_baseline[baseline_idx];
      }
      output[idx] = value;
      if (compute_batch_mean) { sum += value; }
    }

    if (compute_batch_mean) {
      batch_mean[pixel_idx] = sum / static_cast<float>(frames);
    }
  }
}

template <typename InputT>
void launch_correct_with_blr_and_mean(const InputT* input,
                                      const float* dark_frame,
                                      const float* blr_baseline,
                                      float* output,
                                      float* batch_mean,
                                      uint32_t frames,
                                      uint32_t height,
                                      uint32_t width,
                                      uint32_t zlp_width,
                                      uint32_t zlp_group_columns,
                                      uint32_t core_group_columns,
                                      bool subtract_dark,
                                      bool apply_blr,
                                      bool compute_batch_mean,
                                      cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t zlp_bins = apply_blr ? zlp_width / zlp_group_columns : 0;
  const uint32_t core_bins =
      apply_blr ? (width - zlp_width) / core_group_columns : 0;
  const uint32_t bins_per_half = zlp_bins + core_bins;

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks =
      static_cast<uint32_t>(required_blocks > 65535 ? 65535 : required_blocks);

  correct_with_blr_and_mean_kernel<InputT><<<blocks, threads, 0, stream>>>(
      input,
      dark_frame,
      blr_baseline,
      output,
      batch_mean,
      frames,
      height,
      width,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      zlp_bins,
      bins_per_half,
      subtract_dark,
      apply_blr,
      compute_batch_mean);
}

void correct_with_blr_and_mean(const uint16_t* input,
                               const float* dark_frame,
                               const float* blr_baseline,
                               float* output,
                               float* batch_mean,
                               uint32_t frames,
                               uint32_t height,
                               uint32_t width,
                               uint32_t zlp_width,
                               uint32_t zlp_group_columns,
                               uint32_t core_group_columns,
                               bool subtract_dark,
                               bool apply_blr,
                               bool compute_batch_mean,
                               cudaStream_t stream) {
  launch_correct_with_blr_and_mean(
      input,
      dark_frame,
      blr_baseline,
      output,
      batch_mean,
      frames,
      height,
      width,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      subtract_dark,
      apply_blr,
      compute_batch_mean,
      stream);
}

void correct_with_blr_and_mean(const float* input,
                               const float* dark_frame,
                               const float* blr_baseline,
                               float* output,
                               float* batch_mean,
                               uint32_t frames,
                               uint32_t height,
                               uint32_t width,
                               uint32_t zlp_width,
                               uint32_t zlp_group_columns,
                               uint32_t core_group_columns,
                               bool subtract_dark,
                               bool apply_blr,
                               bool compute_batch_mean,
                               cudaStream_t stream) {
  launch_correct_with_blr_and_mean(
      input,
      dark_frame,
      blr_baseline,
      output,
      batch_mean,
      frames,
      height,
      width,
      zlp_width,
      zlp_group_columns,
      core_group_columns,
      subtract_dark,
      apply_blr,
      compute_batch_mean,
      stream);
}

__device__ __forceinline__ float median_from_small_window(float* values, uint32_t count) {
  const uint32_t median_idx = count / 2;
  for (uint32_t i = 0; i <= median_idx; ++i) {
    uint32_t min_idx = i;
    float min_value = values[i];
    for (uint32_t j = i + 1; j < count; ++j) {
      if (values[j] < min_value) {
        min_value = values[j];
        min_idx = j;
      }
    }
    values[min_idx] = values[i];
    values[i] = min_value;
  }
  return values[median_idx];
}

__global__ void apply_dynamic_and_valid_pixel_mask_float_kernel(
    float* input,
    const float* batch_mean,
    const float* valid_pixel_mask,
    uint32_t frames,
    uint32_t height,
    uint32_t width,
    uint32_t median_window_pixels,
    float threshold_ratio,
    float threshold_offset,
    uint32_t excluded_edge_rows,
    bool apply_dynamic_mask,
    bool two_sided,
    bool apply_valid_pixel_mask) {
  constexpr uint32_t kMaxMedianWindowPixels = 129;
  float window_values[kMaxMedianWindowPixels];

  const uint32_t frame_pixels = height * width;
  const uint32_t pixel_stride = blockDim.x * gridDim.x;
  const uint32_t half_height = height / 2;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels;
       pixel_idx += pixel_stride) {
    const uint32_t row = pixel_idx / width;
    const uint32_t col = pixel_idx - row * width;
    bool should_zero =
        apply_valid_pixel_mask && valid_pixel_mask[pixel_idx] == 0.0f;

    if (apply_dynamic_mask && !should_zero) {
      const bool top_half = row < half_height;
      const uint32_t half_start = top_half ? excluded_edge_rows : half_height;
      const uint32_t half_end = top_half ? half_height : height - excluded_edge_rows;

      if (row >= half_start && row < half_end) {
        const uint32_t radius = median_window_pixels / 2;
        uint32_t row_start = row > radius ? row - radius : half_start;
        row_start = row_start < half_start ? half_start : row_start;
        uint32_t row_end = row + radius + 1;
        row_end = row_end > half_end ? half_end : row_end;

        uint32_t count = 0;
        for (uint32_t sample_row = row_start;
             sample_row < row_end && count < kMaxMedianWindowPixels;
             ++sample_row) {
          window_values[count++] =
              batch_mean[static_cast<uint64_t>(sample_row) * width + col];
        }

        if (count > 0) {
          const float local_median = median_from_small_window(window_values, count);
          const float current_value = batch_mean[pixel_idx];
          const float reference = local_median * threshold_ratio;
          const float deviation = current_value - reference;
          should_zero =
              two_sided ? fabsf(deviation) > threshold_offset : deviation > threshold_offset;
        }
      }
    }
    if (!should_zero) { continue; }

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx = static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      input[idx] = 0.0f;
    }
  }
}

void apply_dynamic_and_valid_pixel_mask_float(float* input,
                                              const float* batch_mean,
                                              const float* valid_pixel_mask,
                                              uint32_t frames,
                                              uint32_t height,
                                              uint32_t width,
                                              uint32_t median_window_pixels,
                                              float threshold_ratio,
                                              float threshold_offset,
                                              uint32_t excluded_edge_rows,
                                              bool apply_dynamic_mask,
                                              bool two_sided,
                                              bool apply_valid_pixel_mask,
                                              cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0 || (!apply_dynamic_mask && !apply_valid_pixel_mask)) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks =
      static_cast<uint32_t>(required_blocks > 65535 ? 65535 : required_blocks);

  apply_dynamic_and_valid_pixel_mask_float_kernel<<<blocks, threads, 0, stream>>>(
      input,
      batch_mean,
      valid_pixel_mask,
      frames,
      height,
      width,
      median_window_pixels,
      threshold_ratio,
      threshold_offset,
      excluded_edge_rows,
      apply_dynamic_mask,
      two_sided,
      apply_valid_pixel_mask);
}

__global__ void apply_valid_pixel_mask_float_kernel(float* input,
                                                    const float* valid_pixel_mask,
                                                    uint32_t frames,
                                                    uint32_t frame_pixels) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels;
       pixel_idx += pixel_stride) {
    const float mask_value = valid_pixel_mask[pixel_idx];
    if (mask_value != 0.0f) { continue; }

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx = static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      input[idx] = 0.0f;
    }
  }
}

void apply_valid_pixel_mask_float(float* input,
                                  const float* valid_pixel_mask,
                                  uint32_t frames,
                                  uint32_t height,
                                  uint32_t width,
                                  cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks =
      static_cast<uint32_t>(required_blocks > 65535 ? 65535 : required_blocks);

  apply_valid_pixel_mask_float_kernel<<<blocks, threads, 0, stream>>>(
      input,
      valid_pixel_mask,
      frames,
      static_cast<uint32_t>(frame_pixels));
}
