/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CUDA kernels for the daqiri-based STEM pipeline. See stem_kernels.h for
 * the public surface and which phase each kernel is used by.
 */

#include <cstring>
#include <stdexcept>
#include <string>

#include "stem_kernels.h"
#include "stem_packet.h"

namespace stem {

namespace {

void check_cuda_launch(const char* kernel_name) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) { return; }
  throw std::runtime_error(std::string(kernel_name) +
                           " launch failed: " + cudaGetErrorString(err));
}

}  // namespace

// ===========================================================================
// Phase 1 TX -- host-side STEM header stamp
// ===========================================================================
void stem_tx_stamp_packet(uint8_t* stem_header_dst, uint16_t row_number,
                          uint16_t source_id, uint64_t epoch_us) {
  std::memset(stem_header_dst, 0, STEM_HEADER_SIZE);
  // Bytes 4-5: row_number (little-endian on aarch64/x86; we write byte-wise
  // so the wire layout is portable).
  stem_header_dst[STEM_HDR_OFF_ROW_NUMBER_LO] =
      static_cast<uint8_t>(row_number & 0xff);
  stem_header_dst[STEM_HDR_OFF_ROW_NUMBER_HI] =
      static_cast<uint8_t>((row_number >> 8) & 0xff);
  stem_header_dst[STEM_HDR_OFF_SOURCE_ID_LO] =
      static_cast<uint8_t>(source_id & 0xff);
  stem_header_dst[STEM_HDR_OFF_SOURCE_ID_HI] =
      static_cast<uint8_t>((source_id >> 8) & 0xff);
  for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
    stem_header_dst[STEM_HDR_OFF_EPOCH_US + i] =
        static_cast<uint8_t>((epoch_us >> (i * 8)) & 0xff);
  }
}

// ===========================================================================
// Phase 3 TX -- in-place per-burst GPU header update.
//
// Each thread block handles one packet i in the burst. Writes the row_number
// and source_id bytes (and optionally epoch_us for packet 0). For
// pkt_idx > 0 the kernel ALSO zeroes the epoch_us slot, so the RX never
// sees a stale stamp left behind by a previous burst when DPDK rotates
// this buffer through position 0 -> ... -> position k. Without this the
// Phase 3 latency p50/p99 would reflect mbuf pool rotation, not the
// real end-to-end latency.
// ===========================================================================
__global__ void stem_tx_update_burst_headers_kernel(
    uint8_t** gpu_bufs, const uint16_t* row_numbers, const uint16_t* source_ids,
    uint32_t pkts_in_burst, uint16_t header_offset,
    uint64_t epoch_us_for_first) {
  const uint32_t pkt_idx = blockIdx.x;
  if (pkt_idx >= pkts_in_burst) return;

  uint8_t* p = gpu_bufs[pkt_idx];
  if (p == nullptr) return;
  uint8_t* hdr = p + header_offset;

  if (threadIdx.x == 0) {
    const uint16_t rn = row_numbers[pkt_idx];
    const uint16_t sid = source_ids[pkt_idx];
    hdr[STEM_HDR_OFF_ROW_NUMBER_LO] = static_cast<uint8_t>(rn & 0xff);
    hdr[STEM_HDR_OFF_ROW_NUMBER_HI] = static_cast<uint8_t>((rn >> 8) & 0xff);
    hdr[STEM_HDR_OFF_SOURCE_ID_LO] = static_cast<uint8_t>(sid & 0xff);
    hdr[STEM_HDR_OFF_SOURCE_ID_HI] = static_cast<uint8_t>((sid >> 8) & 0xff);

    if (pkt_idx == 0) {
      if (epoch_us_for_first != 0) {
        for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
          hdr[STEM_HDR_OFF_EPOCH_US + i] =
              static_cast<uint8_t>((epoch_us_for_first >> (i * 8)) & 0xff);
        }
      }
    } else {
      // Zero epoch_us so a buffer that was once at position 0 doesn't
      // surface a stale stamp now that it has moved.
      for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
        hdr[STEM_HDR_OFF_EPOCH_US + i] = 0;
      }
    }
  }
}

void stem_tx_update_burst_headers(
    uint8_t** gpu_bufs, const uint16_t* row_numbers, const uint16_t* source_ids,
    uint32_t pkts_in_burst, uint16_t header_offset, uint64_t epoch_us_for_first,
    cudaStream_t stream) {
  if (pkts_in_burst == 0) { return; }
  stem_tx_update_burst_headers_kernel<<<pkts_in_burst, 32, 0, stream>>>(
      gpu_bufs, row_numbers, source_ids, pkts_in_burst, header_offset,
      epoch_us_for_first);
  check_cuda_launch("stem_tx_update_burst_headers_kernel");
}

// ===========================================================================
// Phase 2 RX -- shared device helper for STEM row layout.
//
// Layout (lifted verbatim from cpp/kernels.cu source_id_to_global_row):
//   - source IDs 0..3 fill rows 511..0 in interleaved groups of 4 ordered
//     as (3, 2, 1, 0) read top-to-bottom.
//   - source IDs 4..7 fill rows 512..1023 in interleaved groups of 4
//     ordered as (4, 5, 6, 7) read top-to-bottom.
// ===========================================================================
__device__ __forceinline__ int32_t
source_id_to_global_row(uint32_t source_id, uint32_t row_offset) {
  if (source_id < 4) {
    return 511 - static_cast<int32_t>(row_offset * 4 + source_id);
  }
  if (source_id < 8) {
    return 512 + static_cast<int32_t>(row_offset * 4 + (source_id - 4));
  }
  return -1;
}

// ===========================================================================
// Phase 2 RX -- header extraction
// ===========================================================================
__global__ void stem_extract_packet_headers_kernel(uint8_t** src_ptrs,
                                                   PacketHeaderInfo* headers,
                                                   uint32_t num_pkts) {
  const uint32_t pkt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pkt_idx >= num_pkts) return;

  PacketHeaderInfo h{};
  h.source_id = 0xFFFF;
  h.global_row = -1;
  h.epoch_us = 0;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src != nullptr) {
    h.row_number =
        (static_cast<uint16_t>(src[5]) << 8) | static_cast<uint16_t>(src[4]);
    h.source_id =
        (static_cast<uint16_t>(src[7]) << 8) | static_cast<uint16_t>(src[6]);
    h.frame_index = h.row_number / ROWS_PER_SOURCE;
    h.row_offset = h.row_number % ROWS_PER_SOURCE;
    h.global_row = static_cast<int16_t>(
        source_id_to_global_row(h.source_id, h.row_offset));
    for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
      h.epoch_us |= static_cast<uint64_t>(src[STEM_HDR_OFF_EPOCH_US + i])
                    << (i * 8);
    }
  }
  headers[pkt_idx] = h;
}

void stem_extract_packet_headers(uint8_t** src_ptrs, PacketHeaderInfo* headers,
                                 uint32_t num_pkts, cudaStream_t stream) {
  if (num_pkts == 0) { return; }
  const uint32_t threads = 256;
  const uint32_t blocks = (num_pkts + threads - 1) / threads;
  stem_extract_packet_headers_kernel<<<blocks, threads, 0, stream>>>(
      src_ptrs, headers, num_pkts);
  check_cuda_launch("stem_extract_packet_headers_kernel");
}

// ===========================================================================
// Phase 2 RX -- tile-readout placement gather.
//
// Ported 1:1 from cpp/kernels.cu::tile_geometry +
// gather_tile_packets_by_placement (upstream jerenner/stem_networking_bench
// `tiling` branch). This is the only RX gather path; the legacy row-based
// path was removed because LBNL's FPGA cannot emit row-shaped payloads.
//
// When the daqiri test TX still emits 7680 B row payloads, the kernel
// optionally fills the missing 256 samples of a 4096-sample tile by
// wrapping back to the start of the available payload.
// ===========================================================================
__device__ __forceinline__ bool tile_geometry(uint32_t tile_index,
                                              uint32_t& row_start,
                                              uint32_t& col_start,
                                              uint32_t& tile_height,
                                              uint32_t& tile_width) {
  constexpr uint32_t kZlpTileCols = TILE_ZLP_COLUMNS / TILE_ZLP_TILE_WIDTH;
  constexpr uint32_t kZlpTileRows = FRAME_HEIGHT / TILE_ZLP_TILE_HEIGHT;
  constexpr uint32_t kZlpTiles = kZlpTileCols * kZlpTileRows;
  constexpr uint32_t kCoreColumns = FRAME_WIDTH - TILE_ZLP_COLUMNS;
  constexpr uint32_t kCoreTileCols = kCoreColumns / TILE_CORE_TILE_WIDTH;
  constexpr uint32_t kCoreTileRows = FRAME_HEIGHT / TILE_CORE_TILE_HEIGHT;
  constexpr uint32_t kCoreTiles = kCoreTileCols * kCoreTileRows;

  if (tile_index < kZlpTiles) {
    const uint32_t tile_row = tile_index / kZlpTileCols;
    const uint32_t tile_col = tile_index % kZlpTileCols;
    row_start = tile_row * TILE_ZLP_TILE_HEIGHT;
    col_start = tile_col * TILE_ZLP_TILE_WIDTH;
    tile_height = TILE_ZLP_TILE_HEIGHT;
    tile_width = TILE_ZLP_TILE_WIDTH;
    return true;
  }

  const uint32_t core_index = tile_index - kZlpTiles;
  if (core_index >= kCoreTiles) { return false; }

  const uint32_t tile_row = core_index / kCoreTileCols;
  const uint32_t tile_col = core_index % kCoreTileCols;
  row_start = tile_row * TILE_CORE_TILE_HEIGHT;
  col_start = TILE_ZLP_COLUMNS + tile_col * TILE_CORE_TILE_WIDTH;
  tile_height = TILE_CORE_TILE_HEIGHT;
  tile_width = TILE_CORE_TILE_WIDTH;
  return true;
}

__global__ void stem_gather_tile_packets_by_placement_kernel(
    uint8_t** src_ptrs, const PacketPlacement* placements, uint8_t* dst_base,
    uint16_t available_payload_len, uint16_t header_len, uint32_t num_pkts,
    uint32_t frames, uint32_t frame_height, uint32_t frame_width,
    bool duplicate_prefix_to_simulate_tile_payload) {
  const uint32_t pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) { return; }

  const PacketPlacement placement = placements[pkt_idx];
  if (!placement.valid || placement.relative_frame >= frames) { return; }

  uint32_t row_start = 0;
  uint32_t col_start = 0;
  uint32_t tile_h = 0;
  uint32_t tile_w = 0;
  if (!tile_geometry(placement.tile_index, row_start, col_start, tile_h,
                     tile_w)) {
    return;
  }
  if (row_start + tile_h > frame_height || col_start + tile_w > frame_width) {
    return;
  }

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) { return; }

  const uint16_t* payload = reinterpret_cast<const uint16_t*>(src + header_len);
  uint16_t* output = reinterpret_cast<uint16_t*>(dst_base);
  const uint32_t available_samples = available_payload_len / sizeof(uint16_t);
  const uint64_t frame_offset =
      static_cast<uint64_t>(placement.relative_frame) *
      static_cast<uint64_t>(frame_height) * static_cast<uint64_t>(frame_width);

  for (uint32_t sample_idx = threadIdx.x; sample_idx < TILE_SAMPLES;
       sample_idx += blockDim.x) {
    uint32_t src_sample_idx = sample_idx;
    if (duplicate_prefix_to_simulate_tile_payload &&
        src_sample_idx >= available_samples) {
      src_sample_idx -= available_samples;
    }
    if (src_sample_idx >= available_samples) { continue; }

    const uint32_t local_row = sample_idx / tile_w;
    const uint32_t local_col = sample_idx - local_row * tile_w;
    if (local_row >= tile_h) { continue; }

    const uint64_t dst_idx = frame_offset +
                             static_cast<uint64_t>(row_start + local_row) *
                                 static_cast<uint64_t>(frame_width) +
                             static_cast<uint64_t>(col_start + local_col);
    output[dst_idx] = payload[src_sample_idx];
  }
}

void stem_gather_tile_packets_by_placement(
    uint8_t** src_ptrs, const PacketPlacement* placements, uint8_t* dst_base,
    uint16_t available_payload_len, uint16_t header_len, uint32_t num_pkts,
    uint32_t frames, uint32_t frame_height, uint32_t frame_width,
    bool duplicate_prefix_to_simulate_tile_payload, cudaStream_t stream) {
  if (num_pkts == 0) { return; }
  stem_gather_tile_packets_by_placement_kernel<<<num_pkts, 256, 0, stream>>>(
      src_ptrs, placements, dst_base, available_payload_len, header_len,
      num_pkts, frames, frame_height, frame_width,
      duplicate_prefix_to_simulate_tile_payload);
  check_cuda_launch("stem_gather_tile_packets_by_placement_kernel");
}

// ===========================================================================
// Phase 3 processor -- dark-frame subtract + valid-pixel mask, uint16 -> fp32
// ===========================================================================
__global__ void stem_dark_correct_uint16_to_float_kernel(
    const uint16_t* input, const float* dark_frame,
    const float* valid_pixel_mask, float* output, uint32_t frames,
    uint32_t frame_pixels, bool subtract_dark, bool apply_valid_pixel_mask) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels; pixel_idx += pixel_stride) {
    const float dark_value = subtract_dark ? dark_frame[pixel_idx] : 0.0f;
    const float mask_value =
        apply_valid_pixel_mask ? valid_pixel_mask[pixel_idx] : 1.0f;

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      output[idx] = (static_cast<float>(input[idx]) - dark_value) * mask_value;
    }
  }
}

void stem_dark_correct_uint16_to_float(
    const uint16_t* input, const float* dark_frame,
    const float* valid_pixel_mask, float* output, uint32_t frames,
    uint32_t height, uint32_t width, bool subtract_dark,
    bool apply_valid_pixel_mask, cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_dark_correct_uint16_to_float_kernel<<<blocks, threads, 0, stream>>>(
      input, dark_frame, valid_pixel_mask, output, frames,
      static_cast<uint32_t>(frame_pixels), subtract_dark,
      apply_valid_pixel_mask);
  check_cuda_launch("stem_dark_correct_uint16_to_float_kernel");
}

__global__ void stem_dark_correct_float_to_float_kernel(
    const float* input, const float* dark_frame, const float* valid_pixel_mask,
    float* output, uint32_t frames, uint32_t frame_pixels, bool subtract_dark,
    bool apply_valid_pixel_mask) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels; pixel_idx += pixel_stride) {
    const float dark_value = subtract_dark ? dark_frame[pixel_idx] : 0.0f;
    const float mask_value =
        apply_valid_pixel_mask ? valid_pixel_mask[pixel_idx] : 1.0f;

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      output[idx] = (input[idx] - dark_value) * mask_value;
    }
  }
}

void stem_dark_correct_float_to_float(
    const float* input, const float* dark_frame, const float* valid_pixel_mask,
    float* output, uint32_t frames, uint32_t height, uint32_t width,
    bool subtract_dark, bool apply_valid_pixel_mask, cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_dark_correct_float_to_float_kernel<<<blocks, threads, 0, stream>>>(
      input, dark_frame, valid_pixel_mask, output, frames,
      static_cast<uint32_t>(frame_pixels), subtract_dark,
      apply_valid_pixel_mask);
  check_cuda_launch("stem_dark_correct_float_to_float_kernel");
}

__global__ void stem_compute_frame_mean_float_kernel(const float* input,
                                                     float* mean,
                                                     uint32_t frames,
                                                     uint32_t frame_pixels) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels; pixel_idx += pixel_stride) {
    float sum = 0.0f;
    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      sum += input[idx];
    }
    mean[pixel_idx] = sum / static_cast<float>(frames);
  }
}

void stem_compute_frame_mean_float(const float* input, float* mean,
                                   uint32_t frames, uint32_t height,
                                   uint32_t width, cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_compute_frame_mean_float_kernel<<<blocks, threads, 0, stream>>>(
      input, mean, frames, static_cast<uint32_t>(frame_pixels));
  check_cuda_launch("stem_compute_frame_mean_float_kernel");
}

__device__ __forceinline__ float stem_median_from_small_window(float* values,
                                                               uint32_t count) {
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

__global__ void stem_apply_dynamic_half_column_mask_float_kernel(
    float* input, const float* batch_mean, uint32_t frames, uint32_t height,
    uint32_t width, uint32_t median_window_pixels, float threshold_ratio,
    float threshold_offset) {
  constexpr uint32_t kMaxMedianWindowPixels = 129;
  float window_values[kMaxMedianWindowPixels];

  const uint32_t frame_pixels = height * width;
  const uint32_t pixel_stride = blockDim.x * gridDim.x;
  const uint32_t half_height = height / 2;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels; pixel_idx += pixel_stride) {
    const uint32_t row = pixel_idx / width;
    const uint32_t col = pixel_idx - row * width;
    const uint32_t half_start = row < half_height ? 0 : half_height;
    const uint32_t half_end = row < half_height ? half_height : height;

    const uint32_t radius = median_window_pixels / 2;
    uint32_t row_start = row > radius ? row - radius : half_start;
    row_start = row_start < half_start ? half_start : row_start;
    uint32_t row_end = row + radius + 1;
    row_end = row_end > half_end ? half_end : row_end;

    uint32_t count = 0;
    for (uint32_t sample_row = row_start;
         sample_row < row_end && count < kMaxMedianWindowPixels; ++sample_row) {
      window_values[count++] =
          batch_mean[static_cast<uint64_t>(sample_row) * width + col];
    }
    if (count == 0) { continue; }

    const float local_median =
        stem_median_from_small_window(window_values, count);
    const float current_value = batch_mean[pixel_idx];
    const float threshold = local_median * threshold_ratio + threshold_offset;
    if (current_value <= threshold) { continue; }

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      input[idx] = 0.0f;
    }
  }
}

void stem_apply_dynamic_half_column_mask_float(
    float* input, const float* batch_mean, uint32_t frames, uint32_t height,
    uint32_t width, uint32_t median_window_pixels, float threshold_ratio,
    float threshold_offset, cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_apply_dynamic_half_column_mask_float_kernel<<<blocks, threads, 0,
                                                     stream>>>(
      input, batch_mean, frames, height, width, median_window_pixels,
      threshold_ratio, threshold_offset);
  check_cuda_launch("stem_apply_dynamic_half_column_mask_float_kernel");
}

__global__ void stem_sum_frames_float_to_frame_kernel(const float* input,
                                                      float* output,
                                                      uint32_t frames,
                                                      uint32_t frame_pixels) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels; pixel_idx += pixel_stride) {
    float sum = 0.0f;
    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      sum += input[idx];
    }
    output[pixel_idx] = sum;
  }
}

void stem_sum_frames_float_to_frame(const float* input, float* output,
                                    uint32_t frames, uint32_t height,
                                    uint32_t width, cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_sum_frames_float_to_frame_kernel<<<blocks, threads, 0, stream>>>(
      input, output, frames, static_cast<uint32_t>(frame_pixels));
  check_cuda_launch("stem_sum_frames_float_to_frame_kernel");
}

}  // namespace stem
