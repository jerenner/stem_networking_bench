/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CUDA kernels for the daqiri-based STEM pipeline. See stem_kernels.h for
 * the public surface and which phase each kernel is used by.
 */

#include "stem_kernels.h"
#include "stem_packet.h"

#include <cstring>

namespace stem {

// ===========================================================================
// Phase 1 TX -- host-side STEM header stamp
// ===========================================================================
void stem_tx_stamp_packet(uint8_t* stem_header_dst,
                          uint16_t row_number,
                          uint16_t source_id,
                          uint64_t epoch_us) {
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
__global__ void stem_tx_update_burst_headers_kernel(uint8_t** gpu_bufs,
                                                    const uint16_t* row_numbers,
                                                    const uint16_t* source_ids,
                                                    uint32_t pkts_in_burst,
                                                    uint16_t header_offset,
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
    hdr[STEM_HDR_OFF_SOURCE_ID_LO]  = static_cast<uint8_t>(sid & 0xff);
    hdr[STEM_HDR_OFF_SOURCE_ID_HI]  = static_cast<uint8_t>((sid >> 8) & 0xff);

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

void stem_tx_update_burst_headers(uint8_t** gpu_bufs,
                                  const uint16_t* row_numbers,
                                  const uint16_t* source_ids,
                                  uint32_t pkts_in_burst,
                                  uint16_t header_offset,
                                  uint64_t epoch_us_for_first,
                                  cudaStream_t stream) {
  if (pkts_in_burst == 0) { return; }
  stem_tx_update_burst_headers_kernel<<<pkts_in_burst, 32, 0, stream>>>(
      gpu_bufs, row_numbers, source_ids, pkts_in_burst,
      header_offset, epoch_us_for_first);
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
__device__ __forceinline__
int32_t source_id_to_global_row(uint32_t source_id, uint32_t row_offset) {
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
    h.row_number = (static_cast<uint16_t>(src[5]) << 8) |
                   static_cast<uint16_t>(src[4]);
    h.source_id  = (static_cast<uint16_t>(src[7]) << 8) |
                   static_cast<uint16_t>(src[6]);
    h.frame_index = h.row_number / ROWS_PER_SOURCE;
    h.row_offset  = h.row_number % ROWS_PER_SOURCE;
    h.global_row = static_cast<int16_t>(
        source_id_to_global_row(h.source_id, h.row_offset));
    for (uint32_t i = 0; i < sizeof(uint64_t); ++i) {
      h.epoch_us |= static_cast<uint64_t>(src[STEM_HDR_OFF_EPOCH_US + i]) << (i * 8);
    }
  }
  headers[pkt_idx] = h;
}

void stem_extract_packet_headers(uint8_t** src_ptrs,
                                 PacketHeaderInfo* headers,
                                 uint32_t num_pkts,
                                 cudaStream_t stream) {
  if (num_pkts == 0) { return; }
  const uint32_t threads = 256;
  const uint32_t blocks = (num_pkts + threads - 1) / threads;
  stem_extract_packet_headers_kernel<<<blocks, threads, 0, stream>>>(
      src_ptrs, headers, num_pkts);
}

// ===========================================================================
// Phase 2 RX -- gather packets by parsing the STEM header on the GPU
// (legacy path, count-based aggregation).
// ===========================================================================
__global__ void stem_gather_packets_kernel(uint8_t** src_ptrs,
                                           uint8_t* dst_base,
                                           uint16_t payload_len,
                                           uint16_t header_len,
                                           uint32_t num_pkts,
                                           uint32_t max_rows,
                                           uint64_t base_absolute_row) {
  const uint32_t pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) return;

  const uint16_t row_number = (static_cast<uint16_t>(src[5]) << 8) |
                              static_cast<uint16_t>(src[4]);
  const uint32_t source_id = (static_cast<uint16_t>(src[7]) << 8) |
                             static_cast<uint16_t>(src[6]);

  const uint32_t frame_idx = row_number / ROWS_PER_SOURCE;
  const uint32_t row_offset = row_number % ROWS_PER_SOURCE;

  const int32_t global_row = source_id_to_global_row(source_id, row_offset);
  if (global_row < 0) return;

  const uint32_t rows_per_frame = FRAME_HEIGHT;
  const uint32_t base_frame_mod =
      static_cast<uint32_t>((base_absolute_row / rows_per_frame) % FRAMES_PER_WRAP);
  const uint32_t batch_frames = max_rows / rows_per_frame;
  const uint32_t relative_frame =
      (frame_idx + FRAMES_PER_WRAP - base_frame_mod) % FRAMES_PER_WRAP;
  if (relative_frame >= batch_frames) return;

  const int64_t target_row_1d =
      static_cast<int64_t>(relative_frame) * rows_per_frame + global_row;
  if (target_row_1d < 0 ||
      target_row_1d >= static_cast<int64_t>(max_rows)) return;

  const uint8_t* payload_src = src + header_len;
  uint8_t* dst = dst_base + target_row_1d * payload_len;

  // The wire packet is only 2-byte-aligned because Eth(14)+IP(20)+UDP(8)
  // = 42 bytes leaves the payload offset at 42 from the start. Using uint4
  // or uint32_t would trip a CUDA Misaligned Address fault; uint16_t is
  // safe and the payload is uint16 samples anyway.
  const uint16_t* src16 = reinterpret_cast<const uint16_t*>(payload_src);
  uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
  const int unroll_len = payload_len / sizeof(uint16_t);
  for (int i = threadIdx.x; i < unroll_len; i += blockDim.x) {
    dst16[i] = src16[i];
  }
}

void stem_gather_packets(uint8_t** src_ptrs,
                         uint8_t* dst_base,
                         uint16_t payload_len,
                         uint16_t header_len,
                         uint32_t num_pkts,
                         uint32_t max_rows,
                         uint64_t base_absolute_row,
                         cudaStream_t stream) {
  if (num_pkts == 0) { return; }
  stem_gather_packets_kernel<<<num_pkts, 256, 0, stream>>>(
      src_ptrs, dst_base, payload_len, header_len,
      num_pkts, max_rows, base_absolute_row);
}

// ===========================================================================
// Phase 2 RX -- gather packets using host-precomputed placements (modern
// slack-based batching path; see cpp/stem_receiver_op.h).
// ===========================================================================
__global__ void stem_gather_packets_by_placement_kernel(uint8_t** src_ptrs,
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
      static_cast<int64_t>(placement.relative_frame) * FRAME_HEIGHT +
      placement.global_row;
  if (target_row_1d < 0 ||
      target_row_1d >= static_cast<int64_t>(max_rows)) return;

  uint8_t* src = src_ptrs[pkt_idx];
  if (src == nullptr) return;

  const uint8_t* payload_src = src + header_len;
  uint8_t* dst = dst_base + target_row_1d * payload_len;

  const uint16_t* src16 = reinterpret_cast<const uint16_t*>(payload_src);
  uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
  const int unroll_len = payload_len / sizeof(uint16_t);
  for (int i = threadIdx.x; i < unroll_len; i += blockDim.x) {
    dst16[i] = src16[i];
  }
}

void stem_gather_packets_by_placement(uint8_t** src_ptrs,
                                      const PacketPlacement* placements,
                                      uint8_t* dst_base,
                                      uint16_t payload_len,
                                      uint16_t header_len,
                                      uint32_t num_pkts,
                                      uint32_t max_rows,
                                      cudaStream_t stream) {
  if (num_pkts == 0) { return; }
  stem_gather_packets_by_placement_kernel<<<num_pkts, 256, 0, stream>>>(
      src_ptrs, placements, dst_base, payload_len, header_len,
      num_pkts, max_rows);
}

// ===========================================================================
// Phase 3 processor -- dark-frame subtract + valid-pixel mask, uint16 -> fp32
// ===========================================================================
__global__ void stem_dark_correct_uint16_to_float_kernel(const uint16_t* input,
                                                         const float* dark_frame,
                                                         const float* valid_pixel_mask,
                                                         float* output,
                                                         uint32_t frames,
                                                         uint32_t frame_pixels,
                                                         bool subtract_dark,
                                                         bool apply_valid_pixel_mask) {
  const uint32_t pixel_stride = blockDim.x * gridDim.x;

  for (uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       pixel_idx < frame_pixels;
       pixel_idx += pixel_stride) {
    const float dark_value = subtract_dark ? dark_frame[pixel_idx] : 0.0f;
    const float mask_value = apply_valid_pixel_mask
                                 ? valid_pixel_mask[pixel_idx]
                                 : 1.0f;

    for (uint32_t frame = 0; frame < frames; ++frame) {
      const uint64_t idx =
          static_cast<uint64_t>(frame) * frame_pixels + pixel_idx;
      output[idx] = (static_cast<float>(input[idx]) - dark_value) * mask_value;
    }
  }
}

void stem_dark_correct_uint16_to_float(const uint16_t* input,
                                       const float* dark_frame,
                                       const float* valid_pixel_mask,
                                       float* output,
                                       uint32_t frames,
                                       uint32_t height,
                                       uint32_t width,
                                       bool subtract_dark,
                                       bool apply_valid_pixel_mask,
                                       cudaStream_t stream) {
  const uint64_t frame_pixels = static_cast<uint64_t>(height) * width;
  const uint64_t total_values = static_cast<uint64_t>(frames) * frame_pixels;
  if (total_values == 0) { return; }

  const uint32_t threads = 256;
  const uint64_t required_blocks = (frame_pixels + threads - 1) / threads;
  const uint32_t blocks = static_cast<uint32_t>(
      required_blocks > 65535ULL ? 65535ULL : required_blocks);

  stem_dark_correct_uint16_to_float_kernel<<<blocks, threads, 0, stream>>>(
      input, dark_frame, valid_pixel_mask, output,
      frames, static_cast<uint32_t>(frame_pixels),
      subtract_dark, apply_valid_pixel_mask);
}

}  // namespace stem
