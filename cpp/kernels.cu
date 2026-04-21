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
