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

__global__ void gather_packets_kernel(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, uint64_t base_absolute_row) {

  int pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  uint8_t* src = src_ptrs[pkt_idx];
  
  // Extract 16-bit row number from custom header (which wraps every 16384 rows)
  uint16_t row_number = ((uint16_t)src[5] << 8) | (uint16_t)src[4];
  
  // Reconstruct chronological index to handle exact wrapping
  uint64_t expected_abs = pkt_idx;
  int64_t diff = (int64_t)row_number - (int64_t)(expected_abs % 16384);
  
  if (diff > 8192) diff -= 16384;
  if (diff < -8192) diff += 16384;
  
  int64_t absolute_idx = (int64_t)expected_abs + diff;
  if (absolute_idx < 0) return;
  
  uint64_t bunch_idx = absolute_idx / 131072;
  uint64_t in_bunch_idx = absolute_idx % 131072;
  
  uint32_t source_id = in_bunch_idx / 16384;
  uint32_t local_seq = in_bunch_idx % 16384;
  
  uint32_t frame_idx = local_seq / 128;
  uint32_t row_offset = local_seq % 128;
  
  uint32_t global_row = 0;
  if (source_id == 0) global_row = row_offset;
  else if (source_id == 1) global_row = 128 + row_offset;
  else if (source_id == 2) global_row = 256 + row_offset;
  else if (source_id == 3) global_row = 384 + row_offset;
  else if (source_id == 4) global_row = 1023 - row_offset;
  else if (source_id == 5) global_row = 895 - row_offset;
  else if (source_id == 6) global_row = 767 - row_offset;
  else if (source_id == 7) global_row = 639 - row_offset;
  
  uint64_t global_frame_idx = bunch_idx * 128 + frame_idx;
  uint64_t target_row_1d = global_frame_idx * 1024 + global_row;
  
  if (target_row_1d >= max_rows) return;

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
