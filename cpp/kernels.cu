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

__global__ void gather_packets_kernel(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows) {
  int pkt_idx = blockIdx.x;
  if (pkt_idx >= num_pkts) return;

  uint8_t* src = src_ptrs[pkt_idx];
  
  // Extract row number from bytes 4 and 5 of the custom header (little-endian)
  uint16_t row_number = ((uint16_t)src[5] << 8) | (uint16_t)src[4];
  
  // Map row to the correct offset within the output tensor
  uint32_t target_row = row_number % max_rows;

  uint8_t* payload_src = src + header_len;
  uint8_t* dst = dst_base + target_row * payload_len;

  // Optimized vectorized copy (7680 bytes is a multiple of 16)
  if (payload_len % 16 == 0) {
      uint4* src4 = (uint4*)payload_src;
      uint4* dst4 = (uint4*)dst;
      int unroll_len = payload_len / sizeof(uint4);
      for (int i = threadIdx.x; i < unroll_len; i += blockDim.x) {
        dst4[i] = src4[i];
      }
  } else {
      for (int i = threadIdx.x; i < payload_len; i += blockDim.x) {
        dst[i] = payload_src[i];
      }
  }
}

void gather_packets(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, cudaStream_t stream) {
  gather_packets_kernel<<<num_pkts, 256, 0, stream>>>(src_ptrs, dst_base, payload_len, header_len, num_pkts, max_rows);
}
