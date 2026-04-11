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

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>

struct PacketDebugSummary {
  uint16_t row_number;
  uint16_t source_id;
  uint16_t frame_index;
  int16_t global_row;
  uint16_t nonzero_count;
  uint16_t first_nonzero_col;
  uint16_t first_nonzero_value;
  uint16_t second_nonzero_col;
  uint16_t second_nonzero_value;
  uint16_t max_value_col;
  uint16_t max_value;
};

void populate_packets(uint8_t** gpu_bufs, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                      cudaStream_t stream);

void copy_headers(uint8_t** gpu_bufs, void* header, uint16_t hdr_size, uint32_t num_pkts,
                  cudaStream_t stream);

void populate_packets_from_frame(uint8_t* frame_buf, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                                 cudaStream_t stream);

void gather_packets(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, uint64_t base_absolute_row, cudaStream_t stream);

void summarize_packets(uint8_t** src_ptrs,
                       PacketDebugSummary* summaries,
                       uint16_t payload_len,
                       uint16_t header_len,
                       uint32_t num_pkts,
                       cudaStream_t stream);
