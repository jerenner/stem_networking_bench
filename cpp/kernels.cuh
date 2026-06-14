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

struct PacketHeaderInfo {
  uint16_t row_number;
  uint16_t source_id;
  uint16_t frame_index;
  uint16_t row_offset;
  int16_t global_row;
};

struct PacketPlacement {
  uint16_t relative_frame;
  int16_t global_row;
  uint16_t tile_index;
  uint16_t valid;
};

void populate_packets(uint8_t** gpu_bufs, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                      cudaStream_t stream);

void copy_headers(uint8_t** gpu_bufs, void* header, uint16_t hdr_size, uint32_t num_pkts,
                  cudaStream_t stream);

void populate_packets_from_frame(uint8_t* frame_buf, uint16_t pkt_len, uint32_t num_pkts, uint16_t offset,
                                 cudaStream_t stream);

void gather_packets(uint8_t** src_ptrs, uint8_t* dst_base, uint16_t payload_len, uint16_t header_len, uint32_t num_pkts, uint32_t max_rows, uint64_t base_absolute_row, cudaStream_t stream);

void extract_packet_headers(uint8_t** src_ptrs,
                            PacketHeaderInfo* headers,
                            uint32_t num_pkts,
                            cudaStream_t stream);

void gather_packets_by_placement(uint8_t** src_ptrs,
                                 const PacketPlacement* placements,
                                 uint8_t* dst_base,
                                 uint16_t payload_len,
                                 uint16_t header_len,
                                 uint32_t num_pkts,
                                 uint32_t max_rows,
                                 cudaStream_t stream);

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
                                      cudaStream_t stream);

void summarize_packets(uint8_t** src_ptrs,
                       PacketDebugSummary* summaries,
                       uint16_t payload_len,
                       uint16_t header_len,
                       uint32_t num_pkts,
                       cudaStream_t stream);

void dark_correct_uint16_to_float(const uint16_t* input,
                                  const float* dark_frame,
                                  const float* valid_pixel_mask,
                                  float* output,
                                  uint32_t frames,
                                  uint32_t height,
                                  uint32_t width,
                                  bool subtract_dark,
                                  bool apply_valid_pixel_mask,
                                  cudaStream_t stream);

void compute_frame_mean_float(const float* input,
                              float* mean,
                              uint32_t frames,
                              uint32_t height,
                              uint32_t width,
                              cudaStream_t stream);

void compute_blr_baseline_float(const float* input,
                                float* baseline,
                                uint32_t frames,
                                uint32_t height,
                                uint32_t width,
                                uint32_t blr_rows,
                                uint32_t zlp_width,
                                uint32_t zlp_group_columns,
                                uint32_t core_group_columns,
                                cudaStream_t stream);

void subtract_blr_baseline_float(float* input,
                                 const float* baseline,
                                 uint32_t frames,
                                 uint32_t height,
                                 uint32_t width,
                                 uint32_t zlp_width,
                                 uint32_t zlp_group_columns,
                                 uint32_t core_group_columns,
                                 cudaStream_t stream);

void apply_dynamic_half_column_mask_float(float* input,
                                          const float* batch_mean,
                                          uint32_t frames,
                                          uint32_t height,
                                          uint32_t width,
                                          uint32_t median_window_pixels,
                                          float threshold_ratio,
                                          float threshold_offset,
                                          uint32_t excluded_edge_rows,
                                          bool two_sided,
                                          cudaStream_t stream);

void apply_valid_pixel_mask_float(float* input,
                                  const float* valid_pixel_mask,
                                  uint32_t frames,
                                  uint32_t height,
                                  uint32_t width,
                                  cudaStream_t stream);
