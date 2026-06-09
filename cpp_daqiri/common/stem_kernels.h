/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CUDA kernels shared by the daqiri-based STEM TX and RX binaries. Ports the
 * compute kernels from cpp/kernels.cu so the daqiri pipeline produces the
 * same wire layout and assembles the same in-memory frame tensors as the
 * original Holoscan StemReceiverOp.
 *
 * Phase 1 (TX) uses:
 *   - stem_tx_stamp_packet (host helper that fills a host-side packet
 *     template's STEM header at offsets 4-7)
 *   - stem_tx_update_burst_headers_kernel (per-burst GPU update of the STEM
 *     header for each packet in a burst, used in Phase 3 for varying row
 *     numbers; harmless to ship in Phase 1)
 *
 * Phase 2 (RX) uses:
 *   - stem_gather_packets (port of gather_packets in cpp/kernels.cu)
 *   - stem_gather_packets_by_placement (port of gather_packets_by_placement)
 *   - stem_extract_packet_headers (port of extract_packet_headers)
 *
 * Phase 3 (processor) uses:
 *   - stem_dark_correct_uint16_to_float (port of dark_correct_uint16_to_float)
 */
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace stem {

// ---------------------------------------------------------------------------
// Phase 2 RX-side packet metadata, kept in lock-step with the Holoscan
// definitions so the two pipelines produce identical batched tensors.
// ---------------------------------------------------------------------------
struct PacketHeaderInfo {
  uint16_t row_number;
  uint16_t source_id;
  uint16_t frame_index;
  uint16_t row_offset;
  int16_t  global_row;  // -1 means invalid source_id
  uint64_t epoch_us;
};

struct PacketPlacement {
  int16_t  global_row;       // row within a 1024-tall frame, -1 = invalid
  uint8_t  relative_frame;   // index within the current batch of frames
  uint8_t  valid;            // 0/1
};

// ---------------------------------------------------------------------------
// Host helpers (Phase 1 TX)
// ---------------------------------------------------------------------------

// Stamp the STEM 64-byte custom header into a HOST-side packet template
// buffer. `stem_header_dst` must point at the start of the 64-byte STEM
// header (i.e. the byte immediately after the 42-byte Eth+IPv4+UDP header).
// Bytes 0-3, 8-15, 24-63 are zeroed. Bytes 4-5 = row_number (u16 LE),
// bytes 6-7 = source_id (u16 LE), bytes 16-23 = epoch_us (u64 LE) for the
// optional Phase 3 latency stamping path.
void stem_tx_stamp_packet(uint8_t* stem_header_dst,
                          uint16_t row_number,
                          uint16_t source_id,
                          uint64_t epoch_us);

// ---------------------------------------------------------------------------
// Device helpers (Phase 3 TX header update). Updates the STEM header bytes
// of `pkts_in_burst` packets in-place on the GPU. Each packet's gpu_bufs[i]
// pointer must be the start of the wire packet (i.e. byte 0 of the Eth
// header). `header_offset` is the offset from the start of the wire packet
// to the start of the STEM header (typically 42).
//
// `row_numbers[i]` and `source_ids[i]` are read for each packet i; both
// arrays must live in GPU memory. `epoch_us_for_first` is stamped into the
// epoch_us slot of packet 0 only (and only if it is non-zero).
// ---------------------------------------------------------------------------
void stem_tx_update_burst_headers(uint8_t** gpu_bufs,
                                  const uint16_t* row_numbers,
                                  const uint16_t* source_ids,
                                  uint32_t pkts_in_burst,
                                  uint16_t header_offset,
                                  uint64_t epoch_us_for_first,
                                  cudaStream_t stream);

// ---------------------------------------------------------------------------
// Device helpers (Phase 2 RX)
//
// Same APIs as cpp/kernels.{cu,cuh}, ported into the stem:: namespace so the
// daqiri RX assembles batched [frames_per_tensor, 1024, 3840] uint16 tensors
// byte-for-byte equivalent to the Holoscan StemReceiverOp output.
// ---------------------------------------------------------------------------
void stem_extract_packet_headers(uint8_t** src_ptrs,
                                 PacketHeaderInfo* headers,
                                 uint32_t num_pkts,
                                 cudaStream_t stream);

void stem_gather_packets(uint8_t** src_ptrs,
                         uint8_t* dst_base,
                         uint16_t payload_len,
                         uint16_t header_len,
                         uint32_t num_pkts,
                         uint32_t max_rows,
                         uint64_t base_absolute_row,
                         cudaStream_t stream);

void stem_gather_packets_by_placement(uint8_t** src_ptrs,
                                      const PacketPlacement* placements,
                                      uint8_t* dst_base,
                                      uint16_t payload_len,
                                      uint16_t header_len,
                                      uint32_t num_pkts,
                                      uint32_t max_rows,
                                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 3 processor: dark-frame subtraction + valid-pixel mask in one fused
// kernel. Operates on a [frames, height, width] uint16 input and writes a
// [frames, height, width] float32 output. Both arrays live on the GPU.
// ---------------------------------------------------------------------------
void stem_dark_correct_uint16_to_float(const uint16_t* input,
                                       const float* dark_frame,
                                       const float* valid_pixel_mask,
                                       float* output,
                                       uint32_t frames,
                                       uint32_t height,
                                       uint32_t width,
                                       bool subtract_dark,
                                       bool apply_valid_pixel_mask,
                                       cudaStream_t stream);

}  // namespace stem
