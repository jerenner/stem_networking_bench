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
 *   - stem_extract_packet_headers (port of extract_packet_headers)
 *   - stem_gather_tile_packets_by_placement (port of
 *     gather_tile_packets_by_placement; the only RX gather kernel since
 *     LBNL's FPGA emits tile-shaped payloads only)
 *
 * Phase 3 (processor) uses:
 *   - stem_dark_correct_uint16_to_float (port of dark_correct_uint16_to_float)
 *   - stem_compute_frame_mean_float / stem_apply_dynamic_half_column_mask_float
 *   - stem_sum_frames_float_to_frame
 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

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
  int16_t global_row;  // -1 means invalid source_id
  uint64_t epoch_us;
};

struct PacketPlacement {
  int16_t global_row;      // row within a 1024-tall frame, -1 = invalid
  uint16_t tile_index;     // tile id within a frame (tile readout only)
  uint8_t relative_frame;  // index within the current batch of frames
  uint8_t valid;           // 0/1
};

// ---------------------------------------------------------------------------
// Tile readout geometry (mirrors cpp/kernels.cu::tile_geometry from upstream
// jerenner/stem_networking_bench `tiling` branch). A 1024 x 3840 frame is
// partitioned into 960 tiles: 192 ZLP tiles (32 x 128) in cols [0, 768) and
// 768 core tiles (128 x 32) in cols [768, 3840). Each tile carries 4096
// uint16 samples (= TILE_PAYLOAD_BYTES = 8192 B). 8 sources contribute 120
// tile packets each => 960 tile packets per frame.
// ---------------------------------------------------------------------------
constexpr uint32_t TILE_ZLP_COLUMNS = 192u * 4u;  // 768
constexpr uint32_t TILE_ZLP_TILE_WIDTH = 32u;
constexpr uint32_t TILE_ZLP_TILE_HEIGHT = 128u;
constexpr uint32_t TILE_CORE_TILE_WIDTH = 128u;
constexpr uint32_t TILE_CORE_TILE_HEIGHT = 32u;
constexpr uint32_t TILE_SAMPLES =
    TILE_ZLP_TILE_WIDTH * TILE_ZLP_TILE_HEIGHT;  // 4096
constexpr uint32_t TILE_PAYLOAD_BYTES =
    TILE_SAMPLES * sizeof(uint16_t);  // 8192
constexpr uint32_t FULL_FRAME_TILE_PACKETS = 960u;
constexpr uint32_t TILE_PACKETS_PER_SOURCE =
    FULL_FRAME_TILE_PACKETS / 8u;  // 120

// ---------------------------------------------------------------------------
// Host helpers (Phase 1 TX)
// ---------------------------------------------------------------------------

// Stamp the STEM 64-byte custom header into a HOST-side packet template
// buffer. `stem_header_dst` must point at the start of the 64-byte STEM
// header (i.e. the byte immediately after the 42-byte Eth+IPv4+UDP header).
// Bytes 0-3, 8-15, 24-63 are zeroed. Bytes 4-5 = row_number (u16 LE),
// bytes 6-7 = source_id (u16 LE), bytes 16-23 = epoch_us (u64 LE) for the
// optional Phase 3 latency stamping path.
void stem_tx_stamp_packet(uint8_t* stem_header_dst, uint16_t row_number,
                          uint16_t source_id, uint64_t epoch_us);

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
void stem_tx_update_burst_headers(
    uint8_t** gpu_bufs, const uint16_t* row_numbers, const uint16_t* source_ids,
    uint32_t pkts_in_burst, uint16_t header_offset, uint64_t epoch_us_for_first,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Device helpers (Phase 2 RX)
//
// Same APIs as cpp/kernels.{cu,cuh}, ported into the stem:: namespace so the
// daqiri RX assembles batched [frames_per_tensor, 1024, 3840] uint16 tensors
// byte-for-byte equivalent to the Holoscan StemReceiverOp output.
// ---------------------------------------------------------------------------
void stem_extract_packet_headers(uint8_t** src_ptrs, PacketHeaderInfo* headers,
                                 uint32_t num_pkts, cudaStream_t stream);

// Tile-readout placement gather: scatter each packet's available payload
// into its (ZLP or core) tile within a full
// [frames, frame_height, frame_width] uint16 plane. Mirrors
// cpp/kernels.cu::gather_tile_packets_by_placement from upstream/tiling.
//
// This is the only RX gather path now that LBNL's FPGA emits tile-shaped
// payloads exclusively; the legacy row-based gather has been removed.
//
// `available_payload_len` is the wire payload length actually present in
// each packet (e.g. 7680 for stem_daqiri_tx, which still emits row-sized
// payloads for testing). When `duplicate_prefix_to_simulate_tile_payload`
// is true and available_payload_len < TILE_PAYLOAD_BYTES, the missing tail
// samples are filled by wrapping back to the front of the payload, so the
// daqiri TX can drive the tile RX in a self-test loop.
void stem_gather_tile_packets_by_placement(
    uint8_t** src_ptrs, const PacketPlacement* placements, uint8_t* dst_base,
    uint16_t available_payload_len, uint16_t header_len, uint32_t num_pkts,
    uint32_t frames, uint32_t frame_height, uint32_t frame_width,
    bool duplicate_prefix_to_simulate_tile_payload, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Phase 3 processor: dark-frame subtraction + valid-pixel mask in one fused
// kernel. Operates on a [frames, height, width] uint16 input and writes a
// [frames, height, width] float32 output. Both arrays live on the GPU.
// ---------------------------------------------------------------------------
void stem_dark_correct_uint16_to_float(
    const uint16_t* input, const float* dark_frame,
    const float* valid_pixel_mask, float* output, uint32_t frames,
    uint32_t height, uint32_t width, bool subtract_dark,
    bool apply_valid_pixel_mask, cudaStream_t stream);

void stem_dark_correct_float_to_float(
    const float* input, const float* dark_frame, const float* valid_pixel_mask,
    float* output, uint32_t frames, uint32_t height, uint32_t width,
    bool subtract_dark, bool apply_valid_pixel_mask, cudaStream_t stream);

void stem_compute_frame_mean_float(const float* input, float* mean,
                                   uint32_t frames, uint32_t height,
                                   uint32_t width, cudaStream_t stream);

void stem_apply_dynamic_half_column_mask_float(
    float* input, const float* batch_mean, uint32_t frames, uint32_t height,
    uint32_t width, uint32_t median_window_pixels, float threshold_ratio,
    float threshold_offset, cudaStream_t stream);

void stem_sum_frames_float_to_frame(const float* input, float* output,
                                    uint32_t frames, uint32_t height,
                                    uint32_t width, cudaStream_t stream);

}  // namespace stem
