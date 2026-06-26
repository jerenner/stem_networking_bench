/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * STEM packet layout + frame geometry constants. Mirrors the packet format
 * understood by cpp/stem_receiver_op.h so that the daqiri TX produces wire
 * traffic that the existing Holoscan StemReceiverOp can assemble.
 */
#pragma once

#include <cstdint>

namespace stem {

// ---------------------------------------------------------------------------
// Frame geometry. Identical to cpp/stem_receiver_op.h.
//
//   frame  = FRAME_HEIGHT (1024) rows of FRAME_WIDTH (3840) uint16 samples
//          = 8 sources, each contributing ROWS_PER_SOURCE (128) rows
//          = 1024 packets per frame on the wire (one row per packet)
// ---------------------------------------------------------------------------
constexpr uint32_t FRAME_WIDTH      = 3840;
constexpr uint32_t FRAME_HEIGHT     = 1024;
constexpr uint32_t ROWS_PER_SOURCE  = 128;
constexpr uint32_t NUM_SOURCES_MAX  = 8;
constexpr uint32_t FRAME_SIZE_BYTES = FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t);

// ---------------------------------------------------------------------------
// Wire layout
//
//   [0,   42)  Eth + IPv4 + UDP                       (daqiri populates)
//   [42, 106)  STEM 64-byte custom header             (we stamp per packet)
//   [106, 7786) 7680-byte row payload (3840 * uint16)
//
// The total on-wire size is 7786 bytes per packet. With daqiri's
// header_size: 42 the reorder kernel sees exactly the same packet layout
// the existing cpp/kernels.cu gather kernel expects: src[0..3] reserved,
// src[4..5] = row_number (u16 LE), src[6..7] = source_id (u16 LE),
// src[64..7743] = payload.
// ---------------------------------------------------------------------------
constexpr uint32_t L4_HEADER_SIZE     = 42;       // Eth(14) + IPv4(20) + UDP(8)
constexpr uint32_t STEM_HEADER_SIZE   = 64;       // custom application header
constexpr uint32_t STEM_PAYLOAD_SIZE  = FRAME_WIDTH * sizeof(uint16_t);  // 7680
constexpr uint32_t STEM_PACKET_BYTES  = L4_HEADER_SIZE + STEM_HEADER_SIZE + STEM_PAYLOAD_SIZE;

// daqiri TX config keeps header_size=42 and packs the 64B STEM header
// + 7680B row payload into the "payload" section.
constexpr uint32_t TX_DAQIRI_HEADER_SIZE  = L4_HEADER_SIZE;                          // 42
constexpr uint32_t TX_DAQIRI_PAYLOAD_SIZE = STEM_HEADER_SIZE + STEM_PAYLOAD_SIZE;     // 7744

// Within the daqiri "payload" buffer (which starts at offset 42 of the wire),
// the STEM custom header occupies bytes [0, 64).
constexpr uint32_t STEM_HEADER_OFFSET_IN_PAYLOAD = 0;
constexpr uint32_t STEM_DATA_OFFSET_IN_PAYLOAD   = STEM_HEADER_SIZE;

// Byte offsets into the STEM custom header (0-based within the 64-byte header).
constexpr uint32_t STEM_HDR_OFF_ROW_NUMBER_LO = 4;   // u16 LE row_number
constexpr uint32_t STEM_HDR_OFF_ROW_NUMBER_HI = 5;
constexpr uint32_t STEM_HDR_OFF_SOURCE_ID_LO  = 6;   // u16 LE source_id
constexpr uint32_t STEM_HDR_OFF_SOURCE_ID_HI  = 7;
// Live latency stamping: epoch_us (uint64 LE) at bytes [16, 24) of STEM hdr.
// Only stamped for the first packet of each frame (source_id==0, row_offset==0).
constexpr uint32_t STEM_HDR_OFF_EPOCH_US      = 16;

// row_number is a 16-bit field that wraps every 16384 rows = 128 frames
// (each frame is 128 rows per source). See gather_packets_kernel in
// cpp/kernels.cu for the sequence-to-frame math.
constexpr uint32_t ROW_NUMBER_WRAP   = 16384;
constexpr uint32_t FRAMES_PER_WRAP   = ROW_NUMBER_WRAP / ROWS_PER_SOURCE;  // 128

// One frame = NUM_SOURCES_MAX sources * ROWS_PER_SOURCE rows = 1024 packets.
constexpr uint32_t PACKETS_PER_FRAME_FULL = NUM_SOURCES_MAX * ROWS_PER_SOURCE;

}  // namespace stem
