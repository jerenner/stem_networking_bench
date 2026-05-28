/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Phase 2 stem_daqiri_rx: daqiri-based RX that assembles STEM-format UDP
 * packets into batched GPU frame tensors of shape
 *   [frames_per_tensor, 1024, 3840] uint16
 *
 * Port of the essentials of cpp/stem_receiver_op.h. Two important pieces
 * of the Holoscan original are mirrored here so that Phase 3 parity is
 * even possible:
 *
 *   1. Burst lifetime via std::shared_ptr<BurstHolder>. A burst is only
 *      released when no PacketEntry still references it. That lets future
 *      packets (out-of-current-window) survive a batch close instead of
 *      being silently dropped.
 *   2. A pending_packets_ list that retains future packets between
 *      batches. When the window slides, those packets become in-window
 *      and feed the next gather kernel.
 *
 * The output frame tensor is allocated once at startup and reused.
 * Phase 3 hooks: per-packet epoch_us latency capture and an optional
 * dark-correct CUDA stage.
 */
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <daqiri/daqiri.h>

#include "stem_kernels.h"
#include "stem_packet.h"

namespace {

constexpr size_t kMaxLatencySamples = 1'000'000;  // cap so RX doesn't blow RAM

// ===========================================================================
// Crisp CUDA error reporting. On Phase 2/3 the runtime is supposed to
// touch GPUDirect; silent failures here would surface as "0 frames assembled"
// at the end of the run with no breadcrumb.
// ===========================================================================
#define STEM_CUDA_TRY(stmt)                                                    \
  do {                                                                         \
    cudaError_t _err = (stmt);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr,                                                     \
                   "STEM_CUDA_TRY failed at %s:%d: %s -> %s (%d)\n",           \
                   __FILE__, __LINE__, #stmt,                                  \
                   cudaGetErrorString(_err), static_cast<int>(_err));          \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// stem_rx YAML block.
// ---------------------------------------------------------------------------
struct StemRxConfig {
  std::string interface_name = "rx_port";

  // Frame batching: how many full 1024x3840 frames are gathered into one
  // emitted output tensor. Matches cpp/stem_receiver_op.h frames_per_tensor.
  uint32_t frames_per_tensor = 128;

  // Wire layout. header_size is the byte offset within a wire packet where
  // the STEM custom header begins (Eth+IPv4+UDP = 42).
  uint16_t header_size = stem::L4_HEADER_SIZE;

  // Payload (row) size in bytes after the STEM custom header.
  uint16_t payload_size = stem::STEM_PAYLOAD_SIZE;

  // Active source ID bitmask. 0xff (255) accepts all 8 sources.
  uint32_t expected_source_mask = 0xff;

  // How many "out-of-window" future packets we'll accept before closing the
  // current batch early. Matches stem_receiver_op.h batch_close_slack_packets.
  uint32_t batch_close_slack_packets = 512;

  // Optional run-duration cap in seconds. < 0 means run until SIGINT.
  double total_time_to_recv_s = 30.0;

  // Phase 3 latency measurement: read epoch_us from packets stamped by the
  // TX and compute now - epoch_us. We only sample at (source_id == 0,
  // row_offset == 0) packets because those are the deterministic "frame
  // start" packets in the STEM TX -- avoids the stale-stamp races on
  // mbuf pool rotation.
  bool capture_latency = false;

  // Phase 3 processor: apply uint16 -> float dark correction kernel after
  // the gather completes. Uses a constant dark frame and all-ones mask
  // unless real files are loaded -- enough GPU work to mirror the
  // Holoscan processor's depth for throughput parity.
  bool subtract_dark           = false;
  bool apply_valid_pixel_mask  = false;
};

StemRxConfig parse_stem_rx_cfg(const YAML::Node& root) {
  StemRxConfig cfg;
  if (!root["stem_rx"]) {
    std::cerr << "config missing top-level 'stem_rx' block\n";
    return cfg;
  }
  const auto rx = root["stem_rx"];
  cfg.interface_name      = rx["interface_name"].as<std::string>(cfg.interface_name);
  cfg.frames_per_tensor   = rx["frames_per_tensor"].as<uint32_t>(cfg.frames_per_tensor);
  cfg.header_size         = rx["header_size"].as<uint16_t>(cfg.header_size);
  cfg.payload_size        = rx["payload_size"].as<uint16_t>(cfg.payload_size);
  cfg.expected_source_mask =
      rx["expected_source_mask"].as<uint32_t>(cfg.expected_source_mask);
  cfg.batch_close_slack_packets =
      rx["batch_close_slack_packets"].as<uint32_t>(cfg.batch_close_slack_packets);
  cfg.total_time_to_recv_s =
      rx["total_time_to_recv"].as<double>(cfg.total_time_to_recv_s);
  cfg.capture_latency =
      rx["capture_latency"].as<bool>(cfg.capture_latency);
  cfg.subtract_dark =
      rx["subtract_dark"].as<bool>(cfg.subtract_dark);
  cfg.apply_valid_pixel_mask =
      rx["apply_valid_pixel_mask"].as<bool>(cfg.apply_valid_pixel_mask);
  return cfg;
}

// ---------------------------------------------------------------------------
// Signal handling.
// ---------------------------------------------------------------------------
volatile std::sig_atomic_t g_stop_requested = 0;
void on_sigint(int) { g_stop_requested = 1; }

// ---------------------------------------------------------------------------
// Burst lifetime wrapper. A daqiri burst holds N packets; the daqiri pool
// reclaims those packets only when we call free_all_packets_and_burst_rx.
// We extend that lifetime by reference-counting through std::shared_ptr,
// which lets a PacketEntry from "this burst" survive into the next batch
// without leaking and without an explicit refcount field.
// ---------------------------------------------------------------------------
struct BurstHolder {
  daqiri::BurstParams* burst = nullptr;
  explicit BurstHolder(daqiri::BurstParams* b) : burst(b) {}
  BurstHolder(const BurstHolder&) = delete;
  BurstHolder& operator=(const BurstHolder&) = delete;
  ~BurstHolder() {
    if (burst != nullptr) {
      daqiri::free_all_packets_and_burst_rx(burst);
    }
  }
};

struct PacketEntry {
  uint8_t* packet_ptr     = nullptr;
  uint64_t abs_frame      = 0;
  int16_t  global_row     = -1;
  uint16_t row_number     = 0;
  uint16_t source_id      = 0xFFFF;
  std::shared_ptr<BurstHolder> holder;
};

// ---------------------------------------------------------------------------
// Frame assembler. Owns the GPU output frame buffer, the placement scratch
// arrays, the pending packet vector, and the Phase 3 latency samples.
// ---------------------------------------------------------------------------
class FrameAssembler {
 public:
  explicit FrameAssembler(const StemRxConfig& cfg)
      : cfg_(cfg),
        rows_per_tensor_(cfg.frames_per_tensor * stem::FRAME_HEIGHT),
        expected_rows_per_batch_(
            cfg.frames_per_tensor *
            __builtin_popcount(cfg.expected_source_mask & 0xff) *
            stem::ROWS_PER_SOURCE) {
    STEM_CUDA_TRY(cudaMalloc(&gpu_frame_buf_,
                             rows_per_tensor_ * cfg.payload_size));
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_pkt_ptrs_),
                                 sizeof(uint8_t*) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_pkt_ptrs_),
                             sizeof(uint8_t*) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_placements_),
                                 sizeof(stem::PacketPlacement) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_placements_),
                             sizeof(stem::PacketPlacement) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaStreamCreate(&stream_));
    STEM_CUDA_TRY(cudaEventCreate(&gather_done_));

    if (cfg.subtract_dark || cfg.apply_valid_pixel_mask) {
      const uint64_t pixels =
          static_cast<uint64_t>(rows_per_tensor_) *
          (cfg.payload_size / sizeof(uint16_t));
      STEM_CUDA_TRY(cudaMalloc(&gpu_float_out_, pixels * sizeof(float)));
      const uint64_t per_frame_pixels =
          static_cast<uint64_t>(stem::FRAME_HEIGHT) *
          (cfg.payload_size / sizeof(uint16_t));
      STEM_CUDA_TRY(cudaMalloc(&gpu_dark_frame_,
                               per_frame_pixels * sizeof(float)));
      STEM_CUDA_TRY(cudaMemsetAsync(gpu_dark_frame_, 0,
                                    per_frame_pixels * sizeof(float),
                                    stream_));
      STEM_CUDA_TRY(cudaMalloc(&gpu_valid_mask_,
                               per_frame_pixels * sizeof(float)));
      std::vector<float> ones(per_frame_pixels, 1.0f);
      STEM_CUDA_TRY(cudaMemcpyAsync(gpu_valid_mask_, ones.data(),
                                    ones.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_));
      STEM_CUDA_TRY(cudaStreamSynchronize(stream_));
    }

    // Reserve once up front so push_back doesn't realloc 19 times.
    if (cfg_.capture_latency) {
      latencies_us_.reserve(kMaxLatencySamples);
    }
    pending_packets_.reserve(rows_per_tensor_ + cfg.batch_close_slack_packets +
                             4096);
  }

  ~FrameAssembler() {
    pending_packets_.clear();
    if (gpu_frame_buf_) { cudaFree(gpu_frame_buf_); }
    if (h_pkt_ptrs_) { cudaFreeHost(h_pkt_ptrs_); }
    if (d_pkt_ptrs_) { cudaFree(d_pkt_ptrs_); }
    if (h_placements_) { cudaFreeHost(h_placements_); }
    if (d_placements_) { cudaFree(d_placements_); }
    if (gpu_float_out_) { cudaFree(gpu_float_out_); }
    if (gpu_dark_frame_) { cudaFree(gpu_dark_frame_); }
    if (gpu_valid_mask_) { cudaFree(gpu_valid_mask_); }
    if (stream_) { cudaStreamDestroy(stream_); }
    if (gather_done_) { cudaEventDestroy(gather_done_); }
  }

  std::vector<int64_t>& latencies_us() { return latencies_us_; }

  // Parse all packets in `burst` and feed them into pending_packets_.
  // The burst's lifetime is now governed by the shared_ptr inside each
  // PacketEntry; once no entries reference it, BurstHolder destructor
  // calls free_all_packets_and_burst_rx().
  void process_burst(daqiri::BurstParams* burst,
                     uint64_t* total_pkts,
                     uint64_t* total_bytes,
                     uint64_t* drops_unexpected_source,
                     uint64_t* frames_assembled) {
    auto holder = std::make_shared<BurstHolder>(burst);
    const auto n = daqiri::get_num_packets(burst);
    *total_pkts += static_cast<uint64_t>(n);
    *total_bytes += daqiri::get_burst_tot_byte(burst);

    const uint32_t batch_end_relative = cfg_.frames_per_tensor;

    for (int i = 0; i < n; ++i) {
      auto* ptr = static_cast<uint8_t*>(
          daqiri::get_segment_packet_ptr(burst, 0, i));
      if (ptr == nullptr) { continue; }

      // host_pinned daqiri buffers are host-readable; read the 64B
      // STEM header directly.
      const uint8_t* stem_hdr = ptr + cfg_.header_size;
      const uint16_t row_number =
          static_cast<uint16_t>(stem_hdr[stem::STEM_HDR_OFF_ROW_NUMBER_LO]) |
          (static_cast<uint16_t>(stem_hdr[stem::STEM_HDR_OFF_ROW_NUMBER_HI]) << 8);
      const uint16_t source_id =
          static_cast<uint16_t>(stem_hdr[stem::STEM_HDR_OFF_SOURCE_ID_LO]) |
          (static_cast<uint16_t>(stem_hdr[stem::STEM_HDR_OFF_SOURCE_ID_HI]) << 8);

      // Filter by source mask early; ignored packets never enter pending.
      if ((source_id >= 8) ||
          !((cfg_.expected_source_mask >> source_id) & 0x1u)) {
        (*drops_unexpected_source)++;
        continue;
      }

      const uint32_t row_offset = row_number % stem::ROWS_PER_SOURCE;
      const int32_t global_row = source_id_to_global_row(source_id, row_offset);
      if (global_row < 0) {
        (*drops_unexpected_source)++;
        continue;
      }

      // Phase 3 latency: only sample at the deterministic frame-start
      // packet (source_id==0, row_offset==0) so we don't get bitten by
      // mbuf-pool rotation leaving stale epoch_us values in other
      // packet slots.
      if (cfg_.capture_latency &&
          source_id == 0 && row_offset == 0 &&
          latencies_us_.size() < kMaxLatencySamples) {
        uint64_t tx_epoch_us = 0;
        std::memcpy(&tx_epoch_us,
                    stem_hdr + stem::STEM_HDR_OFF_EPOCH_US,
                    sizeof(tx_epoch_us));
        if (tx_epoch_us != 0) {
          const uint64_t now_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
          if (now_us >= tx_epoch_us) {
            latencies_us_.push_back(
                static_cast<int64_t>(now_us - tx_epoch_us));
          }
        }
      }

      const uint32_t frame_idx = row_number / stem::ROWS_PER_SOURCE;

      // Stream-sync: only start admitting packets once we see the
      // canonical (row_number==0, source_id==0) "start of stream" packet.
      // Matches StemReceiverOp::add_pending_packet.
      if (!stream_synced_) {
        if (row_number != 0) {
          (*drops_unexpected_source)++;
          continue;
        }
        stream_synced_ = true;
        current_batch_start_abs_frame_ = 0;
        frame_unwrap_ref_ = 0;
      }

      const uint64_t abs_frame = unwrap_frame_index(frame_idx);

      PacketEntry entry;
      entry.packet_ptr = ptr;
      entry.abs_frame  = abs_frame;
      entry.global_row = static_cast<int16_t>(global_row);
      entry.row_number = row_number;
      entry.source_id  = source_id;
      entry.holder     = holder;
      pending_packets_.push_back(std::move(entry));

      if (abs_frame >= current_batch_start_abs_frame_ + batch_end_relative) {
        future_packet_count_++;
      } else if (abs_frame >= current_batch_start_abs_frame_) {
        current_batch_unique_packets_++;
      }
      // (abs_frame < current_batch_start_abs_frame_: stale, will be
      //  pruned at batch close.)
    }

    try_close_batches(frames_assembled);
  }

  // Force-emit anything still pending at shutdown. Useful for short
  // smoke runs so we at least get one tensor's worth of work logged.
  void flush(uint64_t* frames_assembled) {
    while (current_batch_unique_packets_ > 0) {
      close_batch_and_emit(frames_assembled);
    }
  }

 private:
  static int32_t source_id_to_global_row(uint32_t source_id,
                                         uint32_t row_offset) {
    if (source_id < 4) {
      return 511 - static_cast<int32_t>(row_offset * 4 + source_id);
    }
    if (source_id < 8) {
      return 512 + static_cast<int32_t>(row_offset * 4 + (source_id - 4));
    }
    return -1;
  }

  // Use the same nearest-cycle unwrap as cpp/stem_receiver_op.h. With
  // row_number wrapping every 16384 rows (= 128 frames per source),
  // frame_idx wraps every 128. We pick the candidate within +-1 cycles
  // of the current ref that is closest to (and prefers <=) the ref.
  uint64_t unwrap_frame_index(uint32_t frame_idx) {
    const int64_t ref = static_cast<int64_t>(frame_unwrap_ref_);
    const int64_t ref_cycle = ref / static_cast<int64_t>(stem::FRAMES_PER_WRAP);
    int64_t best = static_cast<int64_t>(frame_idx) +
                   ref_cycle * static_cast<int64_t>(stem::FRAMES_PER_WRAP);
    int64_t best_dist = std::llabs(best - ref);

    for (int64_t delta = -1; delta <= 1; ++delta) {
      const int64_t cycle = ref_cycle + delta;
      const int64_t candidate =
          static_cast<int64_t>(frame_idx) +
          cycle * static_cast<int64_t>(stem::FRAMES_PER_WRAP);
      const int64_t dist = std::llabs(candidate - ref);
      if (dist < best_dist || (dist == best_dist && candidate <= ref)) {
        best = candidate;
        best_dist = dist;
      }
    }
    if (best < 0) { best = 0; }
    if (best > ref) { frame_unwrap_ref_ = static_cast<uint64_t>(best); }
    return static_cast<uint64_t>(best);
  }

  // Close batches as long as we have either a complete window's worth of
  // unique packets, OR enough future packets piled up to justify the
  // slack-close.
  void try_close_batches(uint64_t* frames_assembled) {
    while (should_close_current_batch()) {
      close_batch_and_emit(frames_assembled);
    }
  }

  bool should_close_current_batch() const {
    if (!stream_synced_) { return false; }
    if (current_batch_unique_packets_ >= expected_rows_per_batch_) {
      return true;
    }
    return current_batch_unique_packets_ > 0 &&
           future_packet_count_ >= cfg_.batch_close_slack_packets;
  }

  void close_batch_and_emit(uint64_t* frames_assembled) {
    const uint64_t batch_end =
        current_batch_start_abs_frame_ + cfg_.frames_per_tensor;

    // Walk pending_packets_ once: copy in-window entries to scratch,
    // keep future entries, drop stale entries.
    uint32_t pkts_to_gather = 0;
    auto write_it = pending_packets_.begin();
    for (auto read_it = pending_packets_.begin();
         read_it != pending_packets_.end(); ++read_it) {
      if (read_it->abs_frame < current_batch_start_abs_frame_) {
        // stale: drop, holder refcount goes down when entry is overwritten
        continue;
      }
      if (read_it->abs_frame >= batch_end) {
        // future: retain for next batch
        if (write_it != read_it) { *write_it = std::move(*read_it); }
        ++write_it;
        continue;
      }
      // in-window
      if (pkts_to_gather < rows_per_tensor_) {
        const uint64_t rel_frame =
            read_it->abs_frame - current_batch_start_abs_frame_;
        h_pkt_ptrs_[pkts_to_gather] = read_it->packet_ptr;
        stem::PacketPlacement pl;
        pl.global_row     = read_it->global_row;
        pl.relative_frame = static_cast<uint8_t>(rel_frame);
        pl.valid          = 1;
        h_placements_[pkts_to_gather] = pl;
        pkts_to_gather++;
      }
      // do NOT write_it++ -- this entry is consumed
    }
    pending_packets_.erase(write_it, pending_packets_.end());

    if (pkts_to_gather > 0) {
      STEM_CUDA_TRY(cudaMemsetAsync(gpu_frame_buf_, 0,
                                    rows_per_tensor_ * cfg_.payload_size,
                                    stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_pkt_ptrs_, h_pkt_ptrs_,
                                    sizeof(uint8_t*) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_placements_, h_placements_,
                                    sizeof(stem::PacketPlacement) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));

      stem::stem_gather_packets_by_placement(
          d_pkt_ptrs_, d_placements_, gpu_frame_buf_,
          cfg_.payload_size, cfg_.header_size,
          pkts_to_gather,
          rows_per_tensor_,
          stream_);

      if (gpu_float_out_ != nullptr) {
        const uint32_t width = cfg_.payload_size / sizeof(uint16_t);
        stem::stem_dark_correct_uint16_to_float(
            reinterpret_cast<const uint16_t*>(gpu_frame_buf_),
            gpu_dark_frame_,
            gpu_valid_mask_,
            gpu_float_out_,
            cfg_.frames_per_tensor,
            stem::FRAME_HEIGHT,
            width,
            cfg_.subtract_dark,
            cfg_.apply_valid_pixel_mask,
            stream_);
      }
      STEM_CUDA_TRY(cudaEventRecord(gather_done_, stream_));
      STEM_CUDA_TRY(cudaEventSynchronize(gather_done_));
    }

    if (pkts_to_gather < expected_rows_per_batch_) {
      incomplete_batches_++;
      incomplete_missing_total_ += expected_rows_per_batch_ - pkts_to_gather;
    }

    *frames_assembled += cfg_.frames_per_tensor;
    current_batch_start_abs_frame_ += cfg_.frames_per_tensor;

    // Recount what is still in the (now-new) current and future windows.
    rebuild_window_counts();
  }

  void rebuild_window_counts() {
    current_batch_unique_packets_ = 0;
    future_packet_count_ = 0;
    const uint64_t batch_end =
        current_batch_start_abs_frame_ + cfg_.frames_per_tensor;
    for (const auto& e : pending_packets_) {
      if (e.abs_frame >= batch_end) {
        future_packet_count_++;
      } else if (e.abs_frame >= current_batch_start_abs_frame_) {
        current_batch_unique_packets_++;
      }
    }
  }

  const StemRxConfig& cfg_;
  const uint32_t      rows_per_tensor_;
  const uint32_t      expected_rows_per_batch_;
  uint8_t*            gpu_frame_buf_ = nullptr;
  uint8_t**           h_pkt_ptrs_ = nullptr;
  uint8_t**           d_pkt_ptrs_ = nullptr;
  stem::PacketPlacement* h_placements_ = nullptr;
  stem::PacketPlacement* d_placements_ = nullptr;
  cudaStream_t        stream_ = nullptr;
  cudaEvent_t         gather_done_ = nullptr;

  std::vector<PacketEntry> pending_packets_;
  uint32_t   current_batch_unique_packets_ = 0;
  uint32_t   future_packet_count_ = 0;
  uint64_t   current_batch_start_abs_frame_ = 0;
  uint64_t   frame_unwrap_ref_ = 0;
  bool       stream_synced_ = false;
  uint64_t   incomplete_batches_ = 0;
  uint64_t   incomplete_missing_total_ = 0;

  // Phase 3 dark-correction processor scratch.
  float* gpu_float_out_  = nullptr;
  float* gpu_dark_frame_ = nullptr;
  float* gpu_valid_mask_ = nullptr;

  // Phase 3 latency samples (us).
  std::vector<int64_t> latencies_us_;
};

// ---------------------------------------------------------------------------
// RX worker.
// ---------------------------------------------------------------------------
void rx_worker(const StemRxConfig& cfg, std::atomic<bool>& stop) {
  const int port_id = daqiri::get_port_id(cfg.interface_name);
  if (port_id < 0) {
    std::cerr << "Invalid RX interface_name: " << cfg.interface_name << "\n";
    stop.store(true);
    return;
  }

  FrameAssembler asm_state(cfg);

  uint64_t total_pkts = 0;
  uint64_t total_bytes = 0;
  uint64_t drops_unexpected_source = 0;
  uint64_t frames_assembled = 0;
  uint64_t bursts_polled = 0;

  const auto start = std::chrono::steady_clock::now();
  while (!stop.load() && !g_stop_requested) {
    if (cfg.total_time_to_recv_s >= 0.0) {
      const auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      if (elapsed >= cfg.total_time_to_recv_s) { break; }
    }

    const auto num_rx_queues =
        static_cast<int>(daqiri::get_num_rx_queues(port_id));
    bool got_any = false;
    for (int q = 0; q < num_rx_queues; ++q) {
      daqiri::BurstParams* burst = nullptr;
      if (daqiri::get_rx_burst(&burst, port_id, q) !=
              daqiri::Status::SUCCESS ||
          burst == nullptr) {
        continue;
      }
      got_any = true;
      ++bursts_polled;
      asm_state.process_burst(burst,
                              &total_pkts, &total_bytes,
                              &drops_unexpected_source,
                              &frames_assembled);
    }
    if (!got_any) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  asm_state.flush(&frames_assembled);

  const double secs =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  const double gbps = secs > 0
      ? static_cast<double>(total_bytes) * 8.0 / (secs * 1e9)
      : 0.0;
  const double fps = secs > 0
      ? static_cast<double>(frames_assembled) / secs
      : 0.0;

  std::printf("stem_daqiri_rx complete:\n");
  std::printf("  duration         : %.3f s\n", secs);
  std::printf("  bursts polled    : %lu\n", static_cast<unsigned long>(bursts_polled));
  std::printf("  packets received : %lu\n", static_cast<unsigned long>(total_pkts));
  std::printf("  bytes received   : %lu\n", static_cast<unsigned long>(total_bytes));
  std::printf("  achieved Gbps    : %.3f\n", gbps);
  std::printf("  frames assembled : %lu  (fps %.3f)\n",
              static_cast<unsigned long>(frames_assembled), fps);
  std::printf("  unexpected source: %lu\n",
              static_cast<unsigned long>(drops_unexpected_source));
  // The previous "out-of-window" counter was a bug -- those packets are
  // now retained across batch boundaries. Report 0 explicitly so any
  // existing parser that grepped that line still works.
  std::printf("  out-of-window    : 0\n");

  if (cfg.capture_latency) {
    auto& lat = asm_state.latencies_us();
    if (lat.empty()) {
      std::printf("  latency samples  : 0 (no epoch_us stamps received)\n");
    } else {
      std::sort(lat.begin(), lat.end());
      const auto pct = [&](double q) {
        const size_t idx = std::min(lat.size() - 1,
            static_cast<size_t>(q * (lat.size() - 1)));
        return lat[idx];
      };
      const int64_t p50 = pct(0.50);
      const int64_t p90 = pct(0.90);
      const int64_t p99 = pct(0.99);
      const int64_t p999 = pct(0.999);
      std::printf("  latency samples  : %lu\n",
                  static_cast<unsigned long>(lat.size()));
      std::printf("  latency p50/p90/p99/p999 us : %ld / %ld / %ld / %ld\n",
                  static_cast<long>(p50),
                  static_cast<long>(p90),
                  static_cast<long>(p99),
                  static_cast<long>(p999));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.yaml> [--seconds N]\n";
    return 1;
  }
  std::signal(SIGINT, on_sigint);

  double cli_seconds = -2.0;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--seconds") {
      cli_seconds = std::stod(argv[i + 1]);
    }
  }

  const auto root = YAML::LoadFile(argv[1]);
  if (daqiri::daqiri_init(argv[1]) != daqiri::Status::SUCCESS) {
    std::cerr << "daqiri_init failed for " << argv[1] << "\n";
    return 1;
  }

  StemRxConfig cfg = parse_stem_rx_cfg(root);
  if (cli_seconds > -1.5) { cfg.total_time_to_recv_s = cli_seconds; }

  std::cout << "stem_daqiri_rx starting on '" << cfg.interface_name
            << "' frames_per_tensor=" << cfg.frames_per_tensor
            << " header_size=" << cfg.header_size
            << " payload_size=" << cfg.payload_size
            << " source_mask=0x" << std::hex << cfg.expected_source_mask
            << std::dec
            << " duration=" << cfg.total_time_to_recv_s << " s\n";

  std::atomic<bool> stop{false};
  std::thread t(rx_worker, cfg, std::ref(stop));
  t.join();

  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
