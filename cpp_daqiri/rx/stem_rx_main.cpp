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
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef STEM_DAQIRI_HAVE_HDF5
#include <H5Cpp.h>
#endif

#include <daqiri/daqiri.h>

#include "stem_kernels.h"
#include "stem_packet.h"

namespace {

constexpr size_t kMaxLatencySamples = 1'000'000;  // cap so RX doesn't blow RAM

struct WriterConfig {
  std::string filepath = "/tmp/stem_daqiri_rx.h5";
  std::string dataset_name = "/processed";
  bool noop = true;
  uint32_t num_concurrent = 3;
};

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

  // Spark/unified-memory keeps CPU header reads. IGX/dGPU device memory uses
  // a GPU header extraction kernel plus header-only DtoH metadata copy.
  bool gpu_header_extract = false;

  // Test-only correctness gate: copy the assembled uint16 tensor back after
  // gather and verify it matches stem_daqiri_tx's deterministic row ramp.
  bool validate_tx_ramp = false;

  // Phase 3 processor: apply uint16 -> float dark correction kernel after
  // the gather completes. Uses a constant dark frame and all-ones mask
  // unless real files are loaded -- enough GPU work to mirror the
  // Holoscan processor's depth for throughput parity.
  bool subtract_dark           = false;
  bool apply_valid_pixel_mask  = false;

  WriterConfig writer;
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
  cfg.gpu_header_extract =
      rx["gpu_header_extract"].as<bool>(cfg.gpu_header_extract);
  cfg.validate_tx_ramp =
      rx["validate_tx_ramp"].as<bool>(cfg.validate_tx_ramp);
  cfg.subtract_dark =
      rx["subtract_dark"].as<bool>(cfg.subtract_dark);
  cfg.apply_valid_pixel_mask =
      rx["apply_valid_pixel_mask"].as<bool>(cfg.apply_valid_pixel_mask);
  if (root["writer"]) {
    const auto wr = root["writer"];
    cfg.writer.filepath =
        wr["filepath"].as<std::string>(cfg.writer.filepath);
    cfg.writer.dataset_name =
        wr["dataset_name"].as<std::string>(cfg.writer.dataset_name);
    cfg.writer.noop = wr["noop"].as<bool>(cfg.writer.noop);
    cfg.writer.num_concurrent =
        wr["num_concurrent"].as<uint32_t>(cfg.writer.num_concurrent);
    if (cfg.writer.num_concurrent == 0) {
      cfg.writer.num_concurrent = 1;
    }
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// Signal handling.
// ---------------------------------------------------------------------------
volatile std::sig_atomic_t g_stop_requested = 0;
void on_sigint(int) { g_stop_requested = 1; }

struct RxRunStatus {
  std::atomic<bool> validation_failed{false};
};

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

struct OutputSlot {
  uint8_t* gpu_u16 = nullptr;
  float* gpu_float = nullptr;
  cudaEvent_t ready = nullptr;
  std::atomic<bool> leased{false};
  uint64_t batch_index = 0;
  uint32_t frames = 0;
};

class AsyncFrameSink {
 public:
  AsyncFrameSink(const WriterConfig& cfg,
                 uint32_t frames_per_tensor,
                 uint32_t height,
                 uint32_t width,
                 bool write_float)
      : cfg_(cfg),
        frames_per_tensor_(frames_per_tensor),
        height_(height),
        width_(width),
        write_float_(write_float) {
    if (cfg_.noop) { return; }
#ifndef STEM_DAQIRI_HAVE_HDF5
    std::fprintf(stderr,
                 "writer.noop=false requested, but stem_daqiri_rx was built "
                 "without HDF5 support; sink output will be dropped\n");
    errors_++;
    return;
#else
    try {
      file_ = std::make_unique<H5::H5File>(cfg_.filepath, H5F_ACC_TRUNC);
    } catch (const H5::Exception& e) {
      std::fprintf(stderr, "HDF5 open failed for %s: %s\n",
                   cfg_.filepath.c_str(), e.getCDetailMsg());
      errors_++;
      return;
    }
#endif
    active_ = true;
    worker_ = std::thread(&AsyncFrameSink::run, this);
  }

  ~AsyncFrameSink() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
  }

  AsyncFrameSink(const AsyncFrameSink&) = delete;
  AsyncFrameSink& operator=(const AsyncFrameSink&) = delete;

  bool enabled() const { return !cfg_.noop && active_; }

  void enqueue(OutputSlot* slot) {
    if (cfg_.noop || !active_) {
      slot->leased.store(false, std::memory_order_release);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push(slot);
      queued_++;
    }
    cv_.notify_one();
  }

  uint64_t queued() const { return queued_.load(); }
  uint64_t written() const { return written_.load(); }
  uint64_t errors() const { return errors_.load(); }

 private:
  void run() {
    for (;;) {
      OutputSlot* slot = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) { return; }
        slot = queue_.front();
        queue_.pop();
      }

      STEM_CUDA_TRY(cudaEventSynchronize(slot->ready));
      write_slot(slot);
      slot->leased.store(false, std::memory_order_release);
    }
  }

  void write_slot(OutputSlot* slot) {
    const size_t elem_size = write_float_ ? sizeof(float) : sizeof(uint16_t);
    const size_t elems =
        static_cast<size_t>(frames_per_tensor_) * height_ * width_;
    const size_t bytes = elems * elem_size;
    if (host_buffer_.size() < bytes) { host_buffer_.resize(bytes); }

    const void* src = write_float_
        ? static_cast<const void*>(slot->gpu_float)
        : static_cast<const void*>(slot->gpu_u16);
    STEM_CUDA_TRY(cudaMemcpy(host_buffer_.data(), src, bytes,
                             cudaMemcpyDeviceToHost));

#ifdef STEM_DAQIRI_HAVE_HDF5
    if (!file_) {
      errors_++;
      return;
    }
    try {
      const H5::DataType h5_type =
          write_float_ ? H5::PredType::NATIVE_FLOAT
                       : H5::PredType::NATIVE_UINT16;
      if (!dataset_) {
        hsize_t dims[3] = {frames_per_tensor_, height_, width_};
        hsize_t max_dims[3] = {H5S_UNLIMITED, height_, width_};
        H5::DataSpace filespace(3, dims, max_dims);
        H5::DSetCreatPropList prop;
        hsize_t chunk_dims[3] = {1, height_, width_};
        prop.setChunk(3, chunk_dims);
        dataset_ = std::make_unique<H5::DataSet>(
            file_->createDataSet(cfg_.dataset_name, h5_type, filespace, prop));
        current_frames_ = frames_per_tensor_;
      } else {
        current_frames_ += frames_per_tensor_;
        hsize_t dims[3] = {current_frames_, height_, width_};
        dataset_->extend(dims);
      }

      H5::DataSpace filespace(dataset_->getSpace());
      hsize_t offset[3] = {current_frames_ - frames_per_tensor_, 0, 0};
      hsize_t count[3] = {frames_per_tensor_, height_, width_};
      filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
      H5::DataSpace memspace(3, count);
      dataset_->write(host_buffer_.data(), h5_type, memspace, filespace);
      written_++;
    } catch (const H5::Exception& e) {
      std::fprintf(stderr, "HDF5 write failed: %s\n", e.getCDetailMsg());
      errors_++;
    }
#else
    (void)slot;
    errors_++;
#endif
  }

  WriterConfig cfg_;
  uint32_t frames_per_tensor_ = 0;
  uint32_t height_ = 0;
  uint32_t width_ = 0;
  bool write_float_ = false;

  std::thread worker_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::queue<OutputSlot*> queue_;
  bool stopping_ = false;
  bool active_ = false;
  std::atomic<uint64_t> queued_{0};
  std::atomic<uint64_t> written_{0};
  std::atomic<uint64_t> errors_{0};
  std::vector<uint8_t> host_buffer_;

#ifdef STEM_DAQIRI_HAVE_HDF5
  std::unique_ptr<H5::H5File> file_;
  std::unique_ptr<H5::DataSet> dataset_;
  hsize_t current_frames_ = 0;
#endif
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
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_pkt_ptrs_),
                                 sizeof(uint8_t*) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_pkt_ptrs_),
                             sizeof(uint8_t*) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_placements_),
                                 sizeof(stem::PacketPlacement) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_placements_),
                             sizeof(stem::PacketPlacement) * rows_per_tensor_));
    STEM_CUDA_TRY(cudaStreamCreate(&stream_));

    if (cfg.subtract_dark || cfg.apply_valid_pixel_mask) {
      const uint64_t pixels =
          static_cast<uint64_t>(rows_per_tensor_) *
          (cfg.payload_size / sizeof(uint16_t));
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

    const uint32_t slot_count =
        cfg.writer.noop ? 1 : std::max<uint32_t>(1, cfg.writer.num_concurrent);
    output_slots_.reserve(slot_count);
    for (uint32_t i = 0; i < slot_count; ++i) {
      auto slot = std::make_unique<OutputSlot>();
      STEM_CUDA_TRY(cudaMalloc(&slot->gpu_u16,
                               rows_per_tensor_ * cfg.payload_size));
      if (cfg.subtract_dark || cfg.apply_valid_pixel_mask) {
        const uint64_t pixels =
            static_cast<uint64_t>(rows_per_tensor_) *
            (cfg.payload_size / sizeof(uint16_t));
        STEM_CUDA_TRY(cudaMalloc(&slot->gpu_float, pixels * sizeof(float)));
      }
      STEM_CUDA_TRY(cudaEventCreateWithFlags(&slot->ready,
                                             cudaEventDisableTiming));
      output_slots_.push_back(std::move(slot));
    }
    sink_ = std::make_unique<AsyncFrameSink>(
        cfg.writer,
        cfg.frames_per_tensor,
        stem::FRAME_HEIGHT,
        cfg.payload_size / sizeof(uint16_t),
        (cfg.subtract_dark || cfg.apply_valid_pixel_mask));

    // Reserve once up front so push_back doesn't realloc 19 times.
    if (cfg_.capture_latency) {
      latencies_us_.reserve(kMaxLatencySamples);
    }
    pending_packets_.reserve(rows_per_tensor_ + cfg.batch_close_slack_packets +
                             4096);
    current_batch_occupied_.assign(rows_per_tensor_, 0);
    emit_cell_generation_.assign(rows_per_tensor_, 0);
  }

  ~FrameAssembler() {
    sink_.reset();
    pending_packets_.clear();
    for (auto& slot : output_slots_) {
      if (slot->gpu_u16) { cudaFree(slot->gpu_u16); }
      if (slot->gpu_float) { cudaFree(slot->gpu_float); }
      if (slot->ready) { cudaEventDestroy(slot->ready); }
    }
    if (h_pkt_ptrs_) { cudaFreeHost(h_pkt_ptrs_); }
    if (d_pkt_ptrs_) { cudaFree(d_pkt_ptrs_); }
    if (h_placements_) { cudaFreeHost(h_placements_); }
    if (d_placements_) { cudaFree(d_placements_); }
    if (h_burst_ptrs_) { cudaFreeHost(h_burst_ptrs_); }
    if (d_burst_ptrs_) { cudaFree(d_burst_ptrs_); }
    if (h_burst_headers_) { cudaFreeHost(h_burst_headers_); }
    if (d_burst_headers_) { cudaFree(d_burst_headers_); }
    if (gpu_dark_frame_) { cudaFree(gpu_dark_frame_); }
    if (gpu_valid_mask_) { cudaFree(gpu_valid_mask_); }
    if (stream_) { cudaStreamDestroy(stream_); }
  }

  std::vector<int64_t>& latencies_us() { return latencies_us_; }
  uint64_t validation_batches() const { return validation_batches_; }
  uint64_t validation_partial_batches() const { return validation_partial_batches_; }
  uint64_t validation_rows_checked() const { return validation_rows_checked_; }
  uint64_t validation_mismatches() const { return validation_mismatches_; }
  uint64_t incomplete_batches() const { return incomplete_batches_; }
  uint64_t incomplete_missing_total() const { return incomplete_missing_total_; }
  uint64_t incomplete_missing_max() const { return incomplete_missing_max_; }
  uint64_t output_pool_drops() const { return output_pool_drops_; }
  uint64_t sink_queued() const { return sink_ ? sink_->queued() : 0; }
  uint64_t sink_written() const { return sink_ ? sink_->written() : 0; }
  uint64_t sink_errors() const { return sink_ ? sink_->errors() : 0; }

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

    if (cfg_.gpu_header_extract) {
      ensure_header_scratch(static_cast<uint32_t>(n));
      burst_packet_ptrs_.resize(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        auto* ptr = static_cast<uint8_t*>(
            daqiri::get_segment_packet_ptr(burst, 0, i));
        burst_packet_ptrs_[static_cast<size_t>(i)] = ptr;
        h_burst_ptrs_[i] = (ptr == nullptr) ? nullptr : ptr + cfg_.header_size;
      }

      STEM_CUDA_TRY(cudaMemcpyAsync(d_burst_ptrs_, h_burst_ptrs_,
                                    sizeof(uint8_t*) * n,
                                    cudaMemcpyHostToDevice, stream_));
      stem::stem_extract_packet_headers(
          d_burst_ptrs_, d_burst_headers_, static_cast<uint32_t>(n), stream_);
      STEM_CUDA_TRY(cudaMemcpyAsync(h_burst_headers_, d_burst_headers_,
                                    sizeof(stem::PacketHeaderInfo) * n,
                                    cudaMemcpyDeviceToHost, stream_));
      STEM_CUDA_TRY(cudaStreamSynchronize(stream_));

      for (int i = 0; i < n; ++i) {
        auto* ptr = burst_packet_ptrs_[static_cast<size_t>(i)];
        if (ptr == nullptr) { continue; }
        admit_packet(ptr, h_burst_headers_[i], holder,
                     drops_unexpected_source);
      }
    } else {
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
        const uint32_t row_offset = row_number % stem::ROWS_PER_SOURCE;
        const uint32_t frame_idx = row_number / stem::ROWS_PER_SOURCE;
        const int32_t global_row = source_id_to_global_row(source_id, row_offset);
        uint64_t tx_epoch_us = 0;
        if (cfg_.capture_latency) {
          std::memcpy(&tx_epoch_us,
                      stem_hdr + stem::STEM_HDR_OFF_EPOCH_US,
                      sizeof(tx_epoch_us));
        }
        stem::PacketHeaderInfo header{};
        header.row_number = row_number;
        header.source_id = source_id;
        header.frame_index = static_cast<uint16_t>(frame_idx);
        header.row_offset = static_cast<uint16_t>(row_offset);
        header.global_row = static_cast<int16_t>(global_row);
        header.epoch_us = tx_epoch_us;

        admit_packet(ptr, header, holder, drops_unexpected_source);
      }
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

  static bool global_row_to_source(uint32_t global_row,
                                   uint32_t* source_id,
                                   uint32_t* row_offset) {
    if (global_row < 512) {
      const uint32_t packed = 511 - global_row;
      *source_id = packed % 4;
      *row_offset = packed / 4;
      return true;
    }
    if (global_row < stem::FRAME_HEIGHT) {
      const uint32_t packed = global_row - 512;
      *source_id = 4 + (packed % 4);
      *row_offset = packed / 4;
      return true;
    }
    return false;
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

  void ensure_header_scratch(uint32_t n) {
    if (n <= burst_header_capacity_) { return; }
    uint32_t new_capacity = burst_header_capacity_ == 0 ? 1024 : burst_header_capacity_;
    while (new_capacity < n) { new_capacity *= 2; }

    if (h_burst_ptrs_) { STEM_CUDA_TRY(cudaFreeHost(h_burst_ptrs_)); }
    if (d_burst_ptrs_) { STEM_CUDA_TRY(cudaFree(d_burst_ptrs_)); }
    if (h_burst_headers_) { STEM_CUDA_TRY(cudaFreeHost(h_burst_headers_)); }
    if (d_burst_headers_) { STEM_CUDA_TRY(cudaFree(d_burst_headers_)); }

    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_burst_ptrs_),
                                 sizeof(uint8_t*) * new_capacity));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_burst_ptrs_),
                             sizeof(uint8_t*) * new_capacity));
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_burst_headers_),
                                 sizeof(stem::PacketHeaderInfo) * new_capacity));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_burst_headers_),
                             sizeof(stem::PacketHeaderInfo) * new_capacity));
    burst_header_capacity_ = new_capacity;
  }

  void sample_latency_if_needed(const stem::PacketHeaderInfo& header) {
    if (!cfg_.capture_latency ||
        header.source_id != 0 ||
        header.row_offset != 0 ||
        latencies_us_.size() >= kMaxLatencySamples ||
        header.epoch_us == 0) {
      return;
    }
    const uint64_t now_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    if (now_us >= header.epoch_us) {
      latencies_us_.push_back(static_cast<int64_t>(now_us - header.epoch_us));
    }
  }

  void admit_packet(uint8_t* packet_ptr,
                    const stem::PacketHeaderInfo& header,
                    const std::shared_ptr<BurstHolder>& holder,
                    uint64_t* drops_unexpected_source) {
    if ((header.source_id >= 8) ||
        !((cfg_.expected_source_mask >> header.source_id) & 0x1u) ||
        header.global_row < 0) {
      (*drops_unexpected_source)++;
      return;
    }

    sample_latency_if_needed(header);

    // Stream-sync: only start admitting packets once we see the
    // canonical (row_number==0, source_id==0) "start of stream" packet.
    // Matches StemReceiverOp::add_pending_packet.
    if (!stream_synced_) {
      if (header.row_number != 0) {
        (*drops_unexpected_source)++;
        return;
      }
      stream_synced_ = true;
      current_batch_start_abs_frame_ = 0;
      frame_unwrap_ref_ = 0;
    }

    const uint64_t abs_frame = unwrap_frame_index(header.frame_index);

    PacketEntry entry;
    entry.packet_ptr = packet_ptr;
    entry.abs_frame  = abs_frame;
    entry.global_row = header.global_row;
    entry.row_number = header.row_number;
    entry.source_id  = header.source_id;
    entry.holder     = holder;
    pending_packets_.push_back(std::move(entry));

    const uint64_t batch_end =
        current_batch_start_abs_frame_ + cfg_.frames_per_tensor;
    if (abs_frame >= batch_end) {
      future_packet_count_++;
      return;
    }
    if (abs_frame < current_batch_start_abs_frame_) { return; }

    const uint64_t relative_frame = abs_frame - current_batch_start_abs_frame_;
    const uint64_t cell =
        relative_frame * stem::FRAME_HEIGHT + static_cast<uint64_t>(header.global_row);
    // relative_frame < frames_per_tensor and global_row < FRAME_HEIGHT are
    // both enforced upstream, so cell < rows_per_tensor_ by construction.
    assert(cell < current_batch_occupied_.size());
    if (!current_batch_occupied_[cell]) {
      current_batch_occupied_[cell] = 1;
      current_batch_unique_packets_++;
    }
  }

  void begin_emit_generation() {
    emit_generation_++;
    if (emit_generation_ == 0) {
      std::fill(emit_cell_generation_.begin(), emit_cell_generation_.end(), 0);
      emit_generation_ = 1;
    }
  }

  OutputSlot* acquire_output_slot() {
    for (auto& slot : output_slots_) {
      bool expected = false;
      if (slot->leased.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        return slot.get();
      }
    }
    output_pool_drops_++;
    return nullptr;
  }

  void release_output_slot(OutputSlot* slot) {
    if (slot != nullptr) {
      slot->leased.store(false, std::memory_order_release);
    }
  }

  void validate_tx_ramp_batch(uint32_t pkts_to_gather, const uint8_t* gpu_frame_buf) {
    if (!cfg_.validate_tx_ramp) { return; }
    if (pkts_to_gather < expected_rows_per_batch_) {
      validation_partial_batches_++;
      return;
    }

    const uint32_t width = cfg_.payload_size / sizeof(uint16_t);
    const size_t sample_count =
        static_cast<size_t>(rows_per_tensor_) * static_cast<size_t>(width);
    if (validation_host_buf_.size() < sample_count) {
      validation_host_buf_.resize(sample_count);
    }

    STEM_CUDA_TRY(cudaMemcpy(validation_host_buf_.data(), gpu_frame_buf,
                             sample_count * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost));
    validation_batches_++;

    for (uint32_t rel_frame = 0; rel_frame < cfg_.frames_per_tensor; ++rel_frame) {
      for (uint32_t global_row = 0; global_row < stem::FRAME_HEIGHT; ++global_row) {
        uint32_t source_id = 0;
        uint32_t row_offset = 0;
        if (!global_row_to_source(global_row, &source_id, &row_offset) ||
            !((cfg_.expected_source_mask >> source_id) & 0x1u)) {
          continue;
        }
        validation_rows_checked_++;
        const size_t row_base =
            (static_cast<size_t>(rel_frame) * stem::FRAME_HEIGHT + global_row) *
            static_cast<size_t>(width);
        for (uint32_t sample = 0; sample < width; ++sample) {
          const uint16_t expected =
              static_cast<uint16_t>((source_id << 12) |
                                    (row_offset & 0xff) |
                                    ((sample & 0xf) << 8));
          const uint16_t actual = validation_host_buf_[row_base + sample];
          if (actual != expected) {
            validation_mismatches_++;
            if (!validation_first_mismatch_reported_) {
              validation_first_mismatch_reported_ = true;
              std::fprintf(stderr,
                           "validate_tx_ramp first mismatch: batch=%lu "
                           "relative_frame=%u global_row=%u source_id=%u "
                           "row_offset=%u sample=%u expected=0x%04x actual=0x%04x\n",
                           static_cast<unsigned long>(validation_batches_ - 1),
                           rel_frame, global_row, source_id, row_offset, sample,
                           expected, actual);
            }
          }
        }
      }
    }
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
    begin_emit_generation();
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
        const uint64_t cell =
            rel_frame * stem::FRAME_HEIGHT + static_cast<uint64_t>(read_it->global_row);
        // cell < rows_per_tensor_ by construction (see admit_packet).
        assert(cell < emit_cell_generation_.size());
        if (emit_cell_generation_[cell] == emit_generation_) {
          // Duplicate (frame, row) within this close; drop the later copy.
          continue;
        }
        emit_cell_generation_[cell] = emit_generation_;
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

    // Only claim a sink slot when we actually have something to gather; this
    // keeps output_pool_drops a true "writer fell behind" signal instead of
    // including empty-window closes.
    OutputSlot* output_slot =
        pkts_to_gather > 0 ? acquire_output_slot() : nullptr;
    if (output_slot != nullptr) {
      STEM_CUDA_TRY(cudaMemsetAsync(output_slot->gpu_u16, 0,
                                    rows_per_tensor_ * cfg_.payload_size,
                                    stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_pkt_ptrs_, h_pkt_ptrs_,
                                    sizeof(uint8_t*) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_placements_, h_placements_,
                                    sizeof(stem::PacketPlacement) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));

      stem::stem_gather_packets_by_placement(
          d_pkt_ptrs_, d_placements_, output_slot->gpu_u16,
          stem::STEM_PAYLOAD_SIZE,
          static_cast<uint16_t>(cfg_.header_size + stem::STEM_HEADER_SIZE),
          pkts_to_gather,
          rows_per_tensor_,
          stream_);

      if (output_slot->gpu_float != nullptr) {
        const uint32_t width = cfg_.payload_size / sizeof(uint16_t);
        stem::stem_dark_correct_uint16_to_float(
            reinterpret_cast<const uint16_t*>(output_slot->gpu_u16),
            gpu_dark_frame_,
            gpu_valid_mask_,
            output_slot->gpu_float,
            cfg_.frames_per_tensor,
            stem::FRAME_HEIGHT,
            width,
            cfg_.subtract_dark,
            cfg_.apply_valid_pixel_mask,
            stream_);
      }
      output_slot->batch_index = emitted_batches_;
      output_slot->frames = cfg_.frames_per_tensor;
      STEM_CUDA_TRY(cudaEventRecord(output_slot->ready, stream_));
      STEM_CUDA_TRY(cudaEventSynchronize(output_slot->ready));
      validate_tx_ramp_batch(pkts_to_gather, output_slot->gpu_u16);
      sink_->enqueue(output_slot);
      // frames_assembled counts only batches that produced a GPU tensor;
      // empty closes (pkts_to_gather==0) and pool-starved closes
      // (output_slot==nullptr) do not bump it, so the reported fps reflects
      // real downstream-visible frames.
      (*frames_assembled) += cfg_.frames_per_tensor;
      emitted_batches_++;
    }

    pending_packets_.erase(write_it, pending_packets_.end());

    if (pkts_to_gather < expected_rows_per_batch_) {
      const uint64_t missing = expected_rows_per_batch_ - pkts_to_gather;
      incomplete_batches_++;
      incomplete_missing_total_ += missing;
      incomplete_missing_max_ = std::max(incomplete_missing_max_, missing);
    }

    current_batch_start_abs_frame_ += cfg_.frames_per_tensor;

    // Recount what is still in the (now-new) current and future windows.
    rebuild_window_counts();
  }

  void rebuild_window_counts() {
    std::fill(current_batch_occupied_.begin(), current_batch_occupied_.end(), 0);
    current_batch_unique_packets_ = 0;
    future_packet_count_ = 0;
    const uint64_t batch_end =
        current_batch_start_abs_frame_ + cfg_.frames_per_tensor;
    for (const auto& e : pending_packets_) {
      if (e.abs_frame >= batch_end) {
        future_packet_count_++;
      } else if (e.abs_frame >= current_batch_start_abs_frame_) {
        const uint64_t relative_frame = e.abs_frame - current_batch_start_abs_frame_;
        const uint64_t cell =
            relative_frame * stem::FRAME_HEIGHT + static_cast<uint64_t>(e.global_row);
        // cell < rows_per_tensor_ by construction (see admit_packet).
        assert(cell < current_batch_occupied_.size());
        if (!current_batch_occupied_[cell]) {
          current_batch_occupied_[cell] = 1;
          current_batch_unique_packets_++;
        }
      }
    }
  }

  const StemRxConfig& cfg_;
  const uint32_t      rows_per_tensor_;
  const uint32_t      expected_rows_per_batch_;
  uint8_t**           h_pkt_ptrs_ = nullptr;
  uint8_t**           d_pkt_ptrs_ = nullptr;
  stem::PacketPlacement* h_placements_ = nullptr;
  stem::PacketPlacement* d_placements_ = nullptr;
  cudaStream_t        stream_ = nullptr;

  std::vector<PacketEntry> pending_packets_;
  std::vector<uint8_t> current_batch_occupied_;
  std::vector<uint32_t> emit_cell_generation_;
  uint32_t emit_generation_ = 1;
  uint32_t   current_batch_unique_packets_ = 0;
  uint32_t   future_packet_count_ = 0;
  uint64_t   current_batch_start_abs_frame_ = 0;
  uint64_t   frame_unwrap_ref_ = 0;
  bool       stream_synced_ = false;
  uint64_t   incomplete_batches_ = 0;
  uint64_t   incomplete_missing_total_ = 0;
  uint64_t   incomplete_missing_max_ = 0;

  // Phase 3 dark-correction processor scratch.
  float* gpu_dark_frame_ = nullptr;
  float* gpu_valid_mask_ = nullptr;

  std::vector<std::unique_ptr<OutputSlot>> output_slots_;
  std::unique_ptr<AsyncFrameSink> sink_;
  uint64_t output_pool_drops_ = 0;
  uint64_t emitted_batches_ = 0;

  // Phase 3 latency samples (us).
  std::vector<int64_t> latencies_us_;

  // Test-only ramp validation scratch/counters.
  std::vector<uint16_t> validation_host_buf_;
  uint64_t validation_batches_ = 0;
  uint64_t validation_partial_batches_ = 0;
  uint64_t validation_rows_checked_ = 0;
  uint64_t validation_mismatches_ = 0;
  bool validation_first_mismatch_reported_ = false;

  // Device-memory header extraction scratch, grown to the largest seen burst.
  uint32_t burst_header_capacity_ = 0;
  std::vector<uint8_t*> burst_packet_ptrs_;
  uint8_t** h_burst_ptrs_ = nullptr;
  uint8_t** d_burst_ptrs_ = nullptr;
  stem::PacketHeaderInfo* h_burst_headers_ = nullptr;
  stem::PacketHeaderInfo* d_burst_headers_ = nullptr;
};

// ---------------------------------------------------------------------------
// RX worker.
// ---------------------------------------------------------------------------
void rx_worker(const StemRxConfig& cfg, std::atomic<bool>& stop, RxRunStatus* status) {
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
  std::printf("  incomplete batches: %lu\n",
              static_cast<unsigned long>(asm_state.incomplete_batches()));
  std::printf("  incomplete missing: %lu\n",
              static_cast<unsigned long>(asm_state.incomplete_missing_total()));
  std::printf("  incomplete max    : %lu\n",
              static_cast<unsigned long>(asm_state.incomplete_missing_max()));
  std::printf("  sink pool drops  : %lu\n",
              static_cast<unsigned long>(asm_state.output_pool_drops()));
  std::printf("  sink queued      : %lu\n",
              static_cast<unsigned long>(asm_state.sink_queued()));
  std::printf("  sink written     : %lu\n",
              static_cast<unsigned long>(asm_state.sink_written()));
  std::printf("  sink errors      : %lu\n",
              static_cast<unsigned long>(asm_state.sink_errors()));
  if (cfg.validate_tx_ramp) {
    std::printf("  ramp batches     : %lu\n",
                static_cast<unsigned long>(asm_state.validation_batches()));
    std::printf("  ramp partial skip: %lu\n",
                static_cast<unsigned long>(asm_state.validation_partial_batches()));
    std::printf("  ramp rows checked: %lu\n",
                static_cast<unsigned long>(asm_state.validation_rows_checked()));
    std::printf("  ramp mismatches  : %lu\n",
                static_cast<unsigned long>(asm_state.validation_mismatches()));
    const bool failed =
        asm_state.validation_batches() == 0 || asm_state.validation_mismatches() != 0;
    if (failed) {
      std::fprintf(stderr,
                   "validate_tx_ramp failed: full_batches=%lu mismatches=%lu\n",
                   static_cast<unsigned long>(asm_state.validation_batches()),
                   static_cast<unsigned long>(asm_state.validation_mismatches()));
      if (status != nullptr) {
        status->validation_failed.store(true, std::memory_order_release);
      }
    }
  }

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
  bool cli_validate_tx_ramp = false;
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--seconds" && i + 1 < argc) {
      cli_seconds = std::stod(argv[++i]);
    } else if (flag == "--validate-ramp") {
      cli_validate_tx_ramp = true;
    }
  }

  const auto root = YAML::LoadFile(argv[1]);
  if (daqiri::daqiri_init(argv[1]) != daqiri::Status::SUCCESS) {
    std::cerr << "daqiri_init failed for " << argv[1] << "\n";
    return 1;
  }

  StemRxConfig cfg = parse_stem_rx_cfg(root);
  if (cli_seconds > -1.5) { cfg.total_time_to_recv_s = cli_seconds; }
  if (cli_validate_tx_ramp) { cfg.validate_tx_ramp = true; }

  std::cout << "stem_daqiri_rx starting on '" << cfg.interface_name
            << "' frames_per_tensor=" << cfg.frames_per_tensor
            << " header_size=" << cfg.header_size
            << " payload_size=" << cfg.payload_size
            << " gpu_header_extract=" << (cfg.gpu_header_extract ? "true" : "false")
            << " validate_tx_ramp=" << (cfg.validate_tx_ramp ? "true" : "false")
            << " writer.noop=" << (cfg.writer.noop ? "true" : "false")
            << " writer.num_concurrent=" << cfg.writer.num_concurrent
            << " source_mask=0x" << std::hex << cfg.expected_source_mask
            << std::dec
            << " duration=" << cfg.total_time_to_recv_s << " s\n";

  std::atomic<bool> stop{false};
  RxRunStatus status;
  std::thread t(rx_worker, cfg, std::ref(stop), &status);
  t.join();

  daqiri::print_stats();
  daqiri::shutdown();
  return status.validation_failed.load(std::memory_order_acquire) ? 2 : 0;
}
