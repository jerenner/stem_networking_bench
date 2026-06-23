/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Phase 2 stem_daqiri_rx: daqiri-based RX that assembles STEM-format UDP
 * packets into batched GPU frame tensors of shape
 *   [frames_per_tensor, 1024, 3840] uint16
 *
 * Reassembly is tile-readout only (mirrors upstream/tiling): each packet
 * is scattered into its (ZLP or core) tile within the 1024 x 3840 frame
 * plane via stem_gather_tile_packets_by_placement. The legacy row-based
 * gather was removed because LBNL's FPGA can only emit tile-shaped
 * payloads.
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
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

struct ProcessorConfig {
  bool noop = true;
  bool subtract_dark_frame = false;
  std::string dark_frame_path;
  std::string dark_frame_dataset = "/processed";
  bool apply_valid_pixel_mask = false;
  std::string valid_pixel_mask_dataset = "/valid_pixel_mask";
  bool apply_dynamic_half_column_mask = false;
  uint32_t dynamic_mask_median_window_pixels = 31;
  float dynamic_mask_threshold_ratio = 1.0f;
  float dynamic_mask_threshold_offset = 500.0f;
};

struct ReplayerConfig {
  std::string filepath;
  std::string dataset_name = "/frames";
  bool repeat = false;
  uint64_t count = 0;
  uint64_t start_frame = 0;
  uint32_t frames_per_tensor = 128;
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

  // Header/data split mode: daqiri RX queue writes the L2/L3/L4+STEM header
  // into segment 0 and the tile payload into segment 1. This lets IGX steer
  // payload bytes directly into device memory while keeping headers CPU
  // readable.
  bool hds = false;

  // Tile-readout reassembly (the only path; mirrors upstream/tiling
  // cpp/run_with_network_fpga.yaml). The RX reinterprets each packet's
  // (source_id, row_offset) as a tile_index (compacted
  // source_ordinal * 120 + row_offset) and scatters payloads into a full
  // [frames, FRAME_HEIGHT, FRAME_WIDTH] uint16 plane using the ZLP/core
  // tile geometry. row_offset >= 120 is dropped (counted as
  // `tile dropped pkts`). The legacy row-based reassembly path was
  // removed because LBNL's FPGA cannot emit row-shaped payloads.
  //
  // When the daqiri test TX still carries 3840-sample row payloads but we
  // want to fill a 4096-sample tile, wrap the first 256 samples to the end
  // of the tile. Matches Holoscan's tile_duplicate_prefix_to_simulate_payload
  // knob. Set to false when the source is the real FPGA (full tile payloads).
  bool tile_duplicate_prefix_to_simulate_payload = true;

  ProcessorConfig processor;
  WriterConfig writer;
};

WriterConfig parse_writer_cfg(const YAML::Node& root) {
  WriterConfig cfg;
  if (!root["writer"]) { return cfg; }
  const auto wr = root["writer"];
  cfg.filepath = wr["filepath"].as<std::string>(cfg.filepath);
  cfg.dataset_name = wr["dataset_name"].as<std::string>(cfg.dataset_name);
  cfg.noop = wr["noop"].as<bool>(cfg.noop);
  cfg.num_concurrent = wr["num_concurrent"].as<uint32_t>(cfg.num_concurrent);
  if (cfg.num_concurrent == 0) { cfg.num_concurrent = 1; }
  return cfg;
}

ProcessorConfig parse_processor_cfg(const YAML::Node& root,
                                    const YAML::Node& legacy_rx) {
  ProcessorConfig cfg;

  if (legacy_rx) {
    cfg.subtract_dark_frame =
        legacy_rx["subtract_dark"].as<bool>(cfg.subtract_dark_frame);
    cfg.subtract_dark_frame =
        legacy_rx["subtract_dark_frame"].as<bool>(cfg.subtract_dark_frame);
    cfg.apply_valid_pixel_mask =
        legacy_rx["apply_valid_pixel_mask"].as<bool>(cfg.apply_valid_pixel_mask);
  }

  if (root["processor"]) {
    const auto p = root["processor"];
    cfg.noop = p["noop"].as<bool>(cfg.noop);
    cfg.subtract_dark_frame =
        p["subtract_dark_frame"].as<bool>(cfg.subtract_dark_frame);
    cfg.dark_frame_path =
        p["dark_frame_path"].as<std::string>(cfg.dark_frame_path);
    cfg.dark_frame_dataset =
        p["dark_frame_dataset"].as<std::string>(cfg.dark_frame_dataset);
    cfg.apply_valid_pixel_mask =
        p["apply_valid_pixel_mask"].as<bool>(cfg.apply_valid_pixel_mask);
    cfg.valid_pixel_mask_dataset =
        p["valid_pixel_mask_dataset"].as<std::string>(
            cfg.valid_pixel_mask_dataset);
    cfg.apply_dynamic_half_column_mask =
        p["apply_dynamic_half_column_mask"].as<bool>(
            cfg.apply_dynamic_half_column_mask);
    cfg.dynamic_mask_median_window_pixels =
        p["dynamic_mask_median_window_pixels"].as<uint32_t>(
            cfg.dynamic_mask_median_window_pixels);
    cfg.dynamic_mask_threshold_ratio =
        p["dynamic_mask_threshold_ratio"].as<float>(
            cfg.dynamic_mask_threshold_ratio);
    cfg.dynamic_mask_threshold_offset =
        p["dynamic_mask_threshold_offset"].as<float>(
            cfg.dynamic_mask_threshold_offset);
  }

  return cfg;
}

void apply_stem_rx_node(StemRxConfig& cfg, const YAML::Node& rx) {
  if (!rx) { return; }
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
  cfg.hds =
      rx["hds"].as<bool>(cfg.hds);
  cfg.tile_duplicate_prefix_to_simulate_payload =
      rx["tile_duplicate_prefix_to_simulate_payload"].as<bool>(
          cfg.tile_duplicate_prefix_to_simulate_payload);
}

void validate_stem_rx_cfg(const StemRxConfig& cfg, int num_receivers) {
  if (num_receivers > 1 && !cfg.writer.noop) {
    throw std::runtime_error(
        "num_receivers > 1 with writer.noop=false is not supported: "
        "shared HDF5 output would interleave receiver streams by arrival order");
  }
  if (cfg.hds && cfg.gpu_header_extract) {
    throw std::runtime_error(
        "stem_rx.hds=true and stem_rx.gpu_header_extract=true are mutually "
        "exclusive; HDS already keeps headers CPU-readable");
  }
#ifndef STEM_DAQIRI_HAVE_HDF5
  if (!cfg.writer.noop) {
    throw std::runtime_error(
        "writer.noop=false requires HDF5 support; rebuild with "
        "STEM_DAQIRI_REQUIRE_HDF5=ON or set writer.noop=true");
  }
#endif
}

std::vector<StemRxConfig> parse_stem_rx_cfgs(const YAML::Node& root) {
  if (!root["stem_rx"]) {
    throw std::runtime_error("config missing top-level 'stem_rx' block");
  }

  const auto stem_rx = root["stem_rx"];
  const int num_receivers =
      root["num_receivers"].as<int>(stem_rx["num_receivers"].as<int>(1));
  if (num_receivers < 1) {
    throw std::runtime_error("num_receivers must be >= 1");
  }

  StemRxConfig base;
  base.writer = parse_writer_cfg(root);
  base.processor = parse_processor_cfg(root, stem_rx);
  apply_stem_rx_node(base, stem_rx);

  std::vector<StemRxConfig> cfgs;
  cfgs.reserve(static_cast<size_t>(num_receivers));
  if (num_receivers == 1 && !stem_rx["receiver0"]) {
    validate_stem_rx_cfg(base, num_receivers);
    cfgs.push_back(std::move(base));
    return cfgs;
  }

  for (int i = 0; i < num_receivers; ++i) {
    const std::string key = "receiver" + std::to_string(i);
    if (!stem_rx[key]) {
      throw std::runtime_error("num_receivers requires stem_rx." + key);
    }
    StemRxConfig cfg = base;
    apply_stem_rx_node(cfg, stem_rx[key]);
    validate_stem_rx_cfg(cfg, num_receivers);
    cfgs.push_back(std::move(cfg));
  }
  return cfgs;
}

ReplayerConfig parse_replayer_cfg(const YAML::Node& root) {
  if (!root["replayer"]) {
    throw std::runtime_error("source: hdf5 requires a top-level 'replayer' block");
  }
  ReplayerConfig cfg;
  const auto rp = root["replayer"];
  cfg.filepath = rp["filepath"].as<std::string>(cfg.filepath);
  cfg.dataset_name = rp["dataset_name"].as<std::string>(cfg.dataset_name);
  cfg.repeat = rp["repeat"].as<bool>(cfg.repeat);
  cfg.count = rp["count"].as<uint64_t>(cfg.count);
  cfg.start_frame = rp["start_frame"].as<uint64_t>(cfg.start_frame);
  cfg.frames_per_tensor =
      rp["frames_per_tensor"].as<uint32_t>(cfg.frames_per_tensor);
  if (cfg.filepath.empty()) {
    throw std::runtime_error("replayer.filepath must not be empty");
  }
  if (cfg.frames_per_tensor == 0) {
    throw std::runtime_error("replayer.frames_per_tensor must be > 0");
  }
  if (cfg.repeat) {
    throw std::runtime_error(
        "DAQIRI HDF5 replay is finite; set replayer.repeat=false for parity runs");
  }
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
  uint16_t tile_index     = 0;
  uint16_t row_number     = 0;
  uint16_t source_id      = 0xFFFF;
  std::shared_ptr<BurstHolder> holder;
};

struct OutputSlot {
  uint8_t* gpu_u16 = nullptr;
  float* gpu_float = nullptr;
  float* gpu_reduced = nullptr;
  float* gpu_batch_mean = nullptr;
  cudaEvent_t ready = nullptr;
  std::atomic<bool> leased{false};
  uint64_t batch_index = 0;
  uint32_t frames = 0;
  void* output_ptr = nullptr;
  bool output_float = false;
  uint32_t output_rank = 3;
  uint32_t output_frames = 0;
  uint32_t output_height = 0;
  uint32_t output_width = 0;
};

struct Hdf5FrameData {
  std::vector<float> pixels;
  uint32_t height = 0;
  uint32_t width = 0;
  std::string dataset_path;
};

std::string normalize_hdf5_dataset_path(const std::string& dataset_path) {
  if (dataset_path.empty()) { return "/processed"; }
  return dataset_path.front() == '/' ? dataset_path : "/" + dataset_path;
}

#ifdef STEM_DAQIRI_HAVE_HDF5
Hdf5FrameData read_single_frame_float_dataset(const std::string& file_path,
                                              const std::string& dataset_path) {
  const std::string normalized_path = normalize_hdf5_dataset_path(dataset_path);
  H5::H5File file(file_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(normalized_path);
  H5::DataSpace dataspace = dataset.getSpace();

  const int rank = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> dims(static_cast<size_t>(rank));
  dataspace.getSimpleExtentDims(dims.data());

  Hdf5FrameData frame;
  frame.dataset_path = normalized_path;
  if (rank == 2) {
    frame.height = static_cast<uint32_t>(dims[0]);
    frame.width = static_cast<uint32_t>(dims[1]);
  } else if (rank == 3 && dims[0] == 1) {
    frame.height = static_cast<uint32_t>(dims[1]);
    frame.width = static_cast<uint32_t>(dims[2]);
  } else {
    throw std::runtime_error(
        "HDF5 correction dataset must have shape [H,W] or [1,H,W]");
  }

  const size_t num_pixels =
      static_cast<size_t>(frame.height) * static_cast<size_t>(frame.width);
  frame.pixels.resize(num_pixels);
  dataset.read(frame.pixels.data(), H5::PredType::NATIVE_FLOAT);
  return frame;
}
#endif

class AsyncFrameSink {
 public:
  explicit AsyncFrameSink(const WriterConfig& cfg) : cfg_(cfg) {
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
    const uint32_t batch_frames =
        slot->output_rank == 2 ? 1 : slot->output_frames;
    const size_t elem_size =
        slot->output_float ? sizeof(float) : sizeof(uint16_t);
    const size_t elems = static_cast<size_t>(batch_frames) *
                         slot->output_height *
                         slot->output_width;
    const size_t bytes = elems * elem_size;
    if (host_buffer_.size() < bytes) { host_buffer_.resize(bytes); }

    STEM_CUDA_TRY(cudaMemcpy(host_buffer_.data(), slot->output_ptr, bytes,
                             cudaMemcpyDeviceToHost));

#ifdef STEM_DAQIRI_HAVE_HDF5
    if (!file_) {
      errors_++;
      return;
    }
    try {
      const H5::DataType h5_type =
          slot->output_float ? H5::PredType::NATIVE_FLOAT
                             : H5::PredType::NATIVE_UINT16;
      if (!dataset_) {
        hsize_t dims[3] = {batch_frames, slot->output_height, slot->output_width};
        hsize_t max_dims[3] = {H5S_UNLIMITED, slot->output_height, slot->output_width};
        H5::DataSpace filespace(3, dims, max_dims);
        H5::DSetCreatPropList prop;
        hsize_t chunk_dims[3] = {1, slot->output_height, slot->output_width};
        prop.setChunk(3, chunk_dims);
        dataset_ = std::make_unique<H5::DataSet>(
            file_->createDataSet(cfg_.dataset_name, h5_type, filespace, prop));
        dataset_float_ = slot->output_float;
        dataset_height_ = slot->output_height;
        dataset_width_ = slot->output_width;
        current_frames_ = batch_frames;
      } else {
        if (dataset_float_ != slot->output_float ||
            dataset_height_ != slot->output_height ||
            dataset_width_ != slot->output_width) {
          std::fprintf(stderr,
                       "HDF5 write shape/type changed after dataset creation\n");
          errors_++;
          return;
        }
        current_frames_ += batch_frames;
        hsize_t dims[3] = {current_frames_, dataset_height_, dataset_width_};
        dataset_->extend(dims);
      }

      H5::DataSpace filespace(dataset_->getSpace());
      hsize_t offset[3] = {current_frames_ - batch_frames, 0, 0};
      hsize_t count[3] = {batch_frames, slot->output_height, slot->output_width};
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
  bool dataset_float_ = false;
  hsize_t dataset_height_ = 0;
  hsize_t dataset_width_ = 0;
#endif
};

class FramePipeline {
 public:
  FramePipeline(const ProcessorConfig& processor,
                const WriterConfig& writer,
                uint32_t height,
                uint32_t width)
      : processor_(processor),
        height_(height),
        width_(width),
        sink_(writer) {
    if (height_ == 0 || width_ == 0) {
      throw std::runtime_error("FramePipeline requires non-zero frame dimensions");
    }
    if (processor_.dynamic_mask_median_window_pixels == 0 ||
        processor_.dynamic_mask_median_window_pixels > 129) {
      throw std::runtime_error(
          "processor.dynamic_mask_median_window_pixels must be in [1, 129]");
    }
    load_correction_inputs();
  }

  ~FramePipeline() {
    if (gpu_dark_frame_) { cudaFree(gpu_dark_frame_); }
    if (gpu_valid_mask_) { cudaFree(gpu_valid_mask_); }
  }

  FramePipeline(const FramePipeline&) = delete;
  FramePipeline& operator=(const FramePipeline&) = delete;

  bool produces_float() const {
    return !processor_.noop ||
           processor_.subtract_dark_frame ||
           processor_.apply_valid_pixel_mask ||
           processor_.apply_dynamic_half_column_mask;
  }

  void allocate_slot_buffers(OutputSlot* slot, uint32_t max_frames) const {
    const uint64_t frame_pixels = static_cast<uint64_t>(height_) * width_;
    const uint64_t full_values = static_cast<uint64_t>(max_frames) * frame_pixels;
    if (produces_float()) {
      STEM_CUDA_TRY(cudaMalloc(&slot->gpu_float,
                               full_values * sizeof(float)));
    }
    if (!processor_.noop) {
      STEM_CUDA_TRY(cudaMalloc(&slot->gpu_reduced,
                               frame_pixels * sizeof(float)));
    }
    if (processor_.apply_dynamic_half_column_mask) {
      STEM_CUDA_TRY(cudaMalloc(&slot->gpu_batch_mean,
                               frame_pixels * sizeof(float)));
    }
  }

  void process_slot(OutputSlot* slot, uint32_t frames, cudaStream_t stream) const {
    if (frames == 0) { return; }

    if (produces_float()) {
      stem::stem_dark_correct_uint16_to_float(
          reinterpret_cast<const uint16_t*>(slot->gpu_u16),
          processor_.subtract_dark_frame ? gpu_dark_frame_ : nullptr,
          processor_.apply_valid_pixel_mask ? gpu_valid_mask_ : nullptr,
          slot->gpu_float,
          frames,
          height_,
          width_,
          processor_.subtract_dark_frame,
          processor_.apply_valid_pixel_mask,
          stream);

      if (processor_.apply_dynamic_half_column_mask) {
        stem::stem_compute_frame_mean_float(
            slot->gpu_float,
            slot->gpu_batch_mean,
            frames,
            height_,
            width_,
            stream);
        stem::stem_apply_dynamic_half_column_mask_float(
            slot->gpu_float,
            slot->gpu_batch_mean,
            frames,
            height_,
            width_,
            processor_.dynamic_mask_median_window_pixels,
            processor_.dynamic_mask_threshold_ratio,
            processor_.dynamic_mask_threshold_offset,
            stream);
      }

      if (!processor_.noop) {
        stem::stem_sum_frames_float_to_frame(
            slot->gpu_float,
            slot->gpu_reduced,
            frames,
            height_,
            width_,
            stream);
        slot->output_ptr = slot->gpu_reduced;
        slot->output_rank = 2;
        slot->output_frames = 1;
      } else {
        slot->output_ptr = slot->gpu_float;
        slot->output_rank = 3;
        slot->output_frames = frames;
      }
      slot->output_float = true;
    } else {
      slot->output_ptr = slot->gpu_u16;
      slot->output_float = false;
      slot->output_rank = 3;
      slot->output_frames = frames;
    }

    slot->frames = frames;
    slot->output_height = height_;
    slot->output_width = width_;
  }

  void enqueue(OutputSlot* slot) { sink_.enqueue(slot); }

  uint64_t queued() const { return sink_.queued(); }
  uint64_t written() const { return sink_.written(); }
  uint64_t errors() const { return sink_.errors(); }

 private:
  void validate_correction_shape(const Hdf5FrameData& frame,
                                 const std::string& label) const {
    if (frame.height != height_ || frame.width != width_) {
      throw std::runtime_error(
          label + " shape does not match incoming frame shape");
    }
  }

  void load_correction_inputs() {
    const bool needs_hdf5 =
        processor_.subtract_dark_frame || processor_.apply_valid_pixel_mask;
    if (!needs_hdf5) { return; }
    if (processor_.dark_frame_path.empty()) {
      throw std::runtime_error(
          "processor dark/mask correction requires dark_frame_path");
    }

#ifndef STEM_DAQIRI_HAVE_HDF5
    throw std::runtime_error(
        "processor requested HDF5 correction inputs, but stem_daqiri_rx "
        "was built without HDF5 support");
#else
    H5::Exception::dontPrint();
    const uint64_t frame_pixels = static_cast<uint64_t>(height_) * width_;
    try {
      if (processor_.subtract_dark_frame) {
        Hdf5FrameData dark = read_single_frame_float_dataset(
            processor_.dark_frame_path,
            processor_.dark_frame_dataset);
        validate_correction_shape(dark, "dark frame");
        STEM_CUDA_TRY(cudaMalloc(&gpu_dark_frame_,
                                 frame_pixels * sizeof(float)));
        STEM_CUDA_TRY(cudaMemcpy(gpu_dark_frame_,
                                 dark.pixels.data(),
                                 dark.pixels.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));
        std::cout << "loaded dark frame " << processor_.dark_frame_path
                  << ":" << dark.dataset_path << " shape "
                  << dark.height << "x" << dark.width << "\n";
      }
      if (processor_.apply_valid_pixel_mask) {
        Hdf5FrameData mask = read_single_frame_float_dataset(
            processor_.dark_frame_path,
            processor_.valid_pixel_mask_dataset);
        validate_correction_shape(mask, "valid pixel mask");
        STEM_CUDA_TRY(cudaMalloc(&gpu_valid_mask_,
                                 frame_pixels * sizeof(float)));
        STEM_CUDA_TRY(cudaMemcpy(gpu_valid_mask_,
                                 mask.pixels.data(),
                                 mask.pixels.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));
        std::cout << "loaded valid pixel mask " << processor_.dark_frame_path
                  << ":" << mask.dataset_path << " shape "
                  << mask.height << "x" << mask.width << "\n";
      }
    } catch (const H5::Exception& e) {
      throw std::runtime_error(
          std::string("failed to load HDF5 correction input: ") +
          e.getCDetailMsg());
    }
#endif
  }

  ProcessorConfig processor_;
  uint32_t height_ = 0;
  uint32_t width_ = 0;
  float* gpu_dark_frame_ = nullptr;
  float* gpu_valid_mask_ = nullptr;
  AsyncFrameSink sink_;
};

// ---------------------------------------------------------------------------
// Frame assembler. Owns the GPU output frame buffer, the placement scratch
// arrays, the pending packet vector, and the Phase 3 latency samples.
// ---------------------------------------------------------------------------
class FrameAssembler {
 public:
  FrameAssembler(const StemRxConfig& cfg,
                 std::shared_ptr<FramePipeline> pipeline)
      : cfg_(cfg),
        pipeline_(std::move(pipeline)),
        expected_source_count_(
            __builtin_popcount(cfg.expected_source_mask & 0xff)),
        rows_per_tensor_(cfg.frames_per_tensor * stem::FRAME_HEIGHT),
        packet_cells_per_frame_(
            expected_source_count_ * stem::TILE_PACKETS_PER_SOURCE),
        placement_capacity_(
            cfg.frames_per_tensor * stem::FULL_FRAME_TILE_PACKETS),
        expected_packets_per_batch_(
            cfg.frames_per_tensor *
            expected_source_count_ *
            stem::TILE_PACKETS_PER_SOURCE) {
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_pkt_ptrs_),
                                 sizeof(uint8_t*) * placement_capacity_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_pkt_ptrs_),
                             sizeof(uint8_t*) * placement_capacity_));
    STEM_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_placements_),
                                 sizeof(stem::PacketPlacement) * placement_capacity_));
    STEM_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_placements_),
                             sizeof(stem::PacketPlacement) * placement_capacity_));
    STEM_CUDA_TRY(cudaStreamCreate(&stream_));

    const uint32_t slot_count =
        cfg.writer.noop ? 1 : std::max<uint32_t>(1, cfg.writer.num_concurrent);
    output_slots_.reserve(slot_count);
    for (uint32_t i = 0; i < slot_count; ++i) {
      auto slot = std::make_unique<OutputSlot>();
      STEM_CUDA_TRY(cudaMalloc(&slot->gpu_u16,
                               static_cast<uint64_t>(cfg.frames_per_tensor) *
                               stem::FRAME_SIZE_BYTES));
      pipeline_->allocate_slot_buffers(slot.get(), cfg.frames_per_tensor);
      STEM_CUDA_TRY(cudaEventCreateWithFlags(&slot->ready,
                                             cudaEventDisableTiming));
      output_slots_.push_back(std::move(slot));
    }

    // Reserve once up front so push_back doesn't realloc 19 times.
    if (cfg_.capture_latency) {
      latencies_us_.reserve(kMaxLatencySamples);
    }
    const uint32_t total_cells =
        cfg.frames_per_tensor * packet_cells_per_frame_;
    pending_packets_.reserve(placement_capacity_ +
                             cfg.batch_close_slack_packets + 4096);
    current_batch_occupied_.assign(total_cells, 0);
    emit_cell_generation_.assign(total_cells, 0);
  }

  ~FrameAssembler() {
    bool slots_busy = true;
    while (slots_busy) {
      slots_busy = false;
      for (const auto& slot : output_slots_) {
        if (slot->leased.load(std::memory_order_acquire)) {
          slots_busy = true;
          break;
        }
      }
      if (slots_busy) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    pending_packets_.clear();
    for (auto& slot : output_slots_) {
      if (slot->gpu_u16) { cudaFree(slot->gpu_u16); }
      if (slot->gpu_float) { cudaFree(slot->gpu_float); }
      if (slot->gpu_reduced) { cudaFree(slot->gpu_reduced); }
      if (slot->gpu_batch_mean) { cudaFree(slot->gpu_batch_mean); }
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
    if (stream_) { cudaStreamDestroy(stream_); }
  }

  std::vector<int64_t>& latencies_us() { return latencies_us_; }
  uint64_t tile_packets_ignored() const { return tile_packets_ignored_; }
  uint64_t incomplete_batches() const { return incomplete_batches_; }
  uint64_t incomplete_missing_total() const { return incomplete_missing_total_; }
  uint64_t incomplete_missing_max() const { return incomplete_missing_max_; }
  uint64_t output_pool_drops() const { return output_pool_drops_; }
  uint64_t sink_queued() const { return pipeline_ ? pipeline_->queued() : 0; }
  uint64_t sink_written() const { return pipeline_ ? pipeline_->written() : 0; }
  uint64_t sink_errors() const { return pipeline_ ? pipeline_->errors() : 0; }

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

    validate_packet_layout_once(burst);

    if (cfg_.hds) {
      if (n > 0 && burst->hdr.hdr.num_segs < 2) {
        fail_packet_layout("HDS enabled but burst has " +
                           std::to_string(burst->hdr.hdr.num_segs) +
                           " segment(s); expected at least 2");
      }

      for (int i = 0; i < n; ++i) {
        auto* header_ptr = static_cast<uint8_t*>(
            daqiri::get_segment_packet_ptr(burst, 0, i));
        auto* payload_ptr = static_cast<uint8_t*>(
            daqiri::get_segment_packet_ptr(burst, 1, i));
        if (header_ptr == nullptr || payload_ptr == nullptr) {
          fail_packet_layout("HDS packet " + std::to_string(i) +
                             " has null segment pointer(s): seg0=" +
                             std::to_string(reinterpret_cast<uintptr_t>(header_ptr)) +
                             " seg1=" +
                             std::to_string(reinterpret_cast<uintptr_t>(payload_ptr)));
        }

        const stem::PacketHeaderInfo header =
            parse_packet_header(header_ptr + cfg_.header_size);
        admit_packet(payload_ptr, header, holder, drops_unexpected_source);
      }
    } else if (cfg_.gpu_header_extract) {
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
        const stem::PacketHeaderInfo header =
            parse_packet_header(ptr + cfg_.header_size);
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

  uint16_t available_payload_len() const {
    return cfg_.tile_duplicate_prefix_to_simulate_payload
        ? static_cast<uint16_t>(stem::STEM_PAYLOAD_SIZE)
        : static_cast<uint16_t>(stem::TILE_PAYLOAD_BYTES);
  }

  uint16_t packet_header_segment_len() const {
    return static_cast<uint16_t>(cfg_.header_size + stem::STEM_HEADER_SIZE);
  }

  uint16_t gather_header_len() const {
    return cfg_.hds ? 0 : packet_header_segment_len();
  }

  stem::PacketHeaderInfo parse_packet_header(const uint8_t* stem_hdr) const {
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
    return header;
  }

  [[noreturn]] void fail_packet_layout(const std::string& detail) const {
    std::cerr << "stem_daqiri_rx packet layout error: " << detail << "\n";
    std::exit(1);
  }

  void validate_packet_layout_once(daqiri::BurstParams* burst) {
    if (packet_layout_checked_) { return; }
    const auto n = daqiri::get_num_packets(burst);
    if (n <= 0) { return; }

    const uint32_t header_len = packet_header_segment_len();
    const uint32_t payload_len = available_payload_len();

    if (cfg_.hds) {
      if (burst->hdr.hdr.num_segs < 2) {
        fail_packet_layout("HDS enabled but first non-empty burst has " +
                           std::to_string(burst->hdr.hdr.num_segs) +
                           " segment(s); expected at least 2");
      }
      const uint32_t actual_header_len = static_cast<uint32_t>(
          daqiri::get_segment_packet_length(burst, 0, 0));
      const uint32_t actual_payload_len = static_cast<uint32_t>(
          daqiri::get_segment_packet_length(burst, 1, 0));
      if (actual_header_len != header_len || actual_payload_len != payload_len) {
        fail_packet_layout(
            "HDS split mismatch on first packet: seg0=" +
            std::to_string(actual_header_len) + "B expected " +
            std::to_string(header_len) + "B, seg1=" +
            std::to_string(actual_payload_len) + "B expected " +
            std::to_string(payload_len) + "B");
      }
      std::cout << "stem_daqiri_rx HDS layout verified: seg0="
                << actual_header_len << "B seg1=" << actual_payload_len
                << "B; epoch_us remains in seg0 at byte "
                << (cfg_.header_size + stem::STEM_HDR_OFF_EPOCH_US)
                << "\n";
    } else {
      if (burst->hdr.hdr.num_segs < 1) {
        fail_packet_layout("non-HDS burst has no packet segment");
      }
      const uint32_t actual_packet_len = static_cast<uint32_t>(
          daqiri::get_segment_packet_length(burst, 0, 0));
      const uint32_t expected_packet_len = header_len + payload_len;
      if (actual_packet_len != expected_packet_len) {
        fail_packet_layout(
            "single-segment packet length mismatch on first packet: seg0=" +
            std::to_string(actual_packet_len) + "B expected " +
            std::to_string(expected_packet_len) +
            "B (header+STEM header " + std::to_string(header_len) +
            "B, payload " + std::to_string(payload_len) + "B)");
      }
    }

    packet_layout_checked_ = true;
  }

  // Number of expected sources strictly preceding `source_id` in the mask
  // ordering, used to compact 8 sources of 128 row_offsets each into the
  // 960 tile slots/frame. Matches StemReceiverOp upstream.
  uint32_t expected_source_ordinal(uint16_t source_id) const {
    uint32_t ordinal = 0;
    for (uint32_t s = 0; s < source_id && s < 8; ++s) {
      if ((cfg_.expected_source_mask >> s) & 0x1u) { ordinal++; }
    }
    return ordinal;
  }

  // Translate (source_id, row_offset) into a tile_index. Returns false
  // (and increments tile_packets_ignored_) for packets with
  // row_offset >= TILE_PACKETS_PER_SOURCE: the daqiri test TX still emits
  // 128 rows/source but only 120 of them map to tile slots. The real FPGA
  // source naturally only emits the 120 in-tile packets.
  bool compute_tile_index(const stem::PacketHeaderInfo& header,
                          uint16_t* tile_index) {
    if (header.row_offset >= stem::TILE_PACKETS_PER_SOURCE) {
      tile_packets_ignored_++;
      return false;
    }
    const uint32_t source_ordinal = expected_source_ordinal(header.source_id);
    const uint32_t compact =
        source_ordinal * stem::TILE_PACKETS_PER_SOURCE + header.row_offset;
    if (compact >= packet_cells_per_frame_) {
      tile_packets_ignored_++;
      return false;
    }
    *tile_index = static_cast<uint16_t>(compact);
    return true;
  }

  // Cell index used by the per-batch occupied bitmap and emit-generation
  // dedup map: one cell per (frame, compact tile_index).
  uint64_t packet_cell_index(const PacketEntry& entry,
                             uint64_t relative_frame) const {
    return relative_frame *
               static_cast<uint64_t>(packet_cells_per_frame_) +
           static_cast<uint64_t>(entry.tile_index);
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

    uint16_t tile_index = 0;
    if (!compute_tile_index(header, &tile_index)) {
      return;  // counted as tile_packets_ignored_
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
    entry.tile_index = tile_index;
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
    const uint64_t cell = packet_cell_index(pending_packets_.back(), relative_frame);
    // relative_frame < frames_per_tensor and (global_row|tile_index) < cells/frame
    // are both enforced upstream, so cell < total_cells by construction.
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
    if (current_batch_unique_packets_ >= expected_packets_per_batch_) {
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
      if (pkts_to_gather < placement_capacity_) {
        const uint64_t rel_frame =
            read_it->abs_frame - current_batch_start_abs_frame_;
        const uint64_t cell = packet_cell_index(*read_it, rel_frame);
        // cell < total_cells by construction (see admit_packet).
        assert(cell < emit_cell_generation_.size());
        if (emit_cell_generation_[cell] == emit_generation_) {
          // Duplicate (frame, row|tile) within this close; drop the later copy.
          continue;
        }
        emit_cell_generation_[cell] = emit_generation_;
        h_pkt_ptrs_[pkts_to_gather] = read_it->packet_ptr;
        stem::PacketPlacement pl;
        pl.global_row     = read_it->global_row;
        pl.tile_index     = read_it->tile_index;
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
                                    static_cast<uint64_t>(cfg_.frames_per_tensor) *
                                    stem::FRAME_SIZE_BYTES,
                                    stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_pkt_ptrs_, h_pkt_ptrs_,
                                    sizeof(uint8_t*) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));
      STEM_CUDA_TRY(cudaMemcpyAsync(d_placements_, h_placements_,
                                    sizeof(stem::PacketPlacement) * pkts_to_gather,
                                    cudaMemcpyHostToDevice, stream_));

      // Match Holoscan's contract: tile_duplicate_prefix_to_simulate_payload
      // fully determines whether each packet exposes a 7680 B test payload or
      // a native 8192 B tile payload. HDS packet_ptrs already point at seg1
      // payload byte 0; non-HDS packet_ptrs point at the wire packet base.
      stem::stem_gather_tile_packets_by_placement(
          d_pkt_ptrs_, d_placements_, output_slot->gpu_u16,
          available_payload_len(),
          gather_header_len(),
          pkts_to_gather,
          cfg_.frames_per_tensor,
          stem::FRAME_HEIGHT,
          stem::FRAME_WIDTH,
          cfg_.tile_duplicate_prefix_to_simulate_payload,
          stream_);

      pipeline_->process_slot(output_slot, cfg_.frames_per_tensor, stream_);
      output_slot->batch_index = emitted_batches_;
      STEM_CUDA_TRY(cudaEventRecord(output_slot->ready, stream_));
      pipeline_->enqueue(output_slot);
      // frames_assembled counts only batches that produced a GPU tensor;
      // empty closes (pkts_to_gather==0) and pool-starved closes
      // (output_slot==nullptr) do not bump it, so the reported fps reflects
      // real downstream-visible frames.
      (*frames_assembled) += cfg_.frames_per_tensor;
      emitted_batches_++;
    }

    pending_packets_.erase(write_it, pending_packets_.end());

    if (pkts_to_gather < expected_packets_per_batch_) {
      const uint64_t missing = expected_packets_per_batch_ - pkts_to_gather;
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
        const uint64_t cell = packet_cell_index(e, relative_frame);
        // cell < total_cells by construction (see admit_packet).
        assert(cell < current_batch_occupied_.size());
        if (!current_batch_occupied_[cell]) {
          current_batch_occupied_[cell] = 1;
          current_batch_unique_packets_++;
        }
      }
    }
  }

  const StemRxConfig& cfg_;
  std::shared_ptr<FramePipeline> pipeline_;
  const uint32_t      expected_source_count_;
  const uint32_t      rows_per_tensor_;
  const uint32_t      packet_cells_per_frame_;
  const uint32_t      placement_capacity_;
  const uint32_t      expected_packets_per_batch_;
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
  bool       packet_layout_checked_ = false;
  uint64_t   incomplete_batches_ = 0;
  uint64_t   incomplete_missing_total_ = 0;
  uint64_t   incomplete_missing_max_ = 0;

  std::vector<std::unique_ptr<OutputSlot>> output_slots_;
  uint64_t output_pool_drops_ = 0;
  uint64_t emitted_batches_ = 0;

  // Phase 3 latency samples (us).
  std::vector<int64_t> latencies_us_;

  // Packets dropped because their row_offset is >= 120 (outside the 960
  // tiles/frame compact mapping). The daqiri test TX still emits 128
  // rows/source so this is 8 per source per frame in self-test runs; the
  // real FPGA emits only the 120 in-tile packets and this stays at 0.
  uint64_t tile_packets_ignored_ = 0;

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
void rx_worker(const StemRxConfig& cfg,
               std::shared_ptr<FramePipeline> pipeline,
               std::atomic<bool>& stop) {
  const int port_id = daqiri::get_port_id(cfg.interface_name);
  if (port_id < 0) {
    std::cerr << "Invalid RX interface_name: " << cfg.interface_name << "\n";
    stop.store(true);
    return;
  }

  FrameAssembler asm_state(cfg, std::move(pipeline));

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
  std::printf("  tile dropped pkts: %lu  (row_offset >= 120)\n",
              static_cast<unsigned long>(asm_state.tile_packets_ignored()));
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

void print_rx_start(const StemRxConfig& cfg) {
  const uint16_t gather_available_payload_len =
      cfg.tile_duplicate_prefix_to_simulate_payload
          ? static_cast<uint16_t>(stem::STEM_PAYLOAD_SIZE)
          : static_cast<uint16_t>(stem::TILE_PAYLOAD_BYTES);

  std::cout << "stem_daqiri_rx starting on '" << cfg.interface_name
            << "' frames_per_tensor=" << cfg.frames_per_tensor
            << " header_size=" << cfg.header_size
            << " payload_size=" << cfg.payload_size
            << " gpu_header_extract=" << (cfg.gpu_header_extract ? "true" : "false")
            << " hds=" << (cfg.hds ? "true" : "false")
            << " tile_dup_prefix="
            << (cfg.tile_duplicate_prefix_to_simulate_payload ? "true" : "false")
            << " gather_available_payload_len=" << gather_available_payload_len
            << " processor.noop=" << (cfg.processor.noop ? "true" : "false")
            << " subtract_dark_frame="
            << (cfg.processor.subtract_dark_frame ? "true" : "false")
            << " apply_valid_pixel_mask="
            << (cfg.processor.apply_valid_pixel_mask ? "true" : "false")
            << " apply_dynamic_half_column_mask="
            << (cfg.processor.apply_dynamic_half_column_mask ? "true" : "false")
            << " writer.noop=" << (cfg.writer.noop ? "true" : "false")
            << " writer.num_concurrent=" << cfg.writer.num_concurrent
            << " source_mask=0x" << std::hex << cfg.expected_source_mask
            << std::dec
            << " duration=" << cfg.total_time_to_recv_s << " s\n";
}

OutputSlot* acquire_replay_slot(
    const std::vector<std::unique_ptr<OutputSlot>>& slots) {
  for (;;) {
    for (const auto& slot : slots) {
      bool expected = false;
      if (slot->leased.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        return slot.get();
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void wait_for_replay_slots(
    const std::vector<std::unique_ptr<OutputSlot>>& slots) {
  bool busy = true;
  while (busy) {
    busy = false;
    for (const auto& slot : slots) {
      if (slot->leased.load(std::memory_order_acquire)) {
        busy = true;
        break;
      }
    }
    if (busy) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

int run_hdf5_replay(const YAML::Node& root) {
#ifndef STEM_DAQIRI_HAVE_HDF5
  (void)root;
  std::cerr << "source: hdf5 requested, but stem_daqiri_rx was built "
            << "without HDF5 support\n";
  return 1;
#else
  const ReplayerConfig replayer = parse_replayer_cfg(root);
  const WriterConfig writer = parse_writer_cfg(root);
  const ProcessorConfig processor = parse_processor_cfg(root, YAML::Node{});

  H5::Exception::dontPrint();
  H5::H5File file(replayer.filepath, H5F_ACC_RDONLY);
  H5::DataSet dataset =
      file.openDataSet(normalize_hdf5_dataset_path(replayer.dataset_name));
  H5::DataSpace filespace = dataset.getSpace();
  const int rank = filespace.getSimpleExtentNdims();
  if (rank != 3) {
    throw std::runtime_error("replayer dataset must have shape [frames,H,W]");
  }
  hsize_t dims[3] = {0, 0, 0};
  filespace.getSimpleExtentDims(dims);
  if (replayer.start_frame >= dims[0]) {
    throw std::runtime_error("replayer.start_frame is outside the dataset");
  }
  const H5::DataType dataset_type = dataset.getDataType();
  if (dataset_type.getClass() != H5T_INTEGER ||
      dataset_type.getSize() != sizeof(uint16_t)) {
    throw std::runtime_error(
        "DAQIRI HDF5 replay currently expects uint16 [frames,H,W] input");
  }

  auto pipeline = std::make_shared<FramePipeline>(
      processor,
      writer,
      static_cast<uint32_t>(dims[1]),
      static_cast<uint32_t>(dims[2]));

  cudaStream_t stream = nullptr;
  STEM_CUDA_TRY(cudaStreamCreate(&stream));
  const uint32_t slot_count =
      writer.noop ? 1 : std::max<uint32_t>(1, writer.num_concurrent);
  std::vector<std::unique_ptr<OutputSlot>> slots;
  slots.reserve(slot_count);
  const uint64_t frame_pixels = dims[1] * dims[2];
  for (uint32_t i = 0; i < slot_count; ++i) {
    auto slot = std::make_unique<OutputSlot>();
    STEM_CUDA_TRY(cudaMalloc(&slot->gpu_u16,
                             static_cast<uint64_t>(replayer.frames_per_tensor) *
                             frame_pixels * sizeof(uint16_t)));
    pipeline->allocate_slot_buffers(slot.get(), replayer.frames_per_tensor);
    STEM_CUDA_TRY(cudaEventCreateWithFlags(&slot->ready, cudaEventDisableTiming));
    slots.push_back(std::move(slot));
  }

  const uint64_t available_frames = dims[0] - replayer.start_frame;
  const uint64_t target_frames =
      replayer.count > 0 ? replayer.count : available_frames;
  uint64_t emitted_frames = 0;
  uint64_t current_frame = replayer.start_frame;
  uint64_t emitted_batches = 0;
  std::vector<uint16_t> host_buffer;

  const auto start = std::chrono::steady_clock::now();
  while (emitted_frames < target_frames && !g_stop_requested) {
    if (current_frame >= dims[0]) {
      if (replayer.repeat) {
        current_frame = replayer.start_frame;
      } else {
        break;
      }
    }

    const uint64_t frames_left_in_file = dims[0] - current_frame;
    const uint64_t frames_left_target = target_frames - emitted_frames;
    const uint32_t batch_frames = static_cast<uint32_t>(
        std::min<uint64_t>({replayer.frames_per_tensor,
                            frames_left_in_file,
                            frames_left_target}));
    if (batch_frames == 0) { break; }

    hsize_t offset[3] = {current_frame, 0, 0};
    hsize_t count[3] = {batch_frames, dims[1], dims[2]};
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(3, count);
    host_buffer.resize(static_cast<size_t>(batch_frames) *
                       static_cast<size_t>(frame_pixels));
    dataset.read(host_buffer.data(),
                 H5::PredType::NATIVE_UINT16,
                 memspace,
                 filespace);

    OutputSlot* slot = acquire_replay_slot(slots);
    STEM_CUDA_TRY(cudaMemcpyAsync(slot->gpu_u16,
                                  host_buffer.data(),
                                  host_buffer.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    pipeline->process_slot(slot, batch_frames, stream);
    slot->batch_index = emitted_batches++;
    STEM_CUDA_TRY(cudaEventRecord(slot->ready, stream));
    pipeline->enqueue(slot);

    current_frame += batch_frames;
    emitted_frames += batch_frames;
  }

  STEM_CUDA_TRY(cudaStreamSynchronize(stream));
  wait_for_replay_slots(slots);
  for (auto& slot : slots) {
    if (slot->gpu_u16) { cudaFree(slot->gpu_u16); }
    if (slot->gpu_float) { cudaFree(slot->gpu_float); }
    if (slot->gpu_reduced) { cudaFree(slot->gpu_reduced); }
    if (slot->gpu_batch_mean) { cudaFree(slot->gpu_batch_mean); }
    if (slot->ready) { cudaEventDestroy(slot->ready); }
  }
  if (stream) { cudaStreamDestroy(stream); }

  const double secs =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
  const double fps = secs > 0
      ? static_cast<double>(emitted_frames) / secs
      : 0.0;
  std::printf("stem_daqiri_hdf5_replay complete:\n");
  std::printf("  duration         : %.3f s\n", secs);
  std::printf("  frames replayed  : %lu  (fps %.3f)\n",
              static_cast<unsigned long>(emitted_frames), fps);
  std::printf("  sink queued      : %lu\n",
              static_cast<unsigned long>(pipeline->queued()));
  std::printf("  sink written     : %lu\n",
              static_cast<unsigned long>(pipeline->written()));
  std::printf("  sink errors      : %lu\n",
              static_cast<unsigned long>(pipeline->errors()));
  return pipeline->errors() == 0 ? 0 : 1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config.yaml> [--seconds N]\n";
    return 1;
  }
  std::signal(SIGINT, on_sigint);

  double cli_seconds = -2.0;
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--seconds" && i + 1 < argc) {
      cli_seconds = std::stod(argv[++i]);
    }
  }

  try {
    const auto root = YAML::LoadFile(argv[1]);
    const std::string source =
        root["source"].as<std::string>(std::string("network"));
    if (source == "hdf5") {
      return run_hdf5_replay(root);
    }
    if (source != "network") {
      throw std::runtime_error("source must be 'network' or 'hdf5'");
    }

    std::vector<StemRxConfig> cfgs = parse_stem_rx_cfgs(root);
    for (auto& cfg : cfgs) {
      if (cli_seconds > -1.5) { cfg.total_time_to_recv_s = cli_seconds; }
      print_rx_start(cfg);
    }

    auto pipeline = std::make_shared<FramePipeline>(
        cfgs.front().processor,
        cfgs.front().writer,
        stem::FRAME_HEIGHT,
        stem::FRAME_WIDTH);
    if (pipeline->errors() > 0) {
      return 1;
    }

    if (daqiri::daqiri_init(argv[1]) != daqiri::Status::SUCCESS) {
      std::cerr << "daqiri_init failed for " << argv[1] << "\n";
      return 1;
    }

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(cfgs.size());
    for (const auto& cfg : cfgs) {
      threads.emplace_back(rx_worker, cfg, pipeline, std::ref(stop));
    }
    for (auto& t : threads) { t.join(); }

    daqiri::print_stats();
    daqiri::shutdown();
    return pipeline->errors() == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "stem_daqiri_rx failed: " << e.what() << "\n";
    return 1;
  }
}
