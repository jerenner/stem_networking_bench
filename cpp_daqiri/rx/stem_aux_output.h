/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace stem {

enum class ProcessingStage {
  kRaw,
  kDarkSubtracted,
  kDarkBlr,
  kCorrected,
  kThresholded,
  kCounted,
};

ProcessingStage parse_processing_stage(const std::string& value);
const char* processing_stage_name(ProcessingStage stage);

struct ThresholdConfig {
  float zlp = 0.0f;
  float core_loss = 0.0f;
};

struct BurstWriterConfig {
  bool enabled = false;
  ProcessingStage stage = ProcessingStage::kCorrected;
  std::string filepath_template =
      "/tmp/stem_burst_rx{receiver}_{capture}.h5";
  std::string dataset_name = "/frames";
  uint32_t buckets_per_capture = 1;
  uint64_t capture_count = 1;
  bool rearm_after_write = true;
  bool strict_complete = true;
  ThresholdConfig threshold;
};

struct ThinnedStreamConfig {
  bool enabled = false;
  ProcessingStage stage = ProcessingStage::kCorrected;
  std::string endpoint = "tcp://*:5556";
  std::string topic_prefix = "stem";
  double total_refresh_hz = 10.0;
  uint32_t representative_frame_index = 64;
  bool include_representative_frame = true;
  bool include_bucket_sum = true;
  uint32_t queue_depth = 2;
  ThresholdConfig threshold;
};

struct BatchMetadata {
  uint32_t receiver_id = 0;
  std::string interface_name;
  uint64_t batch_index = 0;
  uint64_t first_frame = 0;
  uint32_t frames = 0;
  uint32_t received_packets = 0;
  uint32_t expected_packets = 0;
  bool complete = true;
};

struct BurstWriterStats {
  uint64_t captures_started = 0;
  uint64_t captures_written = 0;
  uint64_t buckets_captured = 0;
  uint64_t buckets_skipped_busy = 0;
  uint64_t buckets_rejected_incomplete = 0;
  uint64_t aborted_captures = 0;
  uint64_t errors = 0;
};

class BurstWriter {
 public:
  struct Reservation {
    void* token = nullptr;
    void* device_ptr = nullptr;
    size_t bytes = 0;
    bool float32 = false;

    explicit operator bool() const { return token != nullptr; }
  };

  BurstWriter(const BurstWriterConfig& config, uint32_t height,
              uint32_t width, uint32_t frames_per_bucket,
              uint32_t receiver_count, bool raw_input_float32);
  ~BurstWriter();

  BurstWriter(const BurstWriter&) = delete;
  BurstWriter& operator=(const BurstWriter&) = delete;

  bool enabled() const;
  ProcessingStage stage() const;
  std::optional<Reservation> reserve(const BatchMetadata& metadata);
  void submit_copy(const Reservation& reservation, const void* source,
                   bool source_float32, cudaStream_t stream);
  void submit_direct(const Reservation& reservation, cudaStream_t stream);
  void drain();
  BurstWriterStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ThinnedStreamStats {
  uint64_t products_queued = 0;
  uint64_t products_published = 0;
  uint64_t products_coalesced = 0;
  uint64_t products_dropped_no_buffer = 0;
  uint64_t send_errors = 0;
};

class ThinnedStreamPublisher {
 public:
  struct Reservation {
    void* token = nullptr;
    float* representative_device = nullptr;
    float* sum_device = nullptr;

    explicit operator bool() const { return token != nullptr; }
  };

  ThinnedStreamPublisher(const ThinnedStreamConfig& config, uint32_t height,
                         uint32_t width, uint32_t frames_per_bucket,
                         uint32_t receiver_count);
  ~ThinnedStreamPublisher();

  ThinnedStreamPublisher(const ThinnedStreamPublisher&) = delete;
  ThinnedStreamPublisher& operator=(const ThinnedStreamPublisher&) = delete;

  bool enabled() const;
  ProcessingStage stage() const;
  std::optional<Reservation> reserve(const BatchMetadata& metadata);
  void submit(const Reservation& reservation, cudaStream_t stream);
  void drain();
  ThinnedStreamStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stem
