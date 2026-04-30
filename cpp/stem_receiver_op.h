/*
 * Original code based on: https://github.com/nvidia-holoscan/holohub/blob/main/applications/adv_networking_bench/cpp/default_bench_op_rx.h
 *
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

#pragma once
#include "advanced_network/common.h"
#include "advanced_network/kernels.h"
#include "kernels.cuh"
#include "nvtx_ranges.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <assert.h>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <cstring>
#include <queue>
#include <string>
#include <sys/time.h>
#include <vector>

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/script.h>

#define BURST_ACCESS_METHOD_RAW_PTR 0
#define BURST_ACCESS_METHOD_DIRECT_ACCESS 1
#define BURST_ACCESS_METHOD BURST_ACCESS_METHOD_RAW_PTR

// New Frame Geometry
#define FRAME_WIDTH 3840
#define FRAME_HEIGHT 1024
#define FRAME_SIZE_BYTES (FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t))

using namespace holoscan::advanced_network;

namespace holoscan::ops {

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         static_cast<int>(_holoscan_cuda_err));                                 \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })


class StemReceiverOp : public Operator {
 public:
  static constexpr int num_concurrent = 4;
  static constexpr int MAX_BURSTS_PER_BATCH = 5000;
  static constexpr uint32_t ROWS_PER_SOURCE = 128;
  static constexpr uint32_t INVALID_HOLDER_INDEX = std::numeric_limits<uint32_t>::max();

  HOLOSCAN_OPERATOR_FORWARD_ARGS(StemReceiverOp)

  struct BatchAggregationParams {
    std::array<BurstParams*, MAX_BURSTS_PER_BATCH> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  struct BurstHolder {
    explicit BurstHolder(BurstParams* burst_in) : burst(burst_in) {}
    ~BurstHolder() {
      if (burst) { free_all_packets_and_burst_rx(burst); }
    }

    BurstHolder(const BurstHolder&) = delete;
    BurstHolder& operator=(const BurstHolder&) = delete;

    BurstParams* burst = nullptr;
  };

  struct PacketEntry {
    void* packet_ptr = nullptr;
    uint64_t abs_frame = 0;
    int16_t global_row = -1;
    uint16_t row_number = 0;
    uint16_t source_id = 0;
    uint32_t holder_index = INVALID_HOLDER_INDEX;
  };

  struct AssembledBatch {
    std::vector<std::shared_ptr<BurstHolder>> holders;
    cudaEvent_t evt = nullptr;
    uint32_t packets_used = 0;
    uint32_t missing_packets = 0;
  };

  struct PendingHolder {
    std::shared_ptr<BurstHolder> holder;
    uint32_t pending_packets = 0;
    uint32_t emit_generation = 0;
  };

  struct HeaderBatchSlot {
    void** h_dev_ptrs = nullptr;
    void* d_dev_ptrs = nullptr;
    PacketHeaderInfo* headers_h = nullptr;
    PacketHeaderInfo* headers_d = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    uint32_t packet_count = 0;
    bool in_flight = false;
    std::vector<std::shared_ptr<BurstHolder>> holders;
    std::vector<uint32_t> packet_holder_indices;
  };

  StemReceiverOp() = default;

  ~StemReceiverOp() {
    HOLOSCAN_LOG_INFO(
        "Finished receiver with {}/{} bytes/packets received, {} packets dropped, and {} packets "
        "ignored from unexpected source IDs. Header batches launched/completed: {}/{}; full-slot waits: {}",
        ttl_bytes_recv_,
        ttl_pkts_recv_,
        ttl_packets_dropped_,
        ttl_packets_ignored_unexpected_source_,
        header_batches_launched_,
        header_batches_completed_,
        header_slot_full_waits_);
    flush_packet_debug_outputs();
    HOLOSCAN_LOG_INFO("StemReceiverOp shutting down");
    freeResources();
  }

  void initialize() override {
    auto has_stop_condition = std::find_if(args().begin(), args().end(), [](const auto& arg) {
      return (arg.name() == "stop_condition");
    });
    if (has_stop_condition == args().end()) {
      auto stop_cond = fragment()->make_condition<BooleanCondition>(name() + "_stop_condition");
      add_arg(Arg("stop_condition", stop_cond));
    }

    // Add a default UnboundedAllocator if no allocator was provided.
    if (!allocator_.has_value()) {
      allocator_ = fragment()->make_resource<UnboundedAllocator>("allocator");
      add_arg(allocator_.get());
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize()");
    holoscan::Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config", interface_name_.get());
      exit(1);
    }

    // Fixed packet payload constants based on protocol
    custom_header_size_ = 64;   // 1 Word of generic info
    nom_payload_size_ = 7680;   // 120 words of actual frame row data

    // Derived from user params
    rows_per_tensor_ = frames_per_tensor_.get() * FRAME_HEIGHT; // FRAME_HEIGHT rows per frame
    expected_source_mask_value_ = expected_source_mask_.get() & 0xFFU;
    expected_source_count_ = count_sources_in_mask(expected_source_mask_value_);
    if (expected_source_count_ == 0) {
      HOLOSCAN_LOG_ERROR("expected_source_mask must include at least one source ID.");
      exit(1);
    }
    expected_rows_per_tensor_ = frames_per_tensor_.get() * expected_source_count_ * ROWS_PER_SOURCE;
    header_batch_packets_value_ = header_batch_packets_.get();
    header_batch_slots_value_ = header_batch_slots_.get();
    if (use_assembled_batching() && !hds_.get() &&
        (header_batch_packets_value_ == 0 || header_batch_slots_value_ == 0)) {
      HOLOSCAN_LOG_ERROR("header_batch_packets and header_batch_slots must both be greater than 0.");
      exit(1);
    }

    for (int n = 0; n < num_concurrent; n++) {
      CUDA_TRY(cudaMalloc(&full_batch_data_d_[n], rows_per_tensor_ * nom_payload_size_));

      if (!gpu_direct_.get()) {
        CUDA_TRY(cudaMallocHost(&full_batch_data_h_[n], rows_per_tensor_ * nom_payload_size_));
      } else {
        CUDA_TRY(cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * rows_per_tensor_));
        CUDA_TRY(cudaMalloc(&d_dev_ptrs_[n], sizeof(void*) * rows_per_tensor_));
        CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&h_packet_placements_[n]),
                                sizeof(PacketPlacement) * rows_per_tensor_));
        CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_packet_placements_[n]),
                            sizeof(PacketPlacement) * rows_per_tensor_));
      }
      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
    }

    if (use_assembled_batching() && !hds_.get()) {
      header_slots_.resize(header_batch_slots_value_);
      for (auto& slot : header_slots_) {
        CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&slot.h_dev_ptrs),
                                sizeof(void*) * header_batch_packets_value_));
        CUDA_TRY(cudaMalloc(&slot.d_dev_ptrs, sizeof(void*) * header_batch_packets_value_));
        CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&slot.headers_h),
                                sizeof(PacketHeaderInfo) * header_batch_packets_value_));
        CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&slot.headers_d),
                            sizeof(PacketHeaderInfo) * header_batch_packets_value_));
        CUDA_TRY(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
        CUDA_TRY(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
        slot.holders.reserve(64);
        slot.packet_holder_indices.reserve(header_batch_packets_value_);
      }
    }

    current_batch_occupied_.assign(rows_per_tensor_, 0);
    emit_cell_generation_.assign(rows_per_tensor_, 0);
    pending_packets_.reserve(rows_per_tensor_ + batch_close_slack_packets_.get() + 4096);
    pending_holders_.reserve(256);

    if (hds_.get()) { assert(gpu_direct_.get()); }

    initialize_packet_debug();

    if (use_assembled_batching()) {
      if (hds_.get()) {
        HOLOSCAN_LOG_INFO(
            "StemReceiverOp::initialize() using header-aware batch assembly with CPU header-data "
            "split, {} slack packets, expected source mask 0x{:02x} ({} sources, {} expected "
            "rows/tensor).",
            batch_close_slack_packets_.get(),
            expected_source_mask_value_,
            expected_source_count_,
            expected_rows_per_tensor_);
      } else {
        HOLOSCAN_LOG_INFO(
            "StemReceiverOp::initialize() using header-aware batch assembly with {} slack packets, "
            "expected source mask 0x{:02x} ({} sources, {} expected rows/tensor), "
            "{} async header slots of {} packets each.",
            batch_close_slack_packets_.get(),
            expected_source_mask_value_,
            expected_source_count_,
            expected_rows_per_tensor_,
            header_batch_slots_value_,
            header_batch_packets_value_);
      }
    } else {
      HOLOSCAN_LOG_INFO(
          "StemReceiverOp::initialize() using legacy arrival-count batching for this configuration.");
    }

    HOLOSCAN_LOG_INFO("StemReceiverOp::initialize() complete. Batching {} frames ({} rows) per tensor.", frames_per_tensor_.get(), rows_per_tensor_);
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("StemReceiverOp::freeResources() start");
    cleanup_packet_debug();
    for (auto& slot : header_slots_) {
      if (slot.in_flight && slot.event) { cudaEventSynchronize(slot.event); }
      reset_header_slot(slot);
      if (slot.h_dev_ptrs) { cudaFreeHost(slot.h_dev_ptrs); }
      if (slot.d_dev_ptrs) { cudaFree(slot.d_dev_ptrs); }
      if (slot.headers_h) { cudaFreeHost(slot.headers_h); }
      if (slot.headers_d) { cudaFree(slot.headers_d); }
      if (slot.stream) { cudaStreamDestroy(slot.stream); }
      if (slot.event) { cudaEventDestroy(slot.event); }
    }
    header_slots_.clear();
    header_inflight_order_.clear();
    header_fill_slot_idx_ = -1;
    while (!assembled_batch_q_.empty()) {
      if (assembled_batch_q_.front().evt) { cudaEventSynchronize(assembled_batch_q_.front().evt); }
      assembled_batch_q_.pop();
    }
    pending_packets_.clear();
    pending_holders_.clear();
    free_pending_holder_indices_.clear();
    for (int n = 0; n < num_concurrent; n++) {
      if (full_batch_data_d_[n]) { cudaFree(full_batch_data_d_[n]); }
      if (full_batch_data_h_[n]) { cudaFreeHost(full_batch_data_h_[n]); }
      if (h_dev_ptrs_[n]) { cudaFreeHost(h_dev_ptrs_[n]); }
      if (d_dev_ptrs_[n]) { cudaFree(d_dev_ptrs_[n]); }
      if (h_packet_placements_[n]) { cudaFreeHost(h_packet_placements_[n]); }
      if (d_packet_placements_[n]) { cudaFree(d_packet_placements_[n]); }
      if (streams_[n]) { cudaStreamDestroy(streams_[n]); }
      if (events_[n]) { cudaEventDestroy(events_[n]); }
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("output")
        .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", 4UL));
    spec.param<std::shared_ptr<holoscan::Allocator>>(allocator_, "allocator", "Allocator", "Allocator for output tensors.");
    spec.param<std::string>(interface_name_, "interface_name", "Port name", "Name of the port to poll on", "rx_port");
    spec.param<bool>(hds_, "split_boundary", "Header-data split boundary", "Byte boundary where header and data is split", false);
    spec.param<bool>(gpu_direct_, "gpu_direct", "GPUDirect enabled", "GPUDirect", false);
    
    // Configurable number of frames packaged together
    spec.param<uint32_t>(frames_per_tensor_, "frames_per_tensor", "Frames per tensor", "Number of 128-row frames to receive per batch.", 10);
    // Left for backwards yaml compat, but unused internally for memory mgmt
    spec.param<uint32_t>(batch_size_, "batch_size", "Legacy Batch size", "Legacy batch size. Ignored in favor of frames_per_tensor.", 1000);
    
    spec.param<uint16_t>(max_packet_size_, "max_packet_size", "Max packet size", "Maximum UDP packet size. Must accommodate headers+payload.", 9100);
    spec.param<uint16_t>(header_size_, "header_size", "Header size", "Header size to strip (ETH+IP+UDP)", 42);
    spec.param<bool>(reorder_kernel_, "reorder_kernel", "Reorder kernel enabled", "Enable reorder kernel", true);
    spec.param<uint64_t>(count_, "count", "Count", "Number of frames to receive. 0 means infinite.", 0UL);
    spec.param<bool>(packet_debug_,
                     "packet_debug",
                     "Packet Debug",
                     "If true, capture packet-level debug summaries to CSV/text files.",
                     false);
    spec.param<std::string>(packet_debug_output_prefix_,
                            "packet_debug_output_prefix",
                            "Packet Debug Output Prefix",
                            "Prefix for receiver packet-debug outputs. Defaults to /tmp/<name>_<interface>_packet_debug.",
                            std::string(""));
    spec.param<uint32_t>(packet_debug_max_batches_,
                         "packet_debug_max_batches",
                         "Packet Debug Max Batches",
                         "Maximum number of full tensors to summarize. 0 means unlimited.",
                         0U);
    spec.param<uint32_t>(batch_close_slack_packets_,
                         "batch_close_slack_packets",
                         "Batch Close Slack Packets",
                         "Future-batch packet count used to close a partially missing batch.",
                         512U);
    spec.param<uint32_t>(expected_source_mask_,
                         "expected_source_mask",
                         "Expected Source Mask",
                         "Bit mask of eth source IDs expected to contribute rows. Default 255 expects IDs 0-7.",
                         0xFFU);
    spec.param<uint32_t>(incomplete_batch_log_interval_,
                         "incomplete_batch_log_interval",
                         "Incomplete Batch Log Interval",
                         "Emit one incomplete-batch summary every N incomplete batches. 0 disables summaries.",
                         100U);
    spec.param<uint32_t>(incomplete_batch_warn_missing_threshold_,
                         "incomplete_batch_warn_missing_threshold",
                         "Incomplete Batch Warning Threshold",
                         "Warn immediately when an incomplete batch is missing at least this many expected rows.",
                         1024U);
    spec.param<uint32_t>(header_batch_packets_,
                         "header_batch_packets",
                         "Header Batch Packets",
                         "Number of packet headers to extract per asynchronous header-transfer chunk.",
                         8192U);
    spec.param<uint32_t>(header_batch_slots_,
                         "header_batch_slots",
                         "Header Batch Slots",
                         "Number of asynchronous header-transfer chunks that can be in flight or filling.",
                         4U);
    spec.param<std::shared_ptr<holoscan::BooleanCondition>>(stop_condition_,
                                                            "stop_condition",
                                                            "Stop Condition",
                                                            "Boolean condition to stop execution.");
  }

  // Free buffers if CUDA processing/copy is complete
  void free_processed_packets() {
    // Iterate through the batches tracked for processing
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      // If CUDA processing/copy is complete, free the packets for all bursts in this batch
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        for (auto m = 0; m < batch.num_bursts; m++) {
          free_all_packets_and_burst_rx(batch.bursts[m]);
        }
        batch_q_.pop();
      } else {
        // No need to check the next batch if the previous one is still being processed
        break;
      }
    }
  }

  void release_batch_bursts(BatchAggregationParams& batch) {
    for (int m = 0; m < batch.num_bursts; m++) {
      if (batch.bursts[m]) {
        free_all_packets_and_burst_rx(batch.bursts[m]);
        batch.bursts[m] = nullptr;
      }
    }
    batch.num_bursts = 0;
  }

  void drop_current_batch(const char* reason) {
    profiling::ScopedRange drop_range("receiver/drop-batch", profiling::color::kBackpressure);

    int64_t freed_packets = 0;
    for (int m = 0; m < cur_batch_.num_bursts; m++) {
      if (cur_batch_.bursts[m]) {
        freed_packets += get_num_packets(cur_batch_.bursts[m]);
      }
    }

    release_batch_bursts(cur_batch_);
    ttl_packets_dropped_ += rows_per_tensor_;

    HOLOSCAN_LOG_ERROR(
        "{} Dropped one logical batch ({} packets). Freed {} fully-owned packets immediately.",
        reason,
        rows_per_tensor_,
        freed_packets);
  }

  void initialize_packet_debug() {
    if (!packet_debug_.get()) { return; }

    if (!gpu_direct_.get()) {
      HOLOSCAN_LOG_WARN("Packet debug requested for {} but requires gpu_direct: true. Disabling packet debug.",
                        name());
      return;
    }
    if (hds_.get()) {
      HOLOSCAN_LOG_WARN(
          "Packet debug requested for {} but does not yet support split_boundary: true. Disabling packet debug.",
          name());
      return;
    }

    packet_debug_summary_capacity_ = rows_per_tensor_;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&packet_debug_summaries_d_),
                        sizeof(PacketDebugSummary) * packet_debug_summary_capacity_));
    CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&packet_debug_summaries_h_),
                            sizeof(PacketDebugSummary) * packet_debug_summary_capacity_));

    packet_debug_column_histogram_.assign(FRAME_WIDTH, 0);
    packet_debug_output_prefix_resolved_ = packet_debug_output_prefix_.get().empty()
        ? (std::string("/tmp/") + name() + "_" + interface_name_.get() + "_packet_debug")
        : packet_debug_output_prefix_.get();
    packet_debug_csv_path_ = packet_debug_output_prefix_resolved_ + ".csv";
    packet_debug_summary_path_ = packet_debug_output_prefix_resolved_ + ".summary.txt";
    packet_debug_histogram_path_ = packet_debug_output_prefix_resolved_ + ".columns.csv";

    packet_debug_csv_.open(packet_debug_csv_path_, std::ios::out | std::ios::trunc);
    if (!packet_debug_csv_.is_open()) {
      HOLOSCAN_LOG_ERROR("Failed to open packet debug CSV output '{}'. Disabling packet debug.",
                         packet_debug_csv_path_);
      cleanup_packet_debug();
      return;
    }

    packet_debug_csv_
        << "receiver,interface,batch_index,packet_index,row_number,eth_id,frame_index,global_row,"
           "nonzero_count,first_nonzero_col,first_nonzero_value,second_nonzero_col,"
           "second_nonzero_value,max_value_col,max_value\n";
    packet_debug_csv_.flush();

    packet_debug_enabled_ = true;
    HOLOSCAN_LOG_INFO("Packet debug enabled for {}. Writing packet records to '{}' (max batches: {}).",
                      name(),
                      packet_debug_csv_path_,
                      packet_debug_max_batches_.get());
  }

  void cleanup_packet_debug() {
    if (packet_debug_csv_.is_open()) {
      packet_debug_csv_.flush();
      packet_debug_csv_.close();
    }
    if (packet_debug_summaries_d_) {
      cudaFree(packet_debug_summaries_d_);
      packet_debug_summaries_d_ = nullptr;
    }
    if (packet_debug_summaries_h_) {
      cudaFreeHost(packet_debug_summaries_h_);
      packet_debug_summaries_h_ = nullptr;
    }
    packet_debug_summary_capacity_ = 0;
    packet_debug_enabled_ = false;
  }

  void flush_packet_debug_outputs() {
    if (!packet_debug_enabled_ && packet_debug_batches_captured_ == 0) { return; }

    if (packet_debug_csv_.is_open()) { packet_debug_csv_.flush(); }

    if (!packet_debug_summary_path_.empty()) {
      std::ofstream summary_stream(packet_debug_summary_path_, std::ios::out | std::ios::trunc);
      if (summary_stream.is_open()) {
        summary_stream << "receiver=" << name() << "\n";
        summary_stream << "interface=" << interface_name_.get() << "\n";
        summary_stream << "batches_captured=" << packet_debug_batches_captured_ << "\n";
        summary_stream << "packets_captured=" << packet_debug_packets_captured_ << "\n";
        summary_stream << "invalid_source_id_packets=" << packet_debug_invalid_source_packets_ << "\n";
        summary_stream << "nonzero_count_histogram:\n";
        summary_stream << "  0=" << packet_debug_nonzero_histogram_[0] << "\n";
        summary_stream << "  1=" << packet_debug_nonzero_histogram_[1] << "\n";
        summary_stream << "  2=" << packet_debug_nonzero_histogram_[2] << "\n";
        summary_stream << "  >=3=" << packet_debug_nonzero_histogram_[3] << "\n";
        summary_stream << "source_id_counts:\n";
        for (size_t source_id = 0; source_id < packet_debug_source_histogram_.size(); ++source_id) {
          summary_stream << "  " << source_id << "=" << packet_debug_source_histogram_[source_id] << "\n";
        }
      }
    }

    if (!packet_debug_histogram_path_.empty()) {
      std::ofstream histogram_stream(packet_debug_histogram_path_, std::ios::out | std::ios::trunc);
      if (histogram_stream.is_open()) {
        histogram_stream << "column,count\n";
        for (size_t col = 0; col < packet_debug_column_histogram_.size(); ++col) {
          if (packet_debug_column_histogram_[col] == 0) { continue; }
          histogram_stream << col << "," << packet_debug_column_histogram_[col] << "\n";
        }
      }
    }
  }

  bool should_capture_packet_debug() const {
    return packet_debug_enabled_ &&
           (packet_debug_max_batches_.get() == 0 ||
            packet_debug_batches_captured_ < packet_debug_max_batches_.get());
  }

  static int packet_debug_optional_value(uint16_t value) {
    return (value == std::numeric_limits<uint16_t>::max()) ? -1 : static_cast<int>(value);
  }

  void capture_packet_debug_batch(int slot_idx, uint32_t num_packets) {
    if (!should_capture_packet_debug()) { return; }
    if (num_packets == 0) { return; }
    if (num_packets > packet_debug_summary_capacity_) {
      HOLOSCAN_LOG_WARN("Packet debug requested {} summaries but capacity is {}. Skipping batch.",
                        num_packets,
                        packet_debug_summary_capacity_);
      return;
    }

    profiling::ScopedRange debug_range("receiver/packet-debug-summary", profiling::color::kCompute);

    summarize_packets(reinterpret_cast<uint8_t**>(d_dev_ptrs_[slot_idx]),
                      packet_debug_summaries_d_,
                      nom_payload_size_,
                      custom_header_size_,
                      num_packets,
                      streams_[slot_idx]);
    CUDA_TRY(cudaMemcpyAsync(packet_debug_summaries_h_,
                             packet_debug_summaries_d_,
                             sizeof(PacketDebugSummary) * num_packets,
                             cudaMemcpyDeviceToHost,
                             streams_[slot_idx]));
    CUDA_TRY(cudaStreamSynchronize(streams_[slot_idx]));

    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
      const auto& summary = packet_debug_summaries_h_[packet_idx];

      packet_debug_packets_captured_++;
      if (summary.source_id < packet_debug_source_histogram_.size()) {
        packet_debug_source_histogram_[summary.source_id]++;
      } else {
        packet_debug_invalid_source_packets_++;
      }

      const size_t nonzero_bucket = (summary.nonzero_count >= 3) ? 3 : summary.nonzero_count;
      packet_debug_nonzero_histogram_[nonzero_bucket]++;
      if (summary.first_nonzero_col != std::numeric_limits<uint16_t>::max()) {
        packet_debug_column_histogram_[summary.first_nonzero_col]++;
      }
      if (summary.second_nonzero_col != std::numeric_limits<uint16_t>::max()) {
        packet_debug_column_histogram_[summary.second_nonzero_col]++;
      }

      if (packet_debug_csv_.is_open()) {
        packet_debug_csv_ << name() << "," << interface_name_.get() << "," << packet_debug_batches_captured_
                          << "," << packet_idx << "," << summary.row_number << "," << summary.source_id
                          << "," << summary.frame_index << "," << summary.global_row << ","
                          << summary.nonzero_count << ","
                          << packet_debug_optional_value(summary.first_nonzero_col) << ","
                          << summary.first_nonzero_value << ","
                          << packet_debug_optional_value(summary.second_nonzero_col) << ","
                          << summary.second_nonzero_value << ","
                          << packet_debug_optional_value(summary.max_value_col) << ","
                          << summary.max_value << "\n";
      }
    }

    if (packet_debug_csv_.is_open()) { packet_debug_csv_.flush(); }

    packet_debug_batches_captured_++;
    flush_packet_debug_outputs();

    HOLOSCAN_LOG_INFO(
        "Packet debug captured batch {} for {} ({} packet summaries written to '{}').",
        packet_debug_batches_captured_,
        name(),
        num_packets,
        packet_debug_csv_path_);
    if (packet_debug_max_batches_.get() > 0 &&
        packet_debug_batches_captured_ >= packet_debug_max_batches_.get()) {
      HOLOSCAN_LOG_INFO("Packet debug reached max batches ({}) for {}. Further packet capture disabled.",
                        packet_debug_max_batches_.get(),
                        name());
    }
  }

  void capture_packet_debug_batch(int slot_idx) {
    capture_packet_debug_batch(slot_idx, rows_per_tensor_);
  }

  bool use_assembled_batching() const {
    return gpu_direct_.get() && reorder_kernel_.get();
  }

  static uint32_t count_sources_in_mask(uint32_t source_mask) {
    uint32_t count = 0;
    source_mask &= 0xFFU;
    for (uint32_t source_id = 0; source_id < 8; ++source_id) {
      if ((source_mask & (1U << source_id)) != 0U) { count++; }
    }
    return count;
  }

  bool source_expected(uint16_t source_id) const {
    return source_id < 8 && (expected_source_mask_value_ & (1U << source_id)) != 0U;
  }

  uint32_t register_pending_holder(std::shared_ptr<BurstHolder> holder) {
    if (!holder) { return INVALID_HOLDER_INDEX; }

    if (!free_pending_holder_indices_.empty()) {
      const uint32_t index = free_pending_holder_indices_.back();
      free_pending_holder_indices_.pop_back();
      pending_holders_[index] = PendingHolder{std::move(holder), 0, 0};
      return index;
    }

    const uint32_t index = static_cast<uint32_t>(pending_holders_.size());
    pending_holders_.push_back(PendingHolder{std::move(holder), 0, 0});
    return index;
  }

  void retain_pending_holder(uint32_t holder_index) {
    if (holder_index == INVALID_HOLDER_INDEX || holder_index >= pending_holders_.size()) { return; }
    pending_holders_[holder_index].pending_packets++;
  }

  void free_pending_holder_slot(uint32_t holder_index) {
    if (holder_index == INVALID_HOLDER_INDEX || holder_index >= pending_holders_.size()) { return; }
    auto& pending_holder = pending_holders_[holder_index];
    if (!pending_holder.holder) { return; }

    pending_holder.holder.reset();
    pending_holder.pending_packets = 0;
    pending_holder.emit_generation = 0;
    free_pending_holder_indices_.push_back(holder_index);
  }

  void release_pending_holder(uint32_t holder_index) {
    if (holder_index == INVALID_HOLDER_INDEX || holder_index >= pending_holders_.size()) { return; }
    auto& pending_holder = pending_holders_[holder_index];
    if (!pending_holder.holder || pending_holder.pending_packets == 0) { return; }

    pending_holder.pending_packets--;
    if (pending_holder.pending_packets == 0) { free_pending_holder_slot(holder_index); }
  }

  void begin_emit_generation() {
    emit_generation_++;
    if (emit_generation_ != 0) { return; }

    std::fill(emit_cell_generation_.begin(), emit_cell_generation_.end(), 0);
    for (auto& pending_holder : pending_holders_) { pending_holder.emit_generation = 0; }
    emit_generation_ = 1;
  }

  void reset_header_slot(HeaderBatchSlot& slot) {
    slot.packet_count = 0;
    slot.in_flight = false;
    slot.holders.clear();
    slot.packet_holder_indices.clear();
  }

  int find_free_header_slot() const {
    for (size_t slot_idx = 0; slot_idx < header_slots_.size(); ++slot_idx) {
      const auto& slot = header_slots_[slot_idx];
      if (!slot.in_flight && slot.packet_count == 0) { return static_cast<int>(slot_idx); }
    }
    return -1;
  }

  bool launch_header_slot(int slot_idx) {
    auto& slot = header_slots_[slot_idx];
    if (slot.packet_count == 0) { return true; }

    {
      profiling::ScopedRange header_range("receiver/launch-header-batch", profiling::color::kCompute);
      if (CUDA_TRY(cudaMemcpyAsync(slot.d_dev_ptrs,
                                   slot.h_dev_ptrs,
                                   sizeof(void*) * slot.packet_count,
                                   cudaMemcpyHostToDevice,
                                   slot.stream)) != cudaSuccess) {
        return false;
      }
      extract_packet_headers(reinterpret_cast<uint8_t**>(slot.d_dev_ptrs),
                             slot.headers_d,
                             slot.packet_count,
                             slot.stream);
      if (CUDA_TRY(cudaMemcpyAsync(slot.headers_h,
                                   slot.headers_d,
                                   sizeof(PacketHeaderInfo) * slot.packet_count,
                                   cudaMemcpyDeviceToHost,
                                   slot.stream)) != cudaSuccess) {
        return false;
      }
      if (CUDA_TRY(cudaEventRecord(slot.event, slot.stream)) != cudaSuccess) { return false; }
    }

    slot.in_flight = true;
    header_inflight_order_.push_back(slot_idx);
    header_batches_launched_++;
    return true;
  }

  bool launch_header_fill_slot() {
    if (header_fill_slot_idx_ < 0) { return true; }

    const int slot_idx = header_fill_slot_idx_;
    header_fill_slot_idx_ = -1;
    if (!launch_header_slot(slot_idx)) {
      ttl_packets_dropped_ += header_slots_[slot_idx].packet_count;
      reset_header_slot(header_slots_[slot_idx]);
      return false;
    }
    return true;
  }

  bool process_header_slot(int slot_idx, OutputContext& op_output, ExecutionContext& context) {
    auto& slot = header_slots_[slot_idx];
    if (!slot.in_flight) { return false; }

    {
      profiling::ScopedRange process_range("receiver/process-header-batch", profiling::color::kReceiver);
      std::vector<uint32_t> holder_indices;
      holder_indices.reserve(slot.holders.size());
      for (const auto& holder : slot.holders) {
        holder_indices.push_back(register_pending_holder(holder));
      }

      for (uint32_t pkt_idx = 0; pkt_idx < slot.packet_count; ++pkt_idx) {
        const uint32_t holder_idx = slot.packet_holder_indices[pkt_idx];
        add_pending_packet(slot.h_dev_ptrs[pkt_idx], slot.headers_h[pkt_idx], holder_indices[holder_idx]);
      }

      for (const uint32_t holder_index : holder_indices) {
        if (holder_index != INVALID_HOLDER_INDEX &&
            holder_index < pending_holders_.size() &&
            pending_holders_[holder_index].pending_packets == 0) {
          free_pending_holder_slot(holder_index);
        }
      }
    }

    header_batches_completed_++;
    reset_header_slot(slot);
    try_emit_assembled_batches(op_output, context);
    return true;
  }

  void process_ready_header_slots(OutputContext& op_output, ExecutionContext& context) {
    while (!header_inflight_order_.empty()) {
      const int slot_idx = header_inflight_order_.front();
      auto& slot = header_slots_[slot_idx];

      const cudaError_t status = cudaEventQuery(slot.event);
      if (status == cudaSuccess) {
        header_inflight_order_.pop_front();
        process_header_slot(slot_idx, op_output, context);
      } else if (status != cudaErrorNotReady) {
        CUDA_TRY(status);
        break;
      } else {
        break;
      }
    }
  }

  int acquire_header_fill_slot(OutputContext& op_output, ExecutionContext& context) {
    if (header_fill_slot_idx_ >= 0) { return header_fill_slot_idx_; }

    process_ready_header_slots(op_output, context);

    int slot_idx = find_free_header_slot();
    if (slot_idx >= 0) {
      header_fill_slot_idx_ = slot_idx;
      return slot_idx;
    }

    header_slot_full_waits_++;
    if (header_slot_full_waits_ == 1 || header_slot_full_waits_ % 100 == 0) {
      HOLOSCAN_LOG_WARN(
          "All {} async header slots are in flight for {}. Waiting for one to complete "
          "(wait count: {}). Consider increasing header_batch_slots or reducing header_batch_packets.",
          header_batch_slots_value_,
          name(),
          header_slot_full_waits_);
    }

    if (!header_inflight_order_.empty()) {
      const int wait_idx = header_inflight_order_.front();
      auto& slot = header_slots_[wait_idx];
      CUDA_TRY(cudaEventSynchronize(slot.event));
      header_inflight_order_.pop_front();
      process_header_slot(wait_idx, op_output, context);
      header_fill_slot_idx_ = wait_idx;
      return header_fill_slot_idx_;
    }

    HOLOSCAN_LOG_ERROR("No available header slot found for {}.", name());
    return -1;
  }

  void free_processed_assembled_batches() {
    while (!assembled_batch_q_.empty()) {
      auto& batch = assembled_batch_q_.front();
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        assembled_batch_q_.pop();
      } else {
        break;
      }
    }
  }

  static int32_t source_id_to_global_row_host(uint32_t source_id, uint32_t row_offset) {
    if (source_id < 4) { return 511 - static_cast<int32_t>(row_offset * 4 + source_id); }
    if (source_id < 8) { return 512 + static_cast<int32_t>(row_offset * 4 + (source_id - 4)); }
    return -1;
  }

  static PacketHeaderInfo parse_packet_header_host(const uint8_t* src) {
    PacketHeaderInfo header{};
    header.source_id = 0xFFFF;
    header.global_row = -1;

    if (src != nullptr) {
      header.row_number = (static_cast<uint16_t>(src[5]) << 8) | static_cast<uint16_t>(src[4]);
      header.source_id = (static_cast<uint16_t>(src[7]) << 8) | static_cast<uint16_t>(src[6]);
      header.frame_index = header.row_number / ROWS_PER_SOURCE;
      header.row_offset = header.row_number % ROWS_PER_SOURCE;
      header.global_row = static_cast<int16_t>(
          source_id_to_global_row_host(header.source_id, header.row_offset));
    }

    return header;
  }

  int64_t unwrap_frame_index(uint32_t frame_idx) {
    const int64_t ref_cycle = static_cast<int64_t>(frame_unwrap_ref_ / 128);
    int64_t best_frame = static_cast<int64_t>(frame_idx) + ref_cycle * 128;
    int64_t best_dist = std::llabs(best_frame - static_cast<int64_t>(frame_unwrap_ref_));

    for (int64_t delta = -1; delta <= 1; ++delta) {
      const int64_t cycle = ref_cycle + delta;
      const int64_t candidate = static_cast<int64_t>(frame_idx) + cycle * 128;
      const int64_t dist = std::llabs(candidate - static_cast<int64_t>(frame_unwrap_ref_));
      if (dist < best_dist || (dist == best_dist && candidate <= static_cast<int64_t>(frame_unwrap_ref_))) {
        best_frame = candidate;
        best_dist = dist;
      }
    }

    if (best_frame >= 0 && best_frame > static_cast<int64_t>(frame_unwrap_ref_)) {
      frame_unwrap_ref_ = static_cast<uint64_t>(best_frame);
    }

    return best_frame;
  }

  void rebuild_current_batch_state() {
    std::fill(current_batch_occupied_.begin(), current_batch_occupied_.end(), 0);
    current_batch_unique_packets_ = 0;
    future_packet_count_ = 0;

    const uint64_t batch_end = current_batch_start_abs_frame_ + frames_per_tensor_.get();
    for (const auto& entry : pending_packets_) {
      if (entry.abs_frame < current_batch_start_abs_frame_) { continue; }
      if (entry.abs_frame >= batch_end) {
        future_packet_count_++;
        continue;
      }

      const uint64_t relative_frame = entry.abs_frame - current_batch_start_abs_frame_;
      const uint64_t cell = relative_frame * FRAME_HEIGHT + static_cast<uint64_t>(entry.global_row);
      if (cell < current_batch_occupied_.size() && !current_batch_occupied_[cell]) {
        current_batch_occupied_[cell] = 1;
        current_batch_unique_packets_++;
      }
    }
  }

  void add_pending_packet(void* packet_ptr,
                          const PacketHeaderInfo& header,
                          uint32_t holder_index) {
    if (header.source_id >= 8 || header.global_row < 0) {
      ttl_packets_dropped_++;
      return;
    }
    if (!source_expected(header.source_id)) {
      ttl_packets_ignored_unexpected_source_++;
      return;
    }

    if (!stream_synced_) {
      if (header.row_number != 0) {
        ttl_packets_dropped_++;
        return;
      }

      stream_synced_ = true;
      frame_unwrap_ref_ = 0;
      current_batch_start_abs_frame_ = 0;
      HOLOSCAN_LOG_INFO(
          "StemReceiverOp {} synchronized stream on row_number=0, source_id={}.",
          name(),
          header.source_id);
    }

    const int64_t abs_frame_signed = unwrap_frame_index(header.frame_index);
    if (abs_frame_signed < 0 ||
        static_cast<uint64_t>(abs_frame_signed) < current_batch_start_abs_frame_) {
      ttl_packets_dropped_++;
      return;
    }
    const uint64_t abs_frame = static_cast<uint64_t>(abs_frame_signed);

    PacketEntry entry;
    entry.packet_ptr = packet_ptr;
    entry.abs_frame = abs_frame;
    entry.global_row = header.global_row;
    entry.row_number = header.row_number;
    entry.source_id = header.source_id;
    entry.holder_index = holder_index;
    retain_pending_holder(holder_index);
    pending_packets_.push_back(std::move(entry));

    const uint64_t batch_end = current_batch_start_abs_frame_ + frames_per_tensor_.get();
    if (abs_frame >= batch_end) {
      future_packet_count_++;
      return;
    }

    const uint64_t relative_frame = abs_frame - current_batch_start_abs_frame_;
    const uint64_t cell = relative_frame * FRAME_HEIGHT + static_cast<uint64_t>(header.global_row);
    if (cell < current_batch_occupied_.size() && !current_batch_occupied_[cell]) {
      current_batch_occupied_[cell] = 1;
      current_batch_unique_packets_++;
    }
  }

  bool should_close_current_assembled_batch() const {
    if (!stream_synced_) { return false; }
    if (current_batch_unique_packets_ >= expected_rows_per_tensor_) { return true; }
    return current_batch_unique_packets_ > 0 &&
           future_packet_count_ >= batch_close_slack_packets_.get();
  }

  bool emit_current_assembled_batch(OutputContext& op_output, ExecutionContext& context) {
    profiling::ScopedRange emit_batch_range("receiver/emit-assembled-batch", profiling::color::kReceiver);

    free_processed_assembled_batches();
    if (assembled_batch_q_.size() >= num_concurrent) {
      HOLOSCAN_LOG_ERROR("Fell behind. All assembled batch buffers queued; holding pending packets.");
      return false;
    }

    const uint64_t batch_end = current_batch_start_abs_frame_ + frames_per_tensor_.get();
    std::vector<std::shared_ptr<BurstHolder>> batch_holders;
    begin_emit_generation();

    uint32_t packets_to_gather = 0;
    for (const auto& entry : pending_packets_) {
      if (entry.abs_frame < current_batch_start_abs_frame_ || entry.abs_frame >= batch_end) {
        continue;
      }

      const uint64_t relative_frame = entry.abs_frame - current_batch_start_abs_frame_;
      const uint64_t cell = relative_frame * FRAME_HEIGHT + static_cast<uint64_t>(entry.global_row);
      if (cell >= emit_cell_generation_.size() || emit_cell_generation_[cell] == emit_generation_) {
        continue;
      }

      emit_cell_generation_[cell] = emit_generation_;
      h_dev_ptrs_[assembled_cur_batch_idx_][packets_to_gather] = entry.packet_ptr;
      h_packet_placements_[assembled_cur_batch_idx_][packets_to_gather] = PacketPlacement{
          static_cast<uint16_t>(relative_frame), entry.global_row, 1};

      if (entry.holder_index != INVALID_HOLDER_INDEX &&
          entry.holder_index < pending_holders_.size()) {
        auto& pending_holder = pending_holders_[entry.holder_index];
        if (pending_holder.holder && pending_holder.emit_generation != emit_generation_) {
          pending_holder.emit_generation = emit_generation_;
          batch_holders.push_back(pending_holder.holder);
        }
      }
      packets_to_gather++;
    }

    const uint32_t missing_packets =
        (packets_to_gather >= expected_rows_per_tensor_) ? 0 : expected_rows_per_tensor_ - packets_to_gather;
    if (missing_packets > 0) {
      incomplete_batches_emitted_++;
      incomplete_rows_missing_total_ += missing_packets;
      incomplete_rows_missing_max_ = std::max<uint32_t>(incomplete_rows_missing_max_, missing_packets);

      const bool warn_now = missing_packets >= incomplete_batch_warn_missing_threshold_.get();
      const bool summarize_now = incomplete_batch_log_interval_.get() > 0 &&
                                 incomplete_batches_emitted_ % incomplete_batch_log_interval_.get() == 0;
      if (warn_now) {
        HOLOSCAN_LOG_WARN(
            "Emitting incomplete batch starting at absolute frame {}: {} / {} expected rows present, "
            "{} expected rows missing.",
            current_batch_start_abs_frame_,
            packets_to_gather,
            expected_rows_per_tensor_,
            missing_packets);
      } else if (summarize_now) {
        const double avg_missing =
            static_cast<double>(incomplete_rows_missing_total_) /
            static_cast<double>(incomplete_batches_emitted_);
        HOLOSCAN_LOG_INFO(
            "Incomplete batch summary for {}: {} incomplete batches, avg {:.1f} missing rows, "
            "max {} missing rows. Latest batch started at absolute frame {} with {} missing rows.",
            name(),
            incomplete_batches_emitted_,
            avg_missing,
            incomplete_rows_missing_max_,
            current_batch_start_abs_frame_,
            missing_packets);
      }
    }

    CUDA_TRY(cudaMemsetAsync(full_batch_data_d_[assembled_cur_batch_idx_],
                             0,
                             rows_per_tensor_ * nom_payload_size_,
                             streams_[assembled_cur_batch_idx_]));

    if (packets_to_gather > 0) {
      {
        profiling::ScopedRange ptr_copy_range("receiver/pointer-list-h2d", profiling::color::kCopy);
        CUDA_TRY(cudaMemcpyAsync(d_dev_ptrs_[assembled_cur_batch_idx_],
                                 h_dev_ptrs_[assembled_cur_batch_idx_],
                                 sizeof(void*) * packets_to_gather,
                                 cudaMemcpyHostToDevice,
                                 streams_[assembled_cur_batch_idx_]));
        CUDA_TRY(cudaMemcpyAsync(d_packet_placements_[assembled_cur_batch_idx_],
                                 h_packet_placements_[assembled_cur_batch_idx_],
                                 sizeof(PacketPlacement) * packets_to_gather,
                                 cudaMemcpyHostToDevice,
                                 streams_[assembled_cur_batch_idx_]));
      }

      if (should_capture_packet_debug()) {
        capture_packet_debug_batch(assembled_cur_batch_idx_, packets_to_gather);
      }

      {
        profiling::ScopedRange gather_range("receiver/gather-packets", profiling::color::kCompute);
        const uint16_t gather_header_size = hds_.get() ? 0 : custom_header_size_;
        gather_packets_by_placement(reinterpret_cast<uint8_t**>(d_dev_ptrs_[assembled_cur_batch_idx_]),
                                    d_packet_placements_[assembled_cur_batch_idx_],
                                    static_cast<uint8_t*>(full_batch_data_d_[assembled_cur_batch_idx_]),
                                    nom_payload_size_,
                                    gather_header_size,
                                    packets_to_gather,
                                    rows_per_tensor_,
                                    streams_[assembled_cur_batch_idx_]);
      }
    }

    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
    auto allocator_handle =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    auto result = gxf_tensor->reshape<uint16_t>(
        nvidia::gxf::Shape{static_cast<int>(frames_per_tensor_.get()), FRAME_HEIGHT, FRAME_WIDTH},
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator_handle.value());
    if (!result) { throw std::runtime_error("Failed to reshape output tensor"); }

    {
      profiling::ScopedRange tensor_copy_range("receiver/batch-d2d-copy", profiling::color::kCopy);
      HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(gxf_tensor->pointer(),
                                         full_batch_data_d_[assembled_cur_batch_idx_],
                                         gxf_tensor->bytes_size(),
                                         cudaMemcpyDeviceToDevice,
                                         streams_[assembled_cur_batch_idx_]));
    }

    auto maybedl = gxf_tensor->toDLManagedTensorContext();
    auto holoscan_tensor = std::make_shared<Tensor>(maybedl.value());

    TensorMap out_message;
    out_message.insert({"frame", holoscan_tensor});
    {
      profiling::ScopedRange emit_range("receiver/emit", profiling::color::kIo);
      op_output.emit(out_message, "output");
    }

    cudaEventRecord(events_[assembled_cur_batch_idx_], streams_[assembled_cur_batch_idx_]);
    assembled_batch_q_.push(AssembledBatch{
        std::move(batch_holders), events_[assembled_cur_batch_idx_], packets_to_gather, missing_packets});

    auto write_it = pending_packets_.begin();
    for (auto read_it = pending_packets_.begin(); read_it != pending_packets_.end(); ++read_it) {
      if (read_it->abs_frame < batch_end) {
        release_pending_holder(read_it->holder_index);
        continue;
      }

      if (write_it != read_it) { *write_it = std::move(*read_it); }
      ++write_it;
    }
    pending_packets_.erase(write_it, pending_packets_.end());

    total_frames_emitted_ += frames_per_tensor_.get();
    current_batch_start_abs_frame_ += frames_per_tensor_.get();
    assembled_cur_batch_idx_ = (++assembled_cur_batch_idx_ % num_concurrent);
    rebuild_current_batch_state();

    if (count_.get() > 0 && total_frames_emitted_ >= count_.get()) {
      HOLOSCAN_LOG_INFO("StemReceiverOp: Reached frame limit of {}", count_.get());
      is_done_ = true;
      stop_condition_.get()->disable_tick();
    }

    return true;
  }

  void try_emit_assembled_batches(OutputContext& op_output, ExecutionContext& context) {
    while (!is_done_ && should_close_current_assembled_batch()) {
      if (!emit_current_assembled_batch(op_output, context)) { break; }
    }
  }

  void process_assembled_burst(BurstParams* burst,
                               OutputContext& op_output,
                               ExecutionContext& context) {
    const auto burst_size = static_cast<uint32_t>(get_num_packets(burst));
    if (burst_size == 0) {
      free_all_packets_and_burst_rx(burst);
      return;
    }

    if (hds_.get()) {
      if (burst->hdr.hdr.num_segs < 2) {
        HOLOSCAN_LOG_ERROR(
            "split_boundary is enabled for {}, but received a burst with only {} segment(s). "
            "Check that the RX queue has CPU header and GPU payload memory regions.",
            name(),
            burst->hdr.hdr.num_segs);
        ttl_packets_dropped_ += burst_size;
        free_all_packets_and_burst_rx(burst);
        return;
      }

      auto holder = std::make_shared<BurstHolder>(burst);
      const uint32_t holder_index = register_pending_holder(holder);
      {
        profiling::ScopedRange parse_range("receiver/parse-hds-headers", profiling::color::kReceiver);
        for (uint32_t pkt_idx = 0; pkt_idx < burst_size; ++pkt_idx) {
          const auto* custom_header =
              reinterpret_cast<const uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_idx)) +
              header_size_.get();
          auto* payload = get_segment_packet_ptr(burst, 1, pkt_idx);
          add_pending_packet(payload, parse_packet_header_host(custom_header), holder_index);
        }
      }
      if (holder_index != INVALID_HOLDER_INDEX &&
          holder_index < pending_holders_.size() &&
          pending_holders_[holder_index].pending_packets == 0) {
        free_pending_holder_slot(holder_index);
      }

      try_emit_assembled_batches(op_output, context);
      return;
    }

    auto holder = std::make_shared<BurstHolder>(burst);
    uint32_t pkt_idx = 0;
    while (pkt_idx < burst_size) {
      const int slot_idx = acquire_header_fill_slot(op_output, context);
      if (is_done_) { return; }
      if (slot_idx < 0) {
        ttl_packets_dropped_ += (burst_size - pkt_idx);
        return;
      }

      auto& slot = header_slots_[slot_idx];
      const uint32_t available = header_batch_packets_value_ - slot.packet_count;
      if (available == 0) {
        if (!launch_header_fill_slot()) {
          ttl_packets_dropped_ += (burst_size - pkt_idx);
          return;
        }
        continue;
      }

      const uint32_t take = std::min<uint32_t>(burst_size - pkt_idx, available);
      const uint32_t holder_index = static_cast<uint32_t>(slot.holders.size());
      slot.holders.push_back(holder);

      for (uint32_t i = 0; i < take; ++i) {
        slot.h_dev_ptrs[slot.packet_count] =
            reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_idx + i)) +
            header_size_.get();
        slot.packet_holder_indices.push_back(holder_index);
        slot.packet_count++;
      }

      pkt_idx += take;
      if (slot.packet_count == header_batch_packets_value_ && !launch_header_fill_slot()) {
        ttl_packets_dropped_ += (burst_size - pkt_idx);
        return;
      }
    }

    process_ready_header_slots(op_output, context);
  }

  void compute_assembled(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    (void)op_input;

    {
      profiling::ScopedRange reclaim_range("receiver/reclaim-completed", profiling::color::kIo);
      free_processed_assembled_batches();
    }
    process_ready_header_slots(op_output, context);

    if (is_done_) return;

    BurstParams* burst = nullptr;
    bool received_burst = false;
    const auto num_rx_queues = get_num_rx_queues(port_id_);
    for (int q = 0; q < num_rx_queues; q++) {
      auto status = Status::SUCCESS;
      {
        profiling::ScopedRange rx_burst_range("receiver/get-rx-burst", profiling::color::kIo);
        status = get_rx_burst(&burst, port_id_, q);
      }
      if (status != Status::SUCCESS) { continue; }

      const auto burst_size = get_num_packets(burst);
      ttl_pkts_recv_ += burst_size;
      received_burst = true;
      process_assembled_burst(burst, op_output, context);
      if (is_done_) { return; }
    }

    if (!received_burst) {
      // If traffic pauses, do not leave a partial header chunk stranded forever.
      launch_header_fill_slot();
    }
    process_ready_header_slots(op_output, context);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    profiling::ScopedRange compute_range("receiver/compute", profiling::color::kReceiver);

    if (use_assembled_batching()) {
      compute_assembled(op_input, op_output, context);
      return;
    }

    // If we processed a batch of packets in a previous compute call, that was done asynchronously,
    // and we'll need to free the packets eventually so the NIC can have space for the next bursts.
    // Ideally, we'd free the packets on a callback from CUDA, but that is slow. For that reason and
    // to keep it simple, we do that check right here on the next epoch of the operator.
    {
      profiling::ScopedRange reclaim_range("receiver/reclaim-completed", profiling::color::kIo);
      free_processed_packets();
    }

    if (is_done_) return;

    BurstParams *burst;

    // In this example, we'll loop through all the rx queues of the interface
    // assuming we want to process the packets the same way for all queues
    const auto num_rx_queues = get_num_rx_queues(port_id_);

    for (int q = 0; q < num_rx_queues; q++) {
      auto status = Status::SUCCESS;
      {
        profiling::ScopedRange rx_burst_range("receiver/get-rx-burst", profiling::color::kIo);
        status = get_rx_burst(&burst, port_id_, q);
      }
      if (status != Status::SUCCESS) { continue; }

      auto burst_size = get_num_packets(burst);

      // Count packets received
      ttl_pkts_recv_ += burst_size;

      int p = 0;
      while (p < burst_size) {
        int space_in_batch = rows_per_tensor_ - aggr_pkts_recv_;
        int packets_to_copy = std::min(static_cast<int>(burst_size - p), space_in_batch);

        {
          profiling::ScopedRange aggregate_range("receiver/aggregate-burst", profiling::color::kReceiver);
          for (int i = 0; i < packets_to_copy; i++) {
            int pkt_in_burst = p + i;
            
            if (gpu_direct_.get()) {
              if (hds_.get()) {
                h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + i] = burst->pkts[1][pkt_in_burst];
              } else {
                h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + i] =
                    reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_in_burst)) +
                    header_size_.get();
              }
            } else {
              // CPU fallback (Warning: naive copy, doesn't reorder within frame)
              auto payload_ptr =
                  reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_in_burst)) +
                  header_size_.get();
              auto burst_offset = (aggr_pkts_recv_ + i) * nom_payload_size_;

              // We strip the 64-byte custom header for the Host copy because no reorder kernel is run for it
              memcpy((char*)full_batch_data_h_[cur_batch_idx_] + burst_offset,
                     payload_ptr + custom_header_size_,
                     nom_payload_size_);
            }
          }
        }
        
        aggr_pkts_recv_ += packets_to_copy;
        p += packets_to_copy;

        // If burst was fully consumed this iteration, the current batch claims ownership.
        if (p == burst_size) {
          if (cur_batch_.num_bursts < MAX_BURSTS_PER_BATCH) {
            cur_batch_.bursts[cur_batch_.num_bursts++] = burst;
          } else {
            HOLOSCAN_LOG_ERROR("Exceeded MAX_BURSTS_PER_BATCH ({}). Memory may leak.", MAX_BURSTS_PER_BATCH);
          }
        }

        // If we have aggregated the required number of rows, process and emit
        if (aggr_pkts_recv_ == rows_per_tensor_) {
          const bool capture_packet_debug = should_capture_packet_debug();
          aggr_pkts_recv_ = 0; // reset for next tensor

          {
            profiling::ScopedRange reclaim_range("receiver/reclaim-before-queue-check", profiling::color::kIo);
            free_processed_packets();
          }
          if (batch_q_.size() >= num_concurrent) {
            drop_current_batch("Fell behind. All buffers queued.");
            continue;
          }

          if (gpu_direct_.get()) {
          // GPUDirect mode: we copy the payload (referenced in h_dev_ptrs_)
          // to a contiguous memory buffer (full_batch_data_d_)
          // NOTE: there is no actual reordering since we use the same order as packets came in,
          //   but they would be reordered if h_dev_ptrs_ was filled based on packet sequence id.
          // We also allow disabling the reorder kernel if alignment and memory types are not
          // supported. Currently the reorder kernel expects the packets to be 16B-aligned, and
          // anything that's not will cause an access error on the GPU
            if (reorder_kernel_.get() || capture_packet_debug) {
              {
                profiling::ScopedRange ptr_copy_range("receiver/pointer-list-h2d", profiling::color::kCopy);
                CUDA_TRY(cudaMemcpyAsync(d_dev_ptrs_[cur_batch_idx_],
                                         h_dev_ptrs_[cur_batch_idx_],
                                         sizeof(void*) * rows_per_tensor_,
                                         cudaMemcpyHostToDevice,
                                         streams_[cur_batch_idx_]));
              }
            }

            if (capture_packet_debug) { capture_packet_debug_batch(cur_batch_idx_); }

            if (reorder_kernel_.get()) {
              uint64_t base_absolute_row = static_cast<uint64_t>(total_frames_emitted_) * static_cast<uint64_t>(FRAME_HEIGHT);

              // gather_packets translates out-of-order packets based on the 16-bit row ID
              {
                profiling::ScopedRange gather_range("receiver/gather-packets", profiling::color::kCompute);
                gather_packets(reinterpret_cast<uint8_t**>(d_dev_ptrs_[cur_batch_idx_]),
                               static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                               nom_payload_size_, custom_header_size_, rows_per_tensor_, rows_per_tensor_,
                               base_absolute_row,
                               streams_[cur_batch_idx_]);
              }
            } else {
              profiling::ScopedRange reorder_range("receiver/reorder-in-arrival-order", profiling::color::kCompute);
              simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                                    reinterpret_cast<const void* const*>(h_dev_ptrs_[cur_batch_idx_]),
                                    nom_payload_size_,
                                    rows_per_tensor_,
                                    streams_[cur_batch_idx_]);
            }
          } else {
            // Non GPUDirect mode: we copy the payload on host-pinned memory (in full_batch_data_h_)
            // to a contiguous memory buffer on the GPU (full_batch_data_d_)
            // NOTE: there is no reordering support here at all
            profiling::ScopedRange host_copy_range("receiver/host-batch-h2d", profiling::color::kCopy);
            CUDA_TRY(cudaMemcpyAsync(full_batch_data_d_[cur_batch_idx_],
                                     full_batch_data_h_[cur_batch_idx_],
                                     rows_per_tensor_ * nom_payload_size_,
                                     cudaMemcpyHostToDevice,
                                     streams_[cur_batch_idx_]));
          }

          // Build Tensor using GXF zero-copy memory wrapping
          auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
          auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
          
          auto result = gxf_tensor->reshape<uint16_t>(
              nvidia::gxf::Shape{static_cast<int>(frames_per_tensor_.get()), FRAME_HEIGHT, FRAME_WIDTH},
              nvidia::gxf::MemoryStorageType::kDevice,
              allocator_handle.value());
          if (!result) { throw std::runtime_error("Failed to reshape output tensor"); }
	        //HOLOSCAN_LOG_INFO("-> Reshaped tensor.");

          // Copy the aggregated frame data to the new tensor's buffer
          {
            profiling::ScopedRange tensor_copy_range("receiver/batch-d2d-copy", profiling::color::kCopy);
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(gxf_tensor->pointer(),
                            full_batch_data_d_[cur_batch_idx_],
                            gxf_tensor->bytes_size(),
                            cudaMemcpyDeviceToDevice,
                            streams_[cur_batch_idx_]));
          }

          auto maybedl = gxf_tensor->toDLManagedTensorContext();
          auto holoscan_tensor = std::make_shared<Tensor>(maybedl.value());
          
          TensorMap out_message;
          out_message.insert({"frame", holoscan_tensor});
          {
            profiling::ScopedRange emit_range("receiver/emit", profiling::color::kIo);
            op_output.emit(out_message, "output");
          }

          total_frames_emitted_ += frames_per_tensor_.get();

          cudaEventRecord(events_[cur_batch_idx_], streams_[cur_batch_idx_]);
          cur_batch_.evt = events_[cur_batch_idx_];
          batch_q_.push(cur_batch_);

          cur_batch_.num_bursts = 0;
          cur_batch_idx_ = (++cur_batch_idx_ % num_concurrent);

          if (count_.get() > 0 && total_frames_emitted_ >= count_.get()) {
            HOLOSCAN_LOG_INFO("StemReceiverOp: Reached frame limit of {}", count_.get());
            is_done_ = true;
            stop_condition_.get()->disable_tick();
            return;
          }
        }
      }
    }
  }

 private:
  int port_id_;
  BatchAggregationParams cur_batch_{};
  int cur_batch_idx_ = 0;
  std::queue<BatchAggregationParams> batch_q_;
  
  int64_t ttl_bytes_recv_ = 0;
  int64_t ttl_pkts_recv_ = 0;
  int64_t ttl_packets_dropped_ = 0;
  int64_t ttl_packets_ignored_unexpected_source_ = 0;

  int64_t aggr_pkts_recv_ = 0;
  uint64_t total_frames_emitted_ = 0;
  
  uint16_t custom_header_size_;  
  uint16_t nom_payload_size_;    
  uint32_t rows_per_tensor_;     
  uint32_t expected_rows_per_tensor_ = 0;
  uint32_t expected_source_mask_value_ = 0xFFU;
  uint32_t expected_source_count_ = 8;
  uint32_t header_batch_packets_value_ = 8192;
  uint32_t header_batch_slots_value_ = 4;

  std::array<void**, num_concurrent> h_dev_ptrs_;  // host pointers 
  std::array<void*, num_concurrent> d_dev_ptrs_;   // device pointers
  std::array<void*, num_concurrent> full_batch_data_d_;
  std::array<void*, num_concurrent> full_batch_data_h_;

  Parameter<std::string> interface_name_;
  Parameter<bool> hds_;
  Parameter<bool> gpu_direct_;
  Parameter<uint32_t> batch_size_;     
  Parameter<uint32_t> frames_per_tensor_;   
  Parameter<uint16_t> max_packet_size_;
  Parameter<uint16_t> header_size_;
  Parameter<bool> reorder_kernel_;
  Parameter<uint64_t> count_;
  Parameter<bool> packet_debug_;
  Parameter<std::string> packet_debug_output_prefix_;
  Parameter<uint32_t> packet_debug_max_batches_;
  Parameter<uint32_t> batch_close_slack_packets_;
  Parameter<uint32_t> expected_source_mask_;
  Parameter<uint32_t> incomplete_batch_log_interval_;
  Parameter<uint32_t> incomplete_batch_warn_missing_threshold_;
  Parameter<uint32_t> header_batch_packets_;
  Parameter<uint32_t> header_batch_slots_;
  Parameter<std::shared_ptr<holoscan::BooleanCondition>> stop_condition_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  bool is_done_ = false;

  bool packet_debug_enabled_ = false;
  uint32_t packet_debug_summary_capacity_ = 0;
  PacketDebugSummary* packet_debug_summaries_d_ = nullptr;
  PacketDebugSummary* packet_debug_summaries_h_ = nullptr;
  std::ofstream packet_debug_csv_;
  std::string packet_debug_output_prefix_resolved_;
  std::string packet_debug_csv_path_;
  std::string packet_debug_summary_path_;
  std::string packet_debug_histogram_path_;
  uint64_t packet_debug_batches_captured_ = 0;
  uint64_t packet_debug_packets_captured_ = 0;
  uint64_t packet_debug_invalid_source_packets_ = 0;
  std::array<uint64_t, 8> packet_debug_source_histogram_{};
  std::array<uint64_t, 4> packet_debug_nonzero_histogram_{};
  std::vector<uint64_t> packet_debug_column_histogram_{};

  std::vector<PacketEntry> pending_packets_;
  std::vector<PendingHolder> pending_holders_;
  std::vector<uint32_t> free_pending_holder_indices_;
  std::queue<AssembledBatch> assembled_batch_q_;
  std::vector<uint8_t> current_batch_occupied_;
  std::vector<uint32_t> emit_cell_generation_;
  uint64_t current_batch_start_abs_frame_ = 0;
  uint64_t frame_unwrap_ref_ = 0;
  uint32_t current_batch_unique_packets_ = 0;
  uint32_t future_packet_count_ = 0;
  uint32_t emit_generation_ = 0;
  uint64_t incomplete_batches_emitted_ = 0;
  uint64_t incomplete_rows_missing_total_ = 0;
  uint32_t incomplete_rows_missing_max_ = 0;
  uint64_t header_batches_launched_ = 0;
  uint64_t header_batches_completed_ = 0;
  uint64_t header_slot_full_waits_ = 0;
  int assembled_cur_batch_idx_ = 0;
  int header_fill_slot_idx_ = -1;
  bool stream_synced_ = false;

  std::vector<HeaderBatchSlot> header_slots_;
  std::deque<int> header_inflight_order_;
  std::array<PacketPlacement*, num_concurrent> h_packet_placements_{};
  std::array<PacketPlacement*, num_concurrent> d_packet_placements_{};

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
};

}  // namespace holoscan::ops
