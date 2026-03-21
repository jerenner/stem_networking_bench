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
#include "holoscan/holoscan.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/script.h>

#define BURST_ACCESS_METHOD_RAW_PTR 0
#define BURST_ACCESS_METHOD_DIRECT_ACCESS 1
#define BURST_ACCESS_METHOD BURST_ACCESS_METHOD_RAW_PTR

// New Frame Geometry
#define FRAME_WIDTH 3840
#define FRAME_HEIGHT 128
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
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StemReceiverOp)

  StemReceiverOp() = default;

  ~StemReceiverOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received and {} packets dropped",
                      ttl_bytes_recv_, ttl_pkts_recv_, ttl_packets_dropped_);
    HOLOSCAN_LOG_INFO("StemReceiverOp shutting down");
    freeResources();
  }

  void initialize() override {

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
    rows_per_tensor_ = frames_per_tensor_.get() * 128; // 128 rows per frame

    for (int n = 0; n < num_concurrent; n++) {
      CUDA_TRY(cudaMalloc(&full_batch_data_d_[n], rows_per_tensor_ * nom_payload_size_));

      if (!gpu_direct_.get()) {
        CUDA_TRY(cudaMallocHost(&full_batch_data_h_[n], rows_per_tensor_ * nom_payload_size_));
      } else {
        CUDA_TRY(cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * rows_per_tensor_));
        CUDA_TRY(cudaMalloc(&d_dev_ptrs_[n], sizeof(void*) * rows_per_tensor_));
      }
      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
    }

    if (hds_.get()) { assert(gpu_direct_.get()); }

    HOLOSCAN_LOG_INFO("StemReceiverOp::initialize() complete. Batching {} frames ({} rows) per tensor.", frames_per_tensor_.get(), rows_per_tensor_);
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("StemReceiverOp::freeResources() start");
    for (int n = 0; n < num_concurrent; n++) {
      if (full_batch_data_d_[n]) { cudaFree(full_batch_data_d_[n]); }
      if (full_batch_data_h_[n]) { cudaFreeHost(full_batch_data_h_[n]); }
      if (h_dev_ptrs_[n]) { cudaFreeHost(h_dev_ptrs_[n]); }
      if (d_dev_ptrs_[n]) { cudaFree(d_dev_ptrs_[n]); }
      if (streams_[n]) { cudaStreamDestroy(streams_[n]); }
      if (events_[n]) { cudaEventDestroy(events_[n]); }
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("output");
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

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    // If we processed a batch of packets in a previous compute call, that was done asynchronously,
    // and we'll need to free the packets eventually so the NIC can have space for the next bursts.
    // Ideally, we'd free the packets on a callback from CUDA, but that is slow. For that reason and
    // to keep it simple, we do that check right here on the next epoch of the operator.
    free_processed_packets();

    if (is_done_) return;

    BurstParams *burst;

    // In this example, we'll loop through all the rx queues of the interface
    // assuming we want to process the packets the same way for all queues
    const auto num_rx_queues = get_num_rx_queues(port_id_);

    for (int q = 0; q < num_rx_queues; q++) {
      auto status = get_rx_burst(&burst, port_id_, q);
      if (status != Status::SUCCESS) { continue; }

      auto burst_size = get_num_packets(burst);

      // Count packets received
      ttl_pkts_recv_ += burst_size;

      int p = 0;
      while (p < burst_size) {
        int space_in_batch = rows_per_tensor_ - aggr_pkts_recv_;
        int packets_to_copy = std::min(static_cast<int>(burst_size - p), space_in_batch);

        for (int i = 0; i < packets_to_copy; i++) {
          int pkt_in_burst = p + i;
          
          if (gpu_direct_.get()) {
            if (hds_.get()) {
              h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + i] = burst->pkts[1][pkt_in_burst];
            } else {
              h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + i] = 
                  reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_in_burst)) + header_size_.get();
            }
          } else {
            // CPU fallback (Warning: naive copy, doesn't reorder within frame)
            auto payload_ptr = reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, pkt_in_burst)) + header_size_.get();
            auto burst_offset = (aggr_pkts_recv_ + i) * nom_payload_size_;
            
            // We strip the 64-byte custom header for the Host copy because no reorder kernel is run for it
            memcpy((char*)full_batch_data_h_[cur_batch_idx_] + burst_offset,
                   payload_ptr + custom_header_size_,
                   nom_payload_size_);
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
          aggr_pkts_recv_ = 0; // reset for next tensor
          
          if (batch_q_.size() == num_concurrent) {
            HOLOSCAN_LOG_ERROR("Fell behind. All buffers queued. Dropping incoming batch silently.");
            cur_batch_.num_bursts = 0; 
            continue; // We technically leak burst tracking here if we just continue, but helps avoid total halt
          }

          if (gpu_direct_.get()) {
          // GPUDirect mode: we copy the payload (referenced in h_dev_ptrs_)
          // to a contiguous memory buffer (full_batch_data_d_)
          // NOTE: there is no actual reordering since we use the same order as packets came in,
          //   but they would be reordered if h_dev_ptrs_ was filled based on packet sequence id.
          // We also allow disabling the reorder kernel if alignment and memory types are not
          // supported. Currently the reorder kernel expects the packets to be 16B-aligned, and
          // anything that's not will cause an access error on the GPU
            if (reorder_kernel_.get()) {

              // Copy the list of pointers to the device
              CUDA_TRY(cudaMemcpyAsync(d_dev_ptrs_[cur_batch_idx_],
                                       h_dev_ptrs_[cur_batch_idx_],
                                       sizeof(void*) * rows_per_tensor_,
                                       cudaMemcpyHostToDevice,
                                       streams_[cur_batch_idx_]));

              // gather_packets translates out-of-order packets based on the 16-bit row ID
              gather_packets(reinterpret_cast<uint8_t**>(d_dev_ptrs_[cur_batch_idx_]),
                             static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                             nom_payload_size_, custom_header_size_, rows_per_tensor_, rows_per_tensor_,
                             streams_[cur_batch_idx_]);
            }
          } else {
            // Non GPUDirect mode: we copy the payload on host-pinned memory (in full_batch_data_h_)
            // to a contiguous memory buffer on the GPU (full_batch_data_d_)
            // NOTE: there is no reordering support here at all
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
          HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(gxf_tensor->pointer(),
                          full_batch_data_d_[cur_batch_idx_],
                          gxf_tensor->bytes_size(),
                          cudaMemcpyDeviceToDevice,
                          streams_[cur_batch_idx_]));

          auto maybedl = gxf_tensor->toDLManagedTensorContext();
          auto holoscan_tensor = std::make_shared<Tensor>(maybedl.value());
          
          TensorMap out_message;
          out_message.insert({"frame", holoscan_tensor});
          op_output.emit(out_message, "output");

          total_frames_emitted_ += frames_per_tensor_.get();

          cudaEventRecord(events_[cur_batch_idx_], streams_[cur_batch_idx_]);
          cur_batch_.evt = events_[cur_batch_idx_];
          batch_q_.push(cur_batch_);

          cur_batch_.num_bursts = 0;
          cur_batch_idx_ = (++cur_batch_idx_ % num_concurrent);

          if (count_.get() > 0 && total_frames_emitted_ >= count_.get()) {
            HOLOSCAN_LOG_INFO("StemReceiverOp: Reached frame limit of {}", count_.get());
            is_done_ = true;
            return;
          }
        }
      }
    }
  }

 private:
  static constexpr int num_concurrent = 20; 
  static constexpr int MAX_BURSTS_PER_BATCH = 5000;

  // Holds burst buffers that cannot be freed yet and CUDA event indicating when they can be freed
  struct BatchAggregationParams {
    std::array<BurstParams*, MAX_BURSTS_PER_BATCH> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  int port_id_;
  BatchAggregationParams cur_batch_{};
  int cur_batch_idx_ = 0;
  std::queue<BatchAggregationParams> batch_q_;
  
  int64_t ttl_bytes_recv_ = 0;
  int64_t ttl_pkts_recv_ = 0;
  int64_t ttl_packets_dropped_ = 0;

  int64_t aggr_pkts_recv_ = 0;
  uint64_t total_frames_emitted_ = 0;
  
  uint16_t custom_header_size_;  
  uint16_t nom_payload_size_;    
  uint32_t rows_per_tensor_;     

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
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  bool is_done_ = false;

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
};

}  // namespace holoscan::ops
