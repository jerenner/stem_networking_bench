// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "holoscan/core/domain/tensor.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops {

void PyTorchProcessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input")
	  .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,holoscan::Arg("capacity", 100UL));
  spec.output<holoscan::TensorMap>("output");

  // Add an allocator parameter needed for creating the output tensor
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output tensors.");
  spec.param(noop_, "noop", "No-Op Mode", "If true, pass input tensor to output without processing.", false);
}

void PyTorchProcessorOp::initialize() {
    // Add a default allocator if not provided
    if (!allocator_.has_value()) {
        allocator_ = fragment()->make_resource<UnboundedAllocator>("pytorch_processor_allocator");
        add_arg(allocator_.get());
    }

    Operator::initialize();
    
    if (torch::cuda::is_available()) {
        conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).stride(1).padding(1).bias(false));
        conv_->to(torch::kCUDA);
        HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Conv2d module initialized on GPU.");
    } else {
        HOLOSCAN_LOG_WARN("PyTorchProcessorOp: CUDA missing in PyTorch! Module will NOT be initialized.");
    }
}

void PyTorchProcessorOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {

  // Receive a gxf::Entity
  //auto in_entity = op_input.receive<gxf::Entity>("input").value();
  // Receive a TensorMap
  auto in_message = op_input.receive<TensorMap>("input").value();

  // Find the tensor within the entity by name
  //auto frame_tensor = in_entity.get<holoscan::Tensor>("frame");
  auto frame_tensor = in_message.at("frame");
  //if (!frame_tensor) {
  //  HOLOSCAN_LOG_ERROR("Failed to get tensor 'frame' from input entity.");
  //  return;
  //}

  // Get the GPU data pointer from the Holoscan Tensor
  void* gpu_data_ptr = frame_tensor->data();
  auto shape = frame_tensor->shape(); // e.g., [height, width]

  void* out_tensor_data_ptr = gpu_data_ptr; // Default to pass-through

  bool use_torch_cuda = torch::cuda::is_available();
  torch::Tensor output_tensor;
  bool is_float = false;

  if (use_torch_cuda && !noop_.get()) {
      auto options = torch::TensorOptions()
          .dtype(torch::kUInt16)
          .device(torch::kCUDA);
      
      std::vector<int64_t> pt_sizes;
      for (auto& s : shape) pt_sizes.push_back((long)s);

      torch::Tensor pt_tensor = torch::from_blob(gpu_data_ptr, pt_sizes, options);
      
      torch::Tensor reshaped_tensor;
      if (shape.size() == 3) {
          // Assume [Batch, Height, Width], reshape to [Batch, 1, Height, Width]
          reshaped_tensor = pt_tensor.reshape({(long)shape[0], 1, (long)shape[1], (long)shape[2]});
      } else {
          reshaped_tensor = pt_tensor;
      }
      
      torch::Tensor frame_float32 = reshaped_tensor.to(torch::kFloat32);

      output_tensor = conv_->forward(frame_float32);
      out_tensor_data_ptr = output_tensor.data_ptr();
      is_float = true;
  } else {
      if (frames_processed_ == 0) {
          HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Bypassing inference (NOOP Mode). Passes original Uint16 tensor natively.");
      }
      is_float = false;
  }

  // === TENSOR EMISSION LOGIC ===
  auto gxf_out_tensor = std::make_shared<nvidia::gxf::Tensor>();
  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  nvidia::gxf::Shape out_gxf_shape;

  if (is_float) {
      auto out_shape = output_tensor.sizes(); // shape is [N, C, H, W]
      out_gxf_shape = nvidia::gxf::Shape{static_cast<int32_t>(out_shape[0]), 
                                         static_cast<int32_t>(out_shape[1]), 
                                         static_cast<int32_t>(out_shape[2]), 
                                         static_cast<int32_t>(out_shape[3])};
      auto result = gxf_out_tensor->reshape<float>(
          out_gxf_shape,
          nvidia::gxf::MemoryStorageType::kDevice,
          allocator_handle.value());
      if (!result) { throw std::runtime_error("Failed to reshape processor output tensor"); }
  } else {
      // NOOP pass-through original uint16_t shape
      std::vector<int32_t> i32_shape;
      for (auto s : shape) i32_shape.push_back(s);
      
      if (shape.size() == 3) {
          out_gxf_shape = nvidia::gxf::Shape{i32_shape[0], i32_shape[1], i32_shape[2]};
      } else if (shape.size() == 4) {
          out_gxf_shape = nvidia::gxf::Shape{i32_shape[0], i32_shape[1], i32_shape[2], i32_shape[3]};
      }
      auto result = gxf_out_tensor->reshape<uint16_t>(
          out_gxf_shape,
          nvidia::gxf::MemoryStorageType::kDevice,
          allocator_handle.value());
      if (!result) { throw std::runtime_error("Failed to reshape processor output tensor"); }
  }

  // 3. Copy data to the new GXF tensor
  HOLOSCAN_CUDA_CALL(cudaMemcpy(gxf_out_tensor->pointer(),
                               out_tensor_data_ptr,
                               gxf_out_tensor->bytes_size(),
                               cudaMemcpyDeviceToDevice));
  
  // 4. Wrap in a Holoscan Tensor
  auto holoscan_out_tensor = std::make_shared<Tensor>(gxf_out_tensor->toDLManagedTensorContext().value());

  // 5. Place in a TensorMap and emit
  TensorMap out_map;
  out_map.insert({"processed_frame", holoscan_out_tensor});
  op_output.emit(out_map, "output");

  frames_processed_++;
  if (frames_processed_ % 100 == 0) {
      HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Processed frame {}", frames_processed_);
  }
}

}  // namespace holoscan::ops
