// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "holoscan/core/domain/tensor.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops {

void PyTorchProcessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input");
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

  // Only allocate and use PyTorch on CUDA if it was natively compiled with GPU support
  torch::Tensor pt_tensor;
  torch::Tensor reshaped_tensor;
  torch::Tensor frame_float32;
  torch::Tensor output_tensor;

  bool use_torch_cuda = torch::cuda::is_available();

  if (use_torch_cuda) {
      auto options = torch::TensorOptions()
          .dtype(torch::kUInt16)
          .device(torch::kCUDA);
      pt_tensor = torch::from_blob(gpu_data_ptr, {(long)shape[0], (long)shape[1]}, options);
      
      reshaped_tensor = pt_tensor.reshape({1, 1, (long)shape[0], (long)shape[1]});
      frame_float32 = reshaped_tensor.to(torch::kFloat32);

      if (!noop_.get()) {
          output_tensor = conv_->forward(frame_float32);
          out_tensor_data_ptr = output_tensor.data_ptr();
      } else {
          output_tensor = frame_float32;
          out_tensor_data_ptr = output_tensor.data_ptr();
      }
  } else {
      if (frames_processed_ == 0) {
          HOLOSCAN_LOG_WARN("PyTorchProcessorOp: PyTorch installation lacks CUDA support! Bypassing PyTorch inference entirely and simulating no-op passthrough.");
      }
  }

  // === TENSOR EMISSION LOGIC ===
  // 1. Create a GXF Tensor for the output
  auto gxf_out_tensor = std::make_shared<nvidia::gxf::Tensor>();
  
  // 2. Reshape it (allocates memory)
  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  
  int32_t out_channels = 1;
  int32_t out_height = shape[0];
  int32_t out_width = shape[1];
  int32_t element_size = sizeof(uint16_t);

  if (use_torch_cuda) {
      auto out_shape = output_tensor.sizes(); // shape is [N, C, H, W]
      out_channels = out_shape[1];
      out_height = out_shape[2];
      out_width = out_shape[3];
      element_size = sizeof(float); // Float32 from PyTorch
  }

  auto result = gxf_out_tensor->reshape<uint8_t>( // Use uint8_t just for raw memory allocation
      nvidia::gxf::Shape{1, out_channels, out_height, out_width * element_size},
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator_handle.value());
  if (!result) { throw std::runtime_error("Failed to reshape processor output tensor"); }

  // 3. Copy from the PyTorch tensor to the new GXF tensor
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
