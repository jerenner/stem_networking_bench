// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "holoscan/core/domain/tensor.hpp"

namespace holoscan::ops {

void PyTorchProcessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input");
}

void PyTorchProcessorOp::initialize() {
    Operator::initialize();
    conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).stride(1).padding(1).bias(false));
    conv_->to(torch::kCUDA);
    HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Conv2d module initialized on GPU.");
}

void PyTorchProcessorOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {

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

  // Wrap the GPU data in a PyTorch tensor (zero-copy)
  auto options = torch::TensorOptions()
      .dtype(torch::kUInt16)
      .device(torch::kCUDA);
  torch::Tensor pt_tensor = torch::from_blob(gpu_data_ptr, {(long)shape[0], (long)shape[1]}, options);
  
  // Reshape and convert to float for the convolution
  torch::Tensor reshaped_tensor = pt_tensor.reshape({1, 1, (long)shape[0], (long)shape[1]});
  torch::Tensor frame_float32 = reshaped_tensor.to(torch::kFloat32);

  // Run the processing
  auto output_tensor = conv_->forward(frame_float32);

  frames_processed_++;
  if (frames_processed_ % 100 == 0) {
      HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Processed frame {}", frames_processed_);
  }
}

}  // namespace holoscan::ops
