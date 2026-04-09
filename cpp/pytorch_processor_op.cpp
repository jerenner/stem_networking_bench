// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "nvtx_ranges.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops {

void PyTorchProcessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input")
	  .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,holoscan::Arg("capacity", 4UL));
  spec.output<holoscan::TensorMap>("output")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,holoscan::Arg("capacity", 4UL));

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
  (void)context;
  profiling::ScopedRange compute_range("processor/compute", profiling::color::kProcessor);

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

  bool use_torch_cuda = torch::cuda::is_available();
  torch::Tensor output_tensor;

  if (use_torch_cuda && !noop_.get()) {
      profiling::ScopedRange torch_sum_range("processor/torch-sum", profiling::color::kCompute);
      auto options = torch::TensorOptions()
          .dtype(torch::kUInt16)
          .device(torch::kCUDA);
      
      std::vector<int64_t> pt_sizes;
      for (auto& s : shape) pt_sizes.push_back((long)s);

      torch::Tensor pt_tensor = torch::from_blob(gpu_data_ptr, pt_sizes, options);
      
      // We expect [128, 1024, 3840].
      // Sum all frames into a single 1024-row frame: [Batch, H, W] -> [H, W]
      // We convert to float32 first to prevent uint16_t overflow during summation.
      output_tensor = pt_tensor.to(torch::kFloat32).sum(0, /*keepdim=*/false).contiguous();
  } else {
      if (frames_processed_ == 0) {
          HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Bypassing inference (NOOP Mode). Passes original Uint16 tensor natively.");
      }
      profiling::ScopedRange noop_range("processor/noop-pass-through", profiling::color::kBackpressure);
      TensorMap out_map;
      out_map.insert({"processed_frame", frame_tensor});
      op_output.emit(out_map, "output");

      frames_processed_++;
      if (frames_processed_ % 100 == 0) {
          HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Processed frame {}", frames_processed_);
      }
      return;
  }

  // === TENSOR EMISSION LOGIC ===
  // 1. Create a GXF Tensor for the output
  profiling::ScopedRange wrap_range("processor/wrap-output", profiling::color::kCopy);
  auto gxf_out_tensor = std::make_shared<nvidia::gxf::Tensor>();
  nvidia::gxf::Shape out_gxf_shape;

  auto out_shape = output_tensor.sizes();
  if (out_shape.size() == 2) {
      out_gxf_shape = nvidia::gxf::Shape{static_cast<int32_t>(out_shape[0]),
                                         static_cast<int32_t>(out_shape[1])};
  } else if (out_shape.size() == 3) {
      out_gxf_shape = nvidia::gxf::Shape{static_cast<int32_t>(out_shape[0]),
                                         static_cast<int32_t>(out_shape[1]),
                                         static_cast<int32_t>(out_shape[2])};
  } else {
      out_gxf_shape = nvidia::gxf::Shape{static_cast<int32_t>(out_shape[0]),
                                         static_cast<int32_t>(out_shape[1]),
                                         static_cast<int32_t>(out_shape[2]),
                                         static_cast<int32_t>(out_shape[3])};
  }

  auto output_tensor_holder = std::make_shared<torch::Tensor>(output_tensor);
  auto result = gxf_out_tensor->wrapMemory(
      out_gxf_shape,
      nvidia::gxf::PrimitiveType::kFloat32,
      sizeof(float),
      nvidia::gxf::ComputeTrivialStrides(out_gxf_shape, sizeof(float)),
      nvidia::gxf::MemoryStorageType::kDevice,
      output_tensor_holder->data_ptr(),
      [output_tensor_holder](void*) mutable {
        output_tensor_holder.reset();
        return nvidia::gxf::Success;
      });
  if (!result) { throw std::runtime_error("Failed to wrap processor output tensor"); }

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
