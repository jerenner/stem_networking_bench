// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "nvtx_ranges.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include <H5Cpp.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::ops {

namespace {

std::string normalizeHdf5DatasetPath(const std::string& dataset_path) {
  if (dataset_path.empty()) {
    return "/processed";
  }
  return dataset_path.front() == '/' ? dataset_path : "/" + dataset_path;
}

torch::Dtype torchDtypeFromDLDataType(const DLDataType& dtype) {
  if (dtype.code == kDLUInt && dtype.bits == 16) {
    return torch::kUInt16;
  }
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    return torch::kFloat32;
  }
  throw std::runtime_error("PyTorchProcessorOp: unsupported input tensor dtype for PyTorch wrapping");
}

nvidia::gxf::Shape makeGxfShape(torch::IntArrayRef sizes) {
  if (sizes.size() == 2) {
    return nvidia::gxf::Shape{static_cast<int32_t>(sizes[0]),
                              static_cast<int32_t>(sizes[1])};
  }
  if (sizes.size() == 3) {
    return nvidia::gxf::Shape{static_cast<int32_t>(sizes[0]),
                              static_cast<int32_t>(sizes[1]),
                              static_cast<int32_t>(sizes[2])};
  }
  if (sizes.size() == 4) {
    return nvidia::gxf::Shape{static_cast<int32_t>(sizes[0]),
                              static_cast<int32_t>(sizes[1]),
                              static_cast<int32_t>(sizes[2]),
                              static_cast<int32_t>(sizes[3])};
  }
  throw std::runtime_error("PyTorchProcessorOp: output tensor rank must be 2, 3, or 4");
}

void validateDarkFrameShape(torch::IntArrayRef frame_sizes,
                            int64_t dark_frame_height,
                            int64_t dark_frame_width) {
  if (frame_sizes.size() < 2) {
    throw std::runtime_error("PyTorchProcessorOp: dark subtraction requires at least a 2D tensor");
  }

  const int64_t frame_height = frame_sizes[frame_sizes.size() - 2];
  const int64_t frame_width = frame_sizes[frame_sizes.size() - 1];
  if (frame_height != dark_frame_height || frame_width != dark_frame_width) {
    throw std::runtime_error("PyTorchProcessorOp: dark frame shape does not match incoming frame shape");
  }
}

}  // namespace

void PyTorchProcessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input")
	  .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,holoscan::Arg("capacity", 4UL));
  spec.output<holoscan::TensorMap>("output")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer,holoscan::Arg("capacity", 4UL));

  // Add an allocator parameter needed for creating the output tensor
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output tensors.");
  spec.param(noop_, "noop", "No-Op Mode", "If true, pass input tensor to output without processing.", false);
  spec.param(subtract_dark_frame_, "subtract_dark_frame", "Subtract Dark Frame",
             "If true, load dark_frame_path and subtract it from each incoming frame on GPU.", false);
  spec.param(dark_frame_path_, "dark_frame_path", "Dark Frame Path",
             "HDF5 file containing a single averaged dark frame.", std::string(""));
  spec.param(dark_frame_dataset_, "dark_frame_dataset", "Dark Frame Dataset",
             "Dataset path inside dark_frame_path. Supports [H,W] or [1,H,W].",
             std::string("/processed"));
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

    if (subtract_dark_frame_.get()) {
        loadDarkFrame();
    }
}

void PyTorchProcessorOp::loadDarkFrame() {
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("PyTorchProcessorOp: dark-frame subtraction requires CUDA-enabled PyTorch");
  }
  if (dark_frame_path_.get().empty()) {
    throw std::runtime_error("PyTorchProcessorOp: subtract_dark_frame=true requires dark_frame_path");
  }

  const std::string dataset_path = normalizeHdf5DatasetPath(dark_frame_dataset_.get());

  try {
    H5::Exception::dontPrint();
    H5::H5File file(dark_frame_path_.get(), H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(dataset_path);
    H5::DataSpace dataspace = dataset.getSpace();

    const int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data());

    if (rank == 2) {
      dark_frame_height_ = static_cast<int64_t>(dims[0]);
      dark_frame_width_ = static_cast<int64_t>(dims[1]);
    } else if (rank == 3 && dims[0] == 1) {
      dark_frame_height_ = static_cast<int64_t>(dims[1]);
      dark_frame_width_ = static_cast<int64_t>(dims[2]);
    } else {
      throw std::runtime_error(
          "dark frame dataset must have shape [H,W] or [1,H,W]; use make_dark_frame.py to average raw stacks");
    }

    const size_t num_pixels = static_cast<size_t>(dark_frame_height_) *
                              static_cast<size_t>(dark_frame_width_);
    std::vector<float> dark_frame_data(num_pixels);
    dataset.read(dark_frame_data.data(), H5::PredType::NATIVE_FLOAT);

    auto cpu_tensor = torch::from_blob(
                          dark_frame_data.data(),
                          {dark_frame_height_, dark_frame_width_},
                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                          .clone();
    dark_frame_tensor_ = cpu_tensor.to(torch::kCUDA).contiguous();
    dark_frame_loaded_ = true;

    HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Loaded dark frame '{}' dataset '{}' with shape [{}, {}]. "
                      "Dark-subtracted output will be float32.",
                      dark_frame_path_.get(),
                      dataset_path,
                      dark_frame_height_,
                      dark_frame_width_);
  } catch (const H5::Exception& e) {
    HOLOSCAN_LOG_ERROR("PyTorchProcessorOp: failed to load dark frame HDF5 '{}:{}': {}",
                       dark_frame_path_.get(),
                       dataset_path,
                       e.getCDetailMsg());
    throw;
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
  bool subtract_dark = subtract_dark_frame_.get();
  torch::Tensor output_tensor;

  if (use_torch_cuda && (!noop_.get() || subtract_dark)) {
      profiling::ScopedRange torch_range("processor/torch-processing", profiling::color::kCompute);
      auto options = torch::TensorOptions()
          .dtype(torchDtypeFromDLDataType(frame_tensor->dtype()))
          .device(torch::kCUDA);
      
      std::vector<int64_t> pt_sizes;
      for (auto& s : shape) pt_sizes.push_back((long)s);

      torch::Tensor pt_tensor = torch::from_blob(gpu_data_ptr, pt_sizes, options);

      torch::Tensor working_tensor = pt_tensor.to(torch::kFloat32);
      if (subtract_dark) {
          if (!dark_frame_loaded_) {
              throw std::runtime_error("PyTorchProcessorOp: subtract_dark_frame=true but no dark frame is loaded");
          }
          validateDarkFrameShape(working_tensor.sizes(), dark_frame_height_, dark_frame_width_);
          working_tensor = working_tensor - dark_frame_tensor_;
      }

      if (!noop_.get()) {
          // We expect [128, 1024, 3840].
          // Sum all frames into a single 1024-row frame: [Batch, H, W] -> [H, W].
          output_tensor = working_tensor.sum(0, /*keepdim=*/false).contiguous();
      } else {
          output_tensor = working_tensor.contiguous();
      }
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

  auto out_shape = output_tensor.sizes();
  nvidia::gxf::Shape out_gxf_shape = makeGxfShape(out_shape);

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
