// pytorch_processor_op.cpp
#include "pytorch_processor_op.h"
#include "nvtx_ranges.hpp"
#include "kernels.cuh"
#include "holoscan/core/domain/tensor.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <H5Cpp.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::ops {

namespace {

struct Hdf5FrameData {
  std::vector<float> pixels;
  int64_t height = 0;
  int64_t width = 0;
  std::string dataset_path;
};

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

bool isUInt16Tensor(const DLDataType& dtype) {
  return dtype.code == kDLUInt && dtype.bits == 16;
}

uint32_t frameCountForShape(torch::IntArrayRef sizes) {
  if (sizes.size() < 2) {
    throw std::runtime_error("PyTorchProcessorOp: expected tensor rank >= 2 for dark correction");
  }

  int64_t frame_count = 1;
  for (size_t dim = 0; dim + 2 < sizes.size(); ++dim) {
    frame_count *= sizes[dim];
  }
  if (frame_count <= 0 ||
      frame_count > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("PyTorchProcessorOp: frame count is outside supported range");
  }
  return static_cast<uint32_t>(frame_count);
}

Hdf5FrameData readSingleFrameFloatDataset(const std::string& file_path,
                                          const std::string& dataset_path) {
  const std::string normalized_path = normalizeHdf5DatasetPath(dataset_path);
  H5::H5File file(file_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(normalized_path);
  H5::DataSpace dataspace = dataset.getSpace();

  const int rank = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> dims(rank);
  dataspace.getSimpleExtentDims(dims.data());

  Hdf5FrameData frame;
  frame.dataset_path = normalized_path;
  if (rank == 2) {
    frame.height = static_cast<int64_t>(dims[0]);
    frame.width = static_cast<int64_t>(dims[1]);
  } else if (rank == 3 && dims[0] == 1) {
    frame.height = static_cast<int64_t>(dims[1]);
    frame.width = static_cast<int64_t>(dims[2]);
  } else {
    throw std::runtime_error(
        "HDF5 dataset must have shape [H,W] or [1,H,W]; use make_dark_frame.py to average raw stacks");
  }

  const size_t num_pixels = static_cast<size_t>(frame.height) * static_cast<size_t>(frame.width);
  frame.pixels.resize(num_pixels);
  dataset.read(frame.pixels.data(), H5::PredType::NATIVE_FLOAT);
  return frame;
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
  spec.param(noop_,
             "noop",
             "No-Op Mode",
             "If true, skip reduction and emit a frame batch. Enabled corrections still run.",
             false);
  spec.param(subtract_dark_frame_, "subtract_dark_frame", "Subtract Dark Frame",
             "If true, load dark_frame_path and subtract it from each incoming frame on GPU.", false);
  spec.param(dark_frame_path_, "dark_frame_path", "Dark Frame Path",
             "HDF5 file containing a single averaged dark frame.", std::string(""));
  spec.param(dark_frame_dataset_, "dark_frame_dataset", "Dark Frame Dataset",
             "Dataset path inside dark_frame_path. Supports [H,W] or [1,H,W].",
             std::string("/processed"));
  spec.param(apply_valid_pixel_mask_, "apply_valid_pixel_mask", "Apply Valid Pixel Mask",
             "If true, load valid_pixel_mask_dataset from dark_frame_path and zero invalid pixels on GPU.",
             false);
  spec.param(valid_pixel_mask_dataset_, "valid_pixel_mask_dataset", "Valid Pixel Mask Dataset",
             "Dataset path for a 0/1 valid-pixel mask. Supports [H,W] or [1,H,W].",
             std::string("/valid_pixel_mask"));
  spec.param(apply_blr_correction_,
             "apply_blr_correction",
             "Apply BLR Correction",
             "If true, estimate and subtract the ImageJ-style per-frame baseline from top/bottom edge rows.",
             false);
  spec.param(blr_rows_,
             "blr_rows",
             "BLR Rows",
             "Number of top and bottom edge rows used to estimate the per-frame baseline.",
             30U);
  spec.param(blr_zlp_width_,
             "blr_zlp_width",
             "BLR ZLP Width",
             "Width in columns of the four-read ZLP region.",
             768U);
  spec.param(blr_zlp_group_columns_,
             "blr_zlp_group_columns",
             "BLR ZLP Group Columns",
             "Number of adjacent ZLP columns averaged into one baseline value.",
             4U);
  spec.param(blr_core_group_columns_,
             "blr_core_group_columns",
             "BLR Core Group Columns",
             "Number of adjacent CoreLoss columns averaged into one baseline value.",
             16U);
  spec.param(apply_dynamic_half_column_mask_,
             "apply_dynamic_half_column_mask",
             "Apply Dynamic Half-Column Mask",
             "If true, compute a batch-mean image on GPU and zero half-column local outliers.",
             false);
  spec.param(dynamic_mask_median_window_pixels_,
             "dynamic_mask_median_window_pixels",
             "Dynamic Mask Median Window Pixels",
             "Number of same-column pixels in the top/bottom half-column median window. Maximum supported value is 129.",
             31U);
  spec.param(dynamic_mask_threshold_ratio_,
             "dynamic_mask_threshold_ratio",
             "Dynamic Mask Threshold Ratio",
             "Scale applied to the local median before evaluating the outlier deviation.",
             1.0f);
  spec.param(dynamic_mask_threshold_offset_,
             "dynamic_mask_threshold_offset",
             "Dynamic Mask Threshold Offset",
             "Absolute deviation from local_median * ratio required to mask a pixel.",
             500.0f);
  spec.param(dynamic_mask_excluded_edge_rows_,
             "dynamic_mask_excluded_edge_rows",
             "Dynamic Mask Excluded Edge Rows",
             "Number of non-imaging rows excluded at the top and bottom of each frame.",
             32U);
  spec.param(dynamic_mask_two_sided_,
             "dynamic_mask_two_sided",
             "Dynamic Mask Two Sided",
             "If true, mask both positive and negative deviations from the local median.",
             true);
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
    if (apply_valid_pixel_mask_.get()) {
        loadValidPixelMask();
    }
    if (apply_blr_correction_.get()) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error(
                "PyTorchProcessorOp: BLR correction requires CUDA-enabled PyTorch");
        }
        if (blr_rows_.get() == 0 || blr_zlp_group_columns_.get() == 0 ||
            blr_core_group_columns_.get() == 0) {
            throw std::runtime_error(
                "PyTorchProcessorOp: BLR row and column-group sizes must be greater than zero");
        }
        HOLOSCAN_LOG_INFO(
            "PyTorchProcessorOp: ImageJ-style BLR correction enabled using {} edge rows, "
            "ZLP width {} grouped by {} columns, and CoreLoss grouped by {} columns.",
            blr_rows_.get(),
            blr_zlp_width_.get(),
            blr_zlp_group_columns_.get(),
            blr_core_group_columns_.get());
    }
    if (apply_dynamic_half_column_mask_.get()) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error(
                "PyTorchProcessorOp: dynamic half-column masking requires CUDA-enabled PyTorch");
        }
        if (dynamic_mask_median_window_pixels_.get() == 0 ||
            dynamic_mask_median_window_pixels_.get() > 129 ||
            dynamic_mask_median_window_pixels_.get() % 2 == 0) {
            throw std::runtime_error(
                "PyTorchProcessorOp: dynamic_mask_median_window_pixels must be odd and in [1, 129]");
        }
        if (dynamic_mask_threshold_offset_.get() < 0.0f) {
            throw std::runtime_error(
                "PyTorchProcessorOp: dynamic_mask_threshold_offset must be non-negative");
        }
        HOLOSCAN_LOG_INFO(
            "PyTorchProcessorOp: Dynamic half-column mask enabled with median window M={}, "
            "{} outlier detection around median * {:.3f} with offset {:.3f}; "
            "excluding {} top/bottom rows.",
            dynamic_mask_median_window_pixels_.get(),
            dynamic_mask_two_sided_.get() ? "two-sided" : "positive-only",
            dynamic_mask_threshold_ratio_.get(),
            dynamic_mask_threshold_offset_.get(),
            dynamic_mask_excluded_edge_rows_.get());
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
    Hdf5FrameData dark_frame = readSingleFrameFloatDataset(dark_frame_path_.get(), dataset_path);
    dark_frame_height_ = dark_frame.height;
    dark_frame_width_ = dark_frame.width;

    auto cpu_tensor = torch::from_blob(
                          dark_frame.pixels.data(),
                          {dark_frame_height_, dark_frame_width_},
                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                          .clone();
    dark_frame_tensor_ = cpu_tensor.to(torch::kCUDA).contiguous();
    dark_frame_loaded_ = true;

    HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Loaded dark frame '{}' dataset '{}' with shape [{}, {}]. "
                      "Dark-subtracted output will be float32.",
                      dark_frame_path_.get(),
                      dark_frame.dataset_path,
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

void PyTorchProcessorOp::loadValidPixelMask() {
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("PyTorchProcessorOp: valid-pixel masking requires CUDA-enabled PyTorch");
  }
  if (dark_frame_path_.get().empty()) {
    throw std::runtime_error("PyTorchProcessorOp: apply_valid_pixel_mask=true requires dark_frame_path");
  }

  const std::string dataset_path = normalizeHdf5DatasetPath(valid_pixel_mask_dataset_.get());

  try {
    H5::Exception::dontPrint();
    Hdf5FrameData valid_pixel_mask = readSingleFrameFloatDataset(dark_frame_path_.get(), dataset_path);
    valid_pixel_mask_height_ = valid_pixel_mask.height;
    valid_pixel_mask_width_ = valid_pixel_mask.width;

    auto cpu_tensor = torch::from_blob(
                          valid_pixel_mask.pixels.data(),
                          {valid_pixel_mask_height_, valid_pixel_mask_width_},
                          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                          .clone();
    valid_pixel_mask_tensor_ = cpu_tensor.to(torch::kCUDA).contiguous();
    valid_pixel_mask_loaded_ = true;

    HOLOSCAN_LOG_INFO("PyTorchProcessorOp: Loaded valid pixel mask '{}' dataset '{}' with shape [{}, {}]. "
                      "Invalid pixels will be zeroed after detector corrections.",
                      dark_frame_path_.get(),
                      valid_pixel_mask.dataset_path,
                      valid_pixel_mask_height_,
                      valid_pixel_mask_width_);
  } catch (const H5::Exception& e) {
    HOLOSCAN_LOG_ERROR("PyTorchProcessorOp: failed to load valid pixel mask HDF5 '{}:{}': {}",
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
  bool apply_valid_pixel_mask = apply_valid_pixel_mask_.get();
  bool apply_blr_correction = apply_blr_correction_.get();
  bool apply_dynamic_half_column_mask = apply_dynamic_half_column_mask_.get();
  torch::Tensor output_tensor;
  torch::Tensor batch_mean;

  if (use_torch_cuda &&
      (!noop_.get() || subtract_dark || apply_valid_pixel_mask || apply_blr_correction ||
       apply_dynamic_half_column_mask)) {
      profiling::ScopedRange torch_range("processor/torch-processing", profiling::color::kCompute);
      auto options = torch::TensorOptions()
          .dtype(torchDtypeFromDLDataType(frame_tensor->dtype()))
          .device(torch::kCUDA);
      
      std::vector<int64_t> pt_sizes;
      for (auto& s : shape) pt_sizes.push_back((long)s);

      torch::Tensor pt_tensor = torch::from_blob(gpu_data_ptr, pt_sizes, options);
      const auto tensor_sizes = pt_tensor.sizes();
      const uint32_t height =
          static_cast<uint32_t>(tensor_sizes[tensor_sizes.size() - 2]);
      const uint32_t width =
          static_cast<uint32_t>(tensor_sizes[tensor_sizes.size() - 1]);
      const uint32_t frames = frameCountForShape(tensor_sizes);
      cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

      if ((apply_blr_correction || apply_dynamic_half_column_mask) && height % 2 != 0) {
          throw std::runtime_error(
              "PyTorchProcessorOp: BLR correction and half-column masking require even frame height");
      }
      if (subtract_dark) {
          if (!dark_frame_loaded_) {
              throw std::runtime_error(
                  "PyTorchProcessorOp: subtract_dark_frame=true but no dark frame is loaded");
          }
          validateDarkFrameShape(tensor_sizes, dark_frame_height_, dark_frame_width_);
      }
      if (apply_valid_pixel_mask) {
          if (!valid_pixel_mask_loaded_) {
              throw std::runtime_error(
                  "PyTorchProcessorOp: apply_valid_pixel_mask=true but no valid pixel mask is loaded");
          }
          validateDarkFrameShape(
              tensor_sizes, valid_pixel_mask_height_, valid_pixel_mask_width_);
      }
      if (apply_blr_correction) {
          if (height < 2 * blr_rows_.get()) {
              throw std::runtime_error(
                  "PyTorchProcessorOp: frame height must be at least twice blr_rows");
          }
          if (blr_zlp_width_.get() > width ||
              blr_zlp_width_.get() % blr_zlp_group_columns_.get() != 0 ||
              (width - blr_zlp_width_.get()) % blr_core_group_columns_.get() != 0) {
              throw std::runtime_error(
                  "PyTorchProcessorOp: frame width is incompatible with configured BLR regions");
          }
      }
      if (apply_dynamic_half_column_mask &&
          2 * dynamic_mask_excluded_edge_rows_.get() >= height) {
          throw std::runtime_error(
              "PyTorchProcessorOp: dynamic_mask_excluded_edge_rows leaves no imaging pixels");
      }

      torch::Tensor working_tensor;
      const bool input_is_uint16 = isUInt16Tensor(frame_tensor->dtype());
      const bool use_fused_correction =
          subtract_dark || apply_valid_pixel_mask || apply_blr_correction ||
          apply_dynamic_half_column_mask;
      if (use_fused_correction) {
          if (!fused_correction_path_logged_) {
              HOLOSCAN_LOG_INFO(
                  "PyTorchProcessorOp: Using fused correction path for {} input "
                  "(dark subtraction: {}, BLR: {}, batch mean/dynamic mask: {}, "
                  "valid-pixel mask: {}).",
                  input_is_uint16 ? "uint16" : "float32",
                  subtract_dark,
                  apply_blr_correction,
                  apply_dynamic_half_column_mask,
                  apply_valid_pixel_mask);
              fused_correction_path_logged_ = true;
          }
          working_tensor = torch::empty(
              pt_sizes,
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

          torch::Tensor blr_baseline;
          if (apply_blr_correction) {
              profiling::ScopedRange baseline_range(
                  "processor/fused-blr-baseline", profiling::color::kCompute);
              const uint32_t blr_bins =
                  blr_zlp_width_.get() / blr_zlp_group_columns_.get() +
                  (width - blr_zlp_width_.get()) / blr_core_group_columns_.get();
              blr_baseline = torch::empty(
                  {static_cast<int64_t>(frames), 2, static_cast<int64_t>(blr_bins)},
                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
              if (input_is_uint16) {
                  compute_blr_baseline(
                      reinterpret_cast<const uint16_t*>(gpu_data_ptr),
                      subtract_dark ? dark_frame_tensor_.data_ptr<float>() : nullptr,
                      blr_baseline.data_ptr<float>(),
                      frames,
                      height,
                      width,
                      blr_rows_.get(),
                      blr_zlp_width_.get(),
                      blr_zlp_group_columns_.get(),
                      blr_core_group_columns_.get(),
                      subtract_dark,
                      stream);
              } else {
                  compute_blr_baseline(
                      reinterpret_cast<const float*>(gpu_data_ptr),
                      subtract_dark ? dark_frame_tensor_.data_ptr<float>() : nullptr,
                      blr_baseline.data_ptr<float>(),
                      frames,
                      height,
                      width,
                      blr_rows_.get(),
                      blr_zlp_width_.get(),
                      blr_zlp_group_columns_.get(),
                      blr_core_group_columns_.get(),
                      subtract_dark,
                      stream);
              }
          }
          if (apply_dynamic_half_column_mask) {
              batch_mean = torch::empty(
                  {static_cast<int64_t>(height), static_cast<int64_t>(width)},
                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
          }

          profiling::ScopedRange fused_range(
              "processor/fused-correction-blr-mean", profiling::color::kCompute);
          if (input_is_uint16) {
              correct_with_blr_and_mean(
                  reinterpret_cast<const uint16_t*>(gpu_data_ptr),
                  subtract_dark ? dark_frame_tensor_.data_ptr<float>() : nullptr,
                  apply_blr_correction ? blr_baseline.data_ptr<float>() : nullptr,
                  working_tensor.data_ptr<float>(),
                  apply_dynamic_half_column_mask ? batch_mean.data_ptr<float>() : nullptr,
                  frames,
                  height,
                  width,
                  blr_zlp_width_.get(),
                  blr_zlp_group_columns_.get(),
                  blr_core_group_columns_.get(),
                  subtract_dark,
                  apply_blr_correction,
                  apply_dynamic_half_column_mask,
                  stream);
          } else {
              correct_with_blr_and_mean(
                  reinterpret_cast<const float*>(gpu_data_ptr),
                  subtract_dark ? dark_frame_tensor_.data_ptr<float>() : nullptr,
                  apply_blr_correction ? blr_baseline.data_ptr<float>() : nullptr,
                  working_tensor.data_ptr<float>(),
                  apply_dynamic_half_column_mask ? batch_mean.data_ptr<float>() : nullptr,
                  frames,
                  height,
                  width,
                  blr_zlp_width_.get(),
                  blr_zlp_group_columns_.get(),
                  blr_core_group_columns_.get(),
                  subtract_dark,
                  apply_blr_correction,
                  apply_dynamic_half_column_mask,
                  stream);
          }
      } else {
          working_tensor = pt_tensor.to(torch::kFloat32);
      }

      if (apply_dynamic_half_column_mask) {
          profiling::ScopedRange mask_range(
              "processor/combined-pixel-mask", profiling::color::kCompute);
          apply_dynamic_and_valid_pixel_mask_float(
              working_tensor.data_ptr<float>(),
              apply_dynamic_half_column_mask ? batch_mean.data_ptr<float>() : nullptr,
              apply_valid_pixel_mask ? valid_pixel_mask_tensor_.data_ptr<float>() : nullptr,
              frames,
              height,
              width,
              dynamic_mask_median_window_pixels_.get(),
              dynamic_mask_threshold_ratio_.get(),
              dynamic_mask_threshold_offset_.get(),
              dynamic_mask_excluded_edge_rows_.get(),
              apply_dynamic_half_column_mask,
              dynamic_mask_two_sided_.get(),
              apply_valid_pixel_mask,
              stream);
      } else if (apply_valid_pixel_mask) {
          profiling::ScopedRange valid_mask_range(
              "processor/valid-pixel-mask", profiling::color::kCompute);
          apply_valid_pixel_mask_float(
              working_tensor.data_ptr<float>(),
              valid_pixel_mask_tensor_.data_ptr<float>(),
              frames,
              height,
              width,
              stream);
      }

      if (!noop_.get()) {
          // Reduce the corrected frame batch from [Batch, H, W] to [H, W].
          output_tensor = working_tensor.sum(0, /*keepdim=*/false).contiguous();
      } else {
          output_tensor = working_tensor.contiguous();
      }
  } else {
      if (frames_processed_ == 0) {
          HOLOSCAN_LOG_INFO(
              "PyTorchProcessorOp: Bypassing inference (NOOP Mode). "
              "Passes the original input tensor natively.");
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
