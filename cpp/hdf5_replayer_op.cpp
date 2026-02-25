// hdf5_replayer_op.cpp
#include "hdf5_replayer_op.h"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "gxf/std/tensor.hpp"

namespace holoscan::ops {

void HDF5ReplayerOp::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("output");

  spec.param(filepath_, "filepath", "File Path", "Path to the HDF5 file.");
  spec.param(dataset_name_, "dataset_name", "Dataset Name", "Name of the dataset to read (e.g., '/frames').");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output tensors.");
  spec.param(repeat_, "repeat", "Repeat", "Repeat reading the dataset when the end is reached.", false);
  spec.param(count_, "count", "Count", "Number of frames to read. 0 means all frames.", 0UL);

  // This condition is used to stop the operator from ticking
  spec.param(stop_condition_, "stop_condition", "Stop Condition", "Boolean condition to stop execution.");
}

void HDF5ReplayerOp::initialize() {

  // Find if an argument for 'allocator' was already provided
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) {
          return (arg.name() == "allocator");
      });
  // If no allocator was provided, create a default UnboundedAllocator and add it as an argument
  if (has_allocator == args().end()) {
      HOLOSCAN_LOG_INFO("Creating default UnboundedAllocator...");
      auto allocator = fragment()->make_resource<UnboundedAllocator>("hdf5_replayer_allocator");
      add_arg(Arg("allocator", allocator));
  }

  // Find if an argument for 'stop_condition' was already provided
  auto has_stop_condition = std::find_if(
      args().begin(), args().end(), [](const auto& arg) {
          return (arg.name() == "stop_condition");
      });
  // If no stop_condition was provided, create a default BooleanCondition and add it as an argument
  if (has_stop_condition == args().end()) {
      auto stop_cond = fragment()->make_condition<BooleanCondition>("hdf5_stop_condition");
      add_arg(Arg("stop_condition", stop_cond));
  }

  Operator::initialize();

  // Open HDF5 file and dataset
  try {
    file_ = std::make_unique<H5::H5File>(filepath_.get(), H5F_ACC_RDONLY);
    dataset_ = std::make_unique<H5::DataSet>(file_->openDataSet(dataset_name_.get()));
    filespace_ = std::make_unique<H5::DataSpace>(dataset_->getSpace());

    int rank = filespace_->getSimpleExtentNdims();
    if (rank != 3) {
      throw std::runtime_error(fmt::format("Expected a 3D dataset, but got rank {}.", rank));
    }
    filespace_->getSimpleExtentDims(dims_, NULL);
    total_frames_ = dims_[0];
    
    HOLOSCAN_LOG_INFO("HDF5 file '{}' opened. Dataset '{}' has {} frames of size {}x{}.",
                      filepath_.get(), dataset_name_.get(), total_frames_, dims_[1], dims_[2]);

    // Allocate host buffer for one frame
    host_buffer_.resize(dims_[1] * dims_[2]);
  } catch (H5::Exception& e) {
    HOLOSCAN_LOG_ERROR("HDF5 error in initialize: {}", e.getCDetailMsg());
    throw;
  }
}

void HDF5ReplayerOp::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {

  HOLOSCAN_LOG_INFO("Starting HDF5 compute...");

  if (current_frame_index_ >= total_frames_ || (count_ > 0 && current_frame_index_ >= count_)) {
    if (repeat_.get()) {
      current_frame_index_ = 0;
      HOLOSCAN_LOG_INFO("End of HDF5 dataset reached, repeating.");
    } else {
      HOLOSCAN_LOG_INFO("End of HDF5 dataset reached, stopping.");
      stop_condition_.get()->disable_tick();
      return;
    }
  }

  // Define the hyperslab to read: one frame
  hsize_t offset[3] = {current_frame_index_, 0, 0};
  hsize_t count[3] = {1, dims_[1], dims_[2]};
  filespace_->selectHyperslab(H5S_SELECT_SET, count, offset);

  // Define the memory dataspace
  hsize_t mem_dims[2] = {dims_[1], dims_[2]};
  H5::DataSpace memspace(2, mem_dims);

  // Read data from the file into the host buffer
  HOLOSCAN_LOG_INFO("Reading data...");
  try {
    dataset_->read(host_buffer_.data(), H5::PredType::NATIVE_UINT16, memspace, *filespace_);
  } catch (H5::Exception& e) {
    HOLOSCAN_LOG_ERROR("HDF5 error in compute (read): {}", e.getCDetailMsg());
    throw;
  }

  HOLOSCAN_LOG_INFO("Creating tensor...");

  // 1. Create a new GXF Entity for the output message
  //auto out_message = nvidia::gxf::Entity::New(context.context());
  //if (!out_message) { throw std::runtime_error("Failed to create output message entity"); }
  //HOLOSCAN_LOG_INFO("-> Created GXF Entity.");

  // 2. Add a GXF Tensor component to the entity
  //auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(dataset_name_.get().c_str());
  //if (!out_tensor) { throw std::runtime_error("Failed to add tensor to output entity"); }
  //HOLOSCAN_LOG_INFO("-> Added GXF Tensor component to the Entity.");
  

  // Create a GXF Tensor to handle allocation.
  auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
  HOLOSCAN_LOG_INFO("-> Created a GXF Tensor");

  // Reshape the GXF Tensor, using the operator's allocator
  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  HOLOSCAN_LOG_INFO("-> Created allocator handle.");
  //auto result = out_tensor.value()->reshape<uint16_t>(
  auto result = gxf_tensor->reshape<uint16_t>(
      nvidia::gxf::Shape{static_cast<int>(dims_[1]), static_cast<int>(dims_[2])},
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator_handle.value());
  if (!result) { throw std::runtime_error("Failed to reshape output tensor"); }
  HOLOSCAN_LOG_INFO("-> Reshaped tensor.");

  // Copy data from host buffer to the new GPU tensor
  HOLOSCAN_CUDA_CALL(cudaMemcpy(gxf_tensor->pointer(), //out_tensor.value()->pointer(),
                               host_buffer_.data(),
                               gxf_tensor->bytes_size(), //out_tensor.value()->bytes_size(),
                               cudaMemcpyHostToDevice));
  HOLOSCAN_LOG_INFO("-> Copied data to GPU tensor");

  // Get the DLManagedTensorContext from the GXF tensor.
  //    This returns an Expected<> object.
  auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();

  // Check if the Expected object has a value before using it.
  if (!maybe_dl_ctx) {
      throw std::runtime_error("Failed to get DLManagedTensorContext from nvidia::gxf::Tensor");
  }
  // Extract the value from the Expected object.
  auto dl_ctx = maybe_dl_ctx.value();

  // Wrap the DL context in a Holoscan Tensor.
  auto holoscan_tensor = std::make_shared<Tensor>(dl_ctx);
  
  // Create a TensorMap and insert the Holoscan Tensor.
  TensorMap out_message;
  out_message.insert({"frame", holoscan_tensor});
  HOLOSCAN_LOG_INFO("-> Created TensorMap");

  // Emit the TensorMap.
  op_output.emit(out_message, "output");
  HOLOSCAN_LOG_INFO("-> Emitted the TensorMap");

  current_frame_index_++;

}

}  // namespace holoscan::ops
