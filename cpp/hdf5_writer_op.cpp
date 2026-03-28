// hdf5_writer_op.cpp
#include "hdf5_writer_op.h"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops {

void HDF5WriterOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("input")
      .connector(holoscan::IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", 4UL));

  spec.param(filepath_, "filepath", "File Path", "Path to the output HDF5 file.");
  spec.param(dataset_name_, "dataset_name", "Dataset Name", "Name of the dataset to write (e.g., '/processed_frames').");
  spec.param(noop_, "noop", "No-Op Mode", "If true, receive tensor but do not write to file.", false);
}

void HDF5WriterOp::initialize() {
  Operator::initialize();

  // Open HDF5 file for writing, truncate if it exists
  if (!noop_.get()) {
    try {
      file_ = std::make_unique<H5::H5File>(filepath_.get(), H5F_ACC_TRUNC);
    } catch (H5::Exception& e) {
      HOLOSCAN_LOG_ERROR("HDF5 error in initialize (open file): {}", e.getCDetailMsg());
      throw;
    }
  }
}

void HDF5WriterOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto in_message = op_input.receive<TensorMap>("input").value();
  
  if (noop_.get()) {
      return;
  }

  auto tensor = in_message.at("processed_frame");

  auto shape = tensor->shape();
  int batch_frames = 1;
  int height = 0;
  int width = 0;
  
  if (shape.size() == 2) { // Reduced 2D [H, W] from PyTorch
      height = shape[0];
      width = shape[1];
  } else if (shape.size() == 4) { // NCHW
      batch_frames = shape[0];
      height = shape[2];
      width = shape[3];
  } else { // default to 3 (Batch, H, W)
      batch_frames = shape[0];
      height = shape[1];
      width = shape[2];
  }

  H5::DataType h5_type = H5::PredType::NATIVE_FLOAT;
  auto dtype = tensor->dtype();
  if (dtype.code == kDLUInt && dtype.bits == 16) {
      h5_type = H5::PredType::NATIVE_UINT16;
  }

  if (!dataset_created_) {
    try {
      // Create a dataspace with unlimited dimension for frames
      hsize_t dims[3] = {(hsize_t)batch_frames, (hsize_t)height, (hsize_t)width};
      hsize_t max_dims[3] = {H5S_UNLIMITED, (hsize_t)height, (hsize_t)width};
      filespace_ = std::make_unique<H5::DataSpace>(3, dims, max_dims);

      // Create dataset creation property list to enable chunking
      H5::DSetCreatPropList prop;
      hsize_t chunk_dims[3] = {1, (hsize_t)height, (hsize_t)width};
      prop.setChunk(3, chunk_dims);

      // Create the dataset
      dataset_ = std::make_unique<H5::DataSet>(
          file_->createDataSet(dataset_name_.get(), h5_type, *filespace_, prop));
      
      current_dims_[0] = batch_frames;
      current_dims_[1] = height;
      current_dims_[2] = width;

      dataset_created_ = true;
    } catch (H5::Exception& e) {
      HOLOSCAN_LOG_ERROR("HDF5 error in compute (create dataset): {}", e.getCDetailMsg());
      throw;
    }
  } else {
    // Extend the dataset size for the new frames
    current_dims_[0] = frame_count_ + batch_frames;
    dataset_->extend(current_dims_);
  }

  // Get a fresh dataspace for the extended dataset
  filespace_ = std::make_unique<H5::DataSpace>(dataset_->getSpace());

  // Define the hyperslab for the new frame block
  hsize_t offset[3] = {frame_count_, 0, 0};
  hsize_t count[3] = {(hsize_t)batch_frames, (hsize_t)height, (hsize_t)width};
  filespace_->selectHyperslab(H5S_SELECT_SET, count, offset);

  // Define memory dataspace for the buffer
  hsize_t mem_dims[3] = {(hsize_t)batch_frames, (hsize_t)height, (hsize_t)width};
  H5::DataSpace memspace(3, mem_dims);

  // Copy data from GPU to host buffer
  host_buffer_.resize(tensor->nbytes());
  HOLOSCAN_CUDA_CALL(cudaMemcpy(host_buffer_.data(), tensor->data(), tensor->nbytes(), cudaMemcpyDeviceToHost));

  // Write the data
  try {
    dataset_->write(host_buffer_.data(), h5_type, memspace, *filespace_);
  } catch (H5::Exception& e) {
    HOLOSCAN_LOG_ERROR("HDF5 error in compute (write): {}", e.getCDetailMsg());
    throw;
  }

  frame_count_ += batch_frames;
  HOLOSCAN_LOG_INFO("HDF5WriterOp: Wrote up to frame {}", frame_count_);
}

}  // namespace holoscan::ops
