// hdf5_writer_op.h
#pragma once

#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"
#include <H5Cpp.h> // HDF5 C++ API

namespace holoscan::ops {

class HDF5WriterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HDF5WriterOp)

  HDF5WriterOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override;

 private:
  Parameter<std::string> filepath_;
  Parameter<std::string> dataset_name_;
  Parameter<bool> noop_;

  // HDF5 specific members
  std::unique_ptr<H5::H5File> file_;
  std::unique_ptr<H5::DataSet> dataset_;
  std::unique_ptr<H5::DataSpace> filespace_;

  hsize_t current_dims_[3] = {0, 0, 0};
  size_t frame_count_ = 0;
  bool dataset_created_ = false;

  // Host buffer for transferring data from GPU before writing
  std::vector<float> host_buffer_;
};

}  // namespace holoscan::ops
