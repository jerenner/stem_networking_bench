// hdf5_replayer_op.h
#pragma once

#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"
#include <H5Cpp.h> // HDF5 C++ API

namespace holoscan::ops {

class HDF5ReplayerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HDF5ReplayerOp)

  HDF5ReplayerOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<std::string> filepath_;
  Parameter<std::string> dataset_name_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> repeat_;
  Parameter<uint64_t> count_;
  Parameter<std::shared_ptr<holoscan::BooleanCondition>> stop_condition_;

  // HDF5 specific members
  std::unique_ptr<H5::H5File> file_;
  std::unique_ptr<H5::DataSet> dataset_;
  std::unique_ptr<H5::DataSpace> filespace_;

  hsize_t dims_[3]; // Assuming 3D dataset [frames, height, width]
  size_t total_frames_ = 0;
  size_t current_frame_index_ = 0;

  // Host buffer for reading from HDF5 before GPU transfer
  std::vector<uint16_t> host_buffer_;
};

}  // namespace holoscan::ops
