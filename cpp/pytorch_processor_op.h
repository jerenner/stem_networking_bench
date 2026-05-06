// pytorch_processor_op.h
#pragma once

#include "holoscan/holoscan.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

namespace holoscan::ops {

class PyTorchProcessorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PyTorchProcessorOp)

  PyTorchProcessorOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override;

 private:
  void loadDarkFrame();
  void loadValidPixelMask();

  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> noop_;
  Parameter<bool> subtract_dark_frame_;
  Parameter<std::string> dark_frame_path_;
  Parameter<std::string> dark_frame_dataset_;
  Parameter<bool> apply_valid_pixel_mask_;
  Parameter<std::string> valid_pixel_mask_dataset_;

  torch::nn::Conv2d conv_{nullptr};
  torch::Tensor dark_frame_tensor_;
  torch::Tensor valid_pixel_mask_tensor_;
  int64_t dark_frame_height_ = 0;
  int64_t dark_frame_width_ = 0;
  int64_t valid_pixel_mask_height_ = 0;
  int64_t valid_pixel_mask_width_ = 0;
  bool dark_frame_loaded_ = false;
  bool valid_pixel_mask_loaded_ = false;
  long long frames_processed_ = 0;
};

}  // namespace holoscan::ops
