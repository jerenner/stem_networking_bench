// pytorch_processor_op.h
#pragma once

#include "holoscan/holoscan.hpp"
#include <torch/torch.h>
#include <torch/script.h>

namespace holoscan::ops {

class PyTorchProcessorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PyTorchProcessorOp)

  PyTorchProcessorOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> noop_;
  torch::nn::Conv2d conv_{nullptr};
  long long frames_processed_ = 0;
};

}  // namespace holoscan::ops
