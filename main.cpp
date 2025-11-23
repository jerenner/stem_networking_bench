/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hdf5_replayer_op.h"
#include "stem_receiver_op.h"
#include "pytorch_processor_op.h"
#include "hdf5_writer_op.h"
#include "advanced_network/kernels.h"
#include "holoscan/holoscan.hpp"
#include <assert.h>
#include <sys/time.h>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // The processor is common to both paths
    auto processor = make_operator<ops::PyTorchProcessorOp>("processor", from_config("processor"));
    auto writer = make_operator<ops::HDF5WriterOp>("writer", from_config("writer"));

    // Check config to decide which data source to use
    std::string source_type = from_config("source").as<std::string>();

    if (source_type == "hdf5") {
      HOLOSCAN_LOG_INFO("Using HDF5 data source.");
      auto replayer = make_operator<ops::HDF5ReplayerOp>("replayer", from_config("replayer"));
      add_flow(replayer, processor);
    } else { // Default to network
      HOLOSCAN_LOG_INFO("Using Advanced Network data source.");

      auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
      if (advanced_network::adv_net_init(adv_net_config) != advanced_network::Status::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
        exit(1);
      }
      HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

      // DPDK is the default manager backend
      auto receiver = make_operator<ops::StemReceiverOp>("receiver", from_config("receiver"));    
      add_flow(receiver, processor, {{"output", "input"}});
    }

    add_flow(processor, writer);
  }
};

int main(int argc, char** argv) {
  using namespace holoscan;
  auto app = make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

  std::filesystem::path config_path(argv[1]);
  if (!config_path.is_absolute()) {
    config_path = std::filesystem::canonical(argv[0]).parent_path() / config_path;
  }
  app->config(config_path);
  app->scheduler(app->make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                           app->from_config("scheduler")));
  app->run();

  advanced_network::shutdown();
  return 0;
}
