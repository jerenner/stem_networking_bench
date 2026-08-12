/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <memory>
#include <string>

namespace stem {

struct ControlServerConfig {
  bool enabled = false;
  std::string endpoint = "tcp://*:5557";
  std::string runtime_config_path = "/tmp/stem_daqiri_runtime.yaml";
};

class ControlServer {
 public:
  using Handler = std::function<std::string(const std::string&)>;

  ControlServer(const ControlServerConfig& config, Handler handler);
  ~ControlServer();

  ControlServer(const ControlServer&) = delete;
  ControlServer& operator=(const ControlServer&) = delete;

  bool enabled() const;
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stem
