/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stem_control_server.h"

namespace {

volatile std::sig_atomic_t g_stop = 0;

void signal_handler(int) { g_stop = 1; }

std::string json_escape(const std::string& value) {
  std::string output;
  output.reserve(value.size() + 16);
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (character < 0x20) {
          char encoded[7];
          std::snprintf(encoded, sizeof(encoded), "\\u%04x", character);
          output += encoded;
        } else {
          output.push_back(static_cast<char>(character));
        }
    }
  }
  return output;
}

template <typename T>
T yaml_value(const YAML::Node& map, const char* key, const T& fallback) {
  if (!map || !map.IsMap()) { return fallback; }
  const YAML::Node value = map[key];
  if (!value || value.IsNull()) { return fallback; }
  return value.as<T>(fallback);
}

struct Operation {
  std::string id;
  std::string name;
  std::string state = "idle";
  std::string step;
  uint32_t completed_steps = 0;
  uint32_t total_steps = 0;
  std::string error;
};

struct ScanConfig {
  uint32_t pause_count = 200;
  uint32_t read_count = 1;
  uint32_t positions_x = 1;
  uint32_t rows = 1;
  uint32_t flyback = 100;
  bool flush_memory = true;
};

class MockInstrumentService {
 public:
  explicit MockInstrumentService(const YAML::Node& root) {
    const YAML::Node instrument = root["instrument"];
    const YAML::Node mock = instrument["mock"];
    step_duration_ms_ = yaml_value<uint32_t>(mock, "step_duration_ms", 150);
    initial_temperature_c_ =
        yaml_value<double>(mock, "initial_temperature_c", 22.0);
    temperature_c_ = initial_temperature_c_;
    target_temperature_c_ =
        yaml_value<double>(mock, "target_temperature_c", -20.0);
    fail_operation_ = yaml_value<std::string>(mock, "fail_operation", "");
    fail_step_ = yaml_value<uint32_t>(mock, "fail_step", 0);
    fail_once_ = yaml_value<bool>(mock, "fail_once", true);
    audit_log_path_ = yaml_value<std::string>(
        mock, "audit_log_path", "/tmp/stem_instrument_mock_audit.jsonl");
    worker_ = std::thread(&MockInstrumentService::run_worker, this);
  }

  ~MockInstrumentService() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
      cancel_requested_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
  }

  std::string handle(const std::string& request_text) {
    try {
      const YAML::Node request = YAML::Load(request_text);
      const std::string command = request["command"].as<std::string>(
          std::string("instrument.get_state"));
      std::lock_guard<std::mutex> lock(mu_);
      update_scan_progress_locked();

      if (command == "instrument.get_state" || command == "get_state" ||
          command == "operation.get") {
        return response_locked();
      }
      if (command == "shutdown") {
        g_stop = 1;
        audit_locked("service_shutdown", command, "requested");
        return response_locked("mock instrument service shutdown requested");
      }
      if (command == "camera.read_temperature") {
        return response_locked("mock camera temperature read");
      }
      if (command == "camera.read_biases") {
        return response_locked("mock detector biases read");
      }
      if (command == "detector.read_link_status") {
        return response_locked("mock detector link status read");
      }
      if (command == "mock.set_failure") {
        fail_operation_ = request["operation"].as<std::string>("");
        fail_step_ = request["step"].as<uint32_t>(0);
        fail_once_ = request["once"].as<bool>(true);
        return response_locked("mock failure injection updated");
      }
      if (command == "scan.configure") {
        require_idle_locked();
        configure_scan_locked(request["scan"] ? request["scan"] : request);
        scan_state_ = "configured";
        audit_locked("configuration", command, "accepted");
        return response_locked("mock scan configuration applied");
      }
      if (command == "scan.abort") {
        cancel_requested_ = true;
        if (operation_.state == "running") {
          operation_.state = "cancelled";
          operation_.error = "operation aborted by operator";
          operation_.step = "aborted";
        }
        scan_running_ = false;
        scan_state_ = "aborted";
        audit_locked("operation_cancelled", command, operation_.error);
        cv_.notify_all();
        return response_locked("mock scan aborted");
      }

      if (command == "camera.power_up") {
        return start_operation_locked(command,
            {"start camera services", "apply detector biases",
             "set temperature target", "configure ADCs",
             "verify camera readiness"});
      }
      if (command == "camera.power_down") {
        return start_operation_locked(
            command, {"retract camera", "stop camera services"});
      }
      if (command == "camera.insert") {
        if (camera_power_state_ != "ready") {
          throw std::runtime_error("camera must be ready before insertion");
        }
        return start_operation_locked(
            command, {"request insertion", "verify inserted position"});
      }
      if (command == "camera.retract") {
        return start_operation_locked(
            command, {"request retraction", "verify retracted position"});
      }
      if (command == "detector.resync") {
        return start_operation_locked(
            command, {"program ADC startup delay", "assert synchronization",
                      "reset GTX links", "reset JESD links",
                      "check ADC status", "release synchronization"});
      }
      if (command == "detector.auto_align") {
        if (!detector_synchronized_) {
          throw std::runtime_error("detector must be synchronized before auto-align");
        }
        return start_operation_locked(
            command, {"capture alignment samples", "measure lane offsets",
                      "apply data shifts", "verify alignment"});
      }
      if (command == "scan.start") {
        if (camera_power_state_ != "ready") {
          throw std::runtime_error("camera must be ready before scan start");
        }
        if (!detector_synchronized_) {
          throw std::runtime_error("detector must be synchronized before scan start");
        }
        return start_operation_locked(
            command, {"validate scan configuration", "arm future boundary",
                      "start scan"});
      }
      if (command == "scan.stop") {
        return start_operation_locked(
            command, {"request scan stop", "verify scan stopped"});
      }
      throw std::runtime_error("unknown instrument command " + command);
    } catch (const std::exception& error) {
      return std::string("{\"ok\":false,\"error\":\"") +
             json_escape(error.what()) + "\"}";
    }
  }

 private:
  void require_idle_locked() const {
    if (operation_.state == "running") {
      throw std::runtime_error("instrument operation already running: " +
                               operation_.name);
    }
  }

  void configure_scan_locked(const YAML::Node& scan) {
    ScanConfig updated = scan_config_;
    updated.pause_count = yaml_value<uint32_t>(scan, "pause_count", updated.pause_count);
    updated.read_count = yaml_value<uint32_t>(scan, "read_count", updated.read_count);
    updated.positions_x = yaml_value<uint32_t>(scan, "positions_x", updated.positions_x);
    updated.rows = yaml_value<uint32_t>(scan, "rows", updated.rows);
    updated.flyback = yaml_value<uint32_t>(scan, "flyback", updated.flyback);
    updated.flush_memory = yaml_value<bool>(scan, "flush_memory", updated.flush_memory);
    if (updated.read_count == 0 || updated.positions_x == 0 || updated.rows == 0) {
      throw std::runtime_error("scan read_count, positions_x, and rows must be positive");
    }
    if (updated.pause_count > 1000000 || updated.read_count > 1000000 ||
        updated.positions_x > 1000000 || updated.rows > 1000000 ||
        updated.flyback > 1000000) {
      throw std::runtime_error("mock scan parameter exceeds validation limit");
    }
    scan_config_ = updated;
    expected_frames_ = static_cast<uint64_t>(scan_config_.read_count) *
                       scan_config_.positions_x * scan_config_.rows;
  }

  std::string start_operation_locked(
      const std::string& name, std::vector<std::string> steps) {
    require_idle_locked();
    cancel_requested_ = false;
    operation_ = {};
    char identifier[32];
    std::snprintf(identifier, sizeof(identifier), "op-%05llu",
                  static_cast<unsigned long long>(next_operation_id_++));
    operation_.id = identifier;
    operation_.name = name;
    operation_.state = "running";
    operation_.step = "queued";
    operation_.total_steps = static_cast<uint32_t>(steps.size());
    pending_steps_ = std::move(steps);
    work_pending_ = true;
    audit_locked("operation_accepted", name, "queued");
    cv_.notify_all();
    return response_locked("mock operation accepted", true);
  }

  void run_worker() {
    while (true) {
      std::string operation_id;
      std::string operation_name;
      std::vector<std::string> steps;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || work_pending_; });
        if (stopping_) { return; }
        work_pending_ = false;
        operation_id = operation_.id;
        operation_name = operation_.name;
        steps = pending_steps_;
      }

      bool finished = true;
      for (size_t index = 0; index < steps.size(); ++index) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (stopping_) { return; }
          if (cancel_requested_ || operation_.id != operation_id ||
              operation_.state != "running") {
            finished = false;
            break;
          }
          operation_.step = steps[index];
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(step_duration_ms_));
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (cancel_requested_ || operation_.id != operation_id ||
              operation_.state != "running") {
            finished = false;
            break;
          }
          const uint32_t step_number = static_cast<uint32_t>(index + 1);
          if (operation_name == fail_operation_ && step_number == fail_step_) {
            operation_.state = "failed";
            operation_.error = "injected mock failure at step " +
                               std::to_string(step_number);
            operation_.completed_steps = static_cast<uint32_t>(index);
            audit_locked("operation_failed", operation_name, operation_.error);
            if (fail_once_) {
              fail_operation_.clear();
              fail_step_ = 0;
            }
            finished = false;
            break;
          }
          operation_.completed_steps = step_number;
        }
      }

      if (finished) {
        std::lock_guard<std::mutex> lock(mu_);
        if (operation_.id == operation_id && operation_.state == "running") {
          apply_success_locked(operation_name);
          operation_.state = "completed";
          operation_.step = "complete";
          operation_.error.clear();
          audit_locked("operation_completed", operation_name, "complete");
        }
      }
    }
  }

  void apply_success_locked(const std::string& name) {
    if (name == "camera.power_up") {
      camera_power_state_ = "ready";
      temperature_c_ = target_temperature_c_;
    } else if (name == "camera.power_down") {
      camera_power_state_ = "off";
      insertion_state_ = "retracted";
      temperature_c_ = initial_temperature_c_;
      detector_synchronized_ = false;
      links_.assign(4, "unknown");
    } else if (name == "camera.insert") {
      insertion_state_ = "inserted";
    } else if (name == "camera.retract") {
      insertion_state_ = "retracted";
    } else if (name == "detector.resync") {
      detector_synchronized_ = true;
      links_.assign(4, "ready");
    } else if (name == "detector.auto_align") {
      detector_aligned_ = true;
    } else if (name == "scan.start") {
      scan_running_ = true;
      scan_state_ = "running";
      received_frames_ = 0;
      ++scan_number_;
      scan_started_ = std::chrono::steady_clock::now();
    } else if (name == "scan.stop") {
      scan_running_ = false;
      scan_state_ = "stopped";
    }
  }

  void update_scan_progress_locked() {
    if (!scan_running_ || expected_frames_ == 0) { return; }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - scan_started_).count();
    received_frames_ = std::min<uint64_t>(
        expected_frames_, static_cast<uint64_t>(std::max<int64_t>(0, elapsed) / 5));
    if (received_frames_ >= expected_frames_) {
      scan_running_ = false;
      scan_state_ = "completed";
    }
  }

  std::string response_locked(const std::string& message = {},
                              bool accepted = false) const {
    std::ostringstream output;
    output << "{\"ok\":true,\"schema\":\"stem.instrument.v1\""
           << ",\"message\":\"" << json_escape(message) << "\"";
    if (accepted) {
      output << ",\"accepted\":true,\"operation_id\":\""
             << json_escape(operation_.id) << "\"";
    }
    output << ",\"instrument\":{"
           << "\"enabled\":true,\"service_online\":true,\"mode\":\"mock\""
           << ",\"camera\":{"
           << "\"power_state\":\"" << json_escape(camera_power_state_) << "\""
           << ",\"insertion_state\":\"" << json_escape(insertion_state_) << "\""
           << ",\"temperature_c\":" << temperature_c_
           << ",\"target_temperature_c\":" << target_temperature_c_
           << ",\"biases\":{\"reset_v\":0.72,\"substrate_v\":1.8,\"guard_v\":0.45}}"
           << ",\"detector\":{"
           << "\"synchronized\":" << (detector_synchronized_ ? "true" : "false")
           << ",\"aligned\":" << (detector_aligned_ ? "true" : "false")
           << ",\"links\":[";
    for (size_t index = 0; index < links_.size(); ++index) {
      if (index != 0) { output << ","; }
      output << "\"" << json_escape(links_[index]) << "\"";
    }
    output << "]}"
           << ",\"scan\":{"
           << "\"state\":\"" << json_escape(scan_state_) << "\""
           << ",\"running\":" << (scan_running_ ? "true" : "false")
           << ",\"scan_number\":" << scan_number_
           << ",\"expected_frames\":" << expected_frames_
           << ",\"received_frames\":" << received_frames_
           << ",\"pause_count\":" << scan_config_.pause_count
           << ",\"read_count\":" << scan_config_.read_count
           << ",\"positions_x\":" << scan_config_.positions_x
           << ",\"rows\":" << scan_config_.rows
           << ",\"flyback\":" << scan_config_.flyback
           << ",\"flush_memory\":" << (scan_config_.flush_memory ? "true" : "false")
           << "}"
           << ",\"operation\":{"
           << "\"id\":\"" << json_escape(operation_.id) << "\""
           << ",\"name\":\"" << json_escape(operation_.name) << "\""
           << ",\"state\":\"" << json_escape(operation_.state) << "\""
           << ",\"step\":\"" << json_escape(operation_.step) << "\""
           << ",\"completed_steps\":" << operation_.completed_steps
           << ",\"total_steps\":" << operation_.total_steps
           << ",\"error\":\"" << json_escape(operation_.error) << "\"}"
           << ",\"mock\":{"
           << "\"step_duration_ms\":" << step_duration_ms_
           << ",\"fail_operation\":\"" << json_escape(fail_operation_) << "\""
           << ",\"fail_step\":" << fail_step_ << "}}}";
    return output.str();
  }

  void audit_locked(const std::string& event, const std::string& command,
                    const std::string& detail) const {
    if (audit_log_path_.empty()) { return; }
    const auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::ofstream output(audit_log_path_, std::ios::app);
    if (!output) { return; }
    output << "{\"epoch_ms\":" << epoch_ms
           << ",\"event\":\"" << json_escape(event) << "\""
           << ",\"command\":\"" << json_escape(command) << "\""
           << ",\"operation_id\":\"" << json_escape(operation_.id) << "\""
           << ",\"detail\":\"" << json_escape(detail) << "\"}\n";
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  bool stopping_ = false;
  bool work_pending_ = false;
  bool cancel_requested_ = false;
  uint64_t next_operation_id_ = 1;
  uint32_t step_duration_ms_ = 150;
  std::vector<std::string> pending_steps_;
  Operation operation_;

  double initial_temperature_c_ = 22.0;
  double temperature_c_ = 22.0;
  double target_temperature_c_ = -20.0;
  std::string camera_power_state_ = "off";
  std::string insertion_state_ = "retracted";
  bool detector_synchronized_ = false;
  bool detector_aligned_ = false;
  std::vector<std::string> links_ = {"unknown", "unknown", "unknown", "unknown"};

  ScanConfig scan_config_;
  bool scan_running_ = false;
  std::string scan_state_ = "idle";
  uint64_t scan_number_ = 0;
  uint64_t expected_frames_ = 1;
  uint64_t received_frames_ = 0;
  std::chrono::steady_clock::time_point scan_started_{};

  std::string fail_operation_;
  uint32_t fail_step_ = 0;
  bool fail_once_ = true;
  std::string audit_log_path_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: stem_daqiri_instrument_mock CONFIG\n";
    return 2;
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    const YAML::Node root = YAML::LoadFile(argv[1]);
    const YAML::Node instrument = root["instrument"];
    if (!yaml_value<bool>(instrument, "enabled", false)) {
      throw std::runtime_error("instrument.enabled must be true");
    }
    if (yaml_value<std::string>(instrument, "mode", "mock") != "mock") {
      throw std::runtime_error("mock service requires instrument.mode=mock");
    }
    stem::ControlServerConfig control;
    control.enabled = true;
    control.endpoint = yaml_value<std::string>(
        instrument, "endpoint", "ipc:///tmp/stem_daqiri_instrument.ipc");
    MockInstrumentService service(root);
    stem::ControlServer server(
        control, [&service](const std::string& request) {
          return service.handle(request);
        });
    std::cout << "mock instrument service listening on " << control.endpoint
              << "\n";
    while (!g_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    server.stop();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stem_daqiri_instrument_mock failed: " << error.what() << "\n";
    return 1;
  }
}
