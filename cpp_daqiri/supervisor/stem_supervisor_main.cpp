/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <yaml-cpp/yaml.h>
#include <zmq.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "stem_control_server.h"

extern char** environ;

namespace {

constexpr int kRestartExitCode = 75;

volatile std::sig_atomic_t g_stop = 0;

void signal_handler(int) { g_stop = 1; }

std::string getenv_or(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}

double getenv_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) { return fallback; }
  return std::stod(value);
}

bool getenv_bool(const char* name, bool fallback) {
  const std::string value = getenv_or(name, fallback ? "true" : "false");
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

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

void replace_all(std::string& value, const std::string& from,
                 const std::string& to) {
  if (from.empty() || from == to) { return; }
  size_t position = 0;
  while ((position = value.find(from, position)) != std::string::npos) {
    value.replace(position, from.size(), to);
    position += to.size();
  }
}

bool scalar_is_number(const std::string& value) {
  if (value.empty()) { return false; }
  char* end = nullptr;
  std::strtod(value.c_str(), &end);
  return end != value.c_str() && end != nullptr && *end == '\0';
}

std::string yaml_to_json(const YAML::Node& node) {
  if (!node || node.IsNull()) { return "null"; }
  if (node.IsMap()) {
    std::ostringstream output;
    output << "{";
    bool first = true;
    for (const auto& entry : node) {
      if (!first) { output << ","; }
      first = false;
      output << "\"" << json_escape(entry.first.as<std::string>()) << "\":"
             << yaml_to_json(entry.second);
    }
    output << "}";
    return output.str();
  }
  if (node.IsSequence()) {
    std::ostringstream output;
    output << "[";
    for (size_t index = 0; index < node.size(); ++index) {
      if (index != 0) { output << ","; }
      output << yaml_to_json(node[index]);
    }
    output << "]";
    return output.str();
  }
  const std::string scalar = node.as<std::string>();
  if (scalar == "true" || scalar == "false" || scalar == "null" ||
      scalar_is_number(scalar)) {
    return scalar;
  }
  return "\"" + json_escape(scalar) + "\"";
}

template <typename T>
T yaml_map_value(const YAML::Node& map, const char* key, const T& fallback) {
  if (!map || !map.IsMap()) { return fallback; }
  const YAML::Node value = map[key];
  if (!value || value.IsNull()) { return fallback; }
  return value.as<T>(fallback);
}

YAML::Node yaml_map_node(const YAML::Node& map, const char* key) {
  if (!map || !map.IsMap()) { return {}; }
  const YAML::Node value = map[key];
  return value && value.IsMap() ? value : YAML::Node{};
}

void write_yaml_atomic(const YAML::Node& root,
                       const std::filesystem::path& path) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  YAML::Emitter emitter;
  emitter << root;
  std::ofstream output(temporary);
  if (!output) {
    throw std::runtime_error("cannot create child config " + temporary.string());
  }
  output << "%YAML 1.2\n---\n" << emitter.c_str() << "\n";
  output.close();
  if (!output) {
    throw std::runtime_error("failed writing child config " + temporary.string());
  }
  std::filesystem::rename(temporary, path);
}

class PersistentSupervisor {
 public:
  PersistentSupervisor(std::string initial_config,
                       std::vector<std::string> rx_arguments)
      : rx_arguments_(std::move(rx_arguments)),
        rx_binary_(getenv_or("STEM_DAQIRI_RX_BIN",
                             "/opt/stem_daqiri/bin/stem_daqiri_rx")),
        child_endpoint_(getenv_or(
            "STEM_DAQIRI_CHILD_CONTROL_ENDPOINT",
            "ipc:///tmp/stem_daqiri_rx_control.ipc")),
        child_config_(getenv_or("STEM_DAQIRI_CHILD_CONFIG",
                                "/tmp/stem_daqiri_child.yaml")),
        restart_delay_seconds_(
            getenv_double("STEM_DAQIRI_RESTART_DELAY_SECONDS", 1.0)) {
    const YAML::Node original_root = YAML::LoadFile(initial_config);
    if (!original_root["control"] ||
        !original_root["control"]["enabled"].as<bool>(false)) {
      throw std::runtime_error(
          "persistent supervisor requires control.enabled=true");
    }
    public_endpoint_ = getenv_or(
        "STEM_DAQIRI_SUPERVISOR_ENDPOINT",
        original_root["control"]["endpoint"].as<std::string>(
            "tcp://*:5557"));
    runtime_config_ = getenv_or(
        "STEM_DAQIRI_RUNTIME_CONFIG",
        original_root["control"]["runtime_config_path"].as<std::string>(
            "/tmp/stem_daqiri_runtime.yaml"));
    start_acquisition_on_launch_ =
        original_root["control"]["start_acquisition"].as<bool>(true);
    prepare_child_config(initial_config, child_config_);
    current_child_config_ = child_config_;
    cached_state_ = initial_state(original_root);
  }

  ~PersistentSupervisor() { stop_child(true); }

  const std::string& public_endpoint() const { return public_endpoint_; }

  void start_initial() {
    if (start_acquisition_on_launch_ &&
        !getenv_bool("STEM_DAQIRI_START_STOPPED", false)) {
      start_child(current_child_config_);
    }
  }

  void poll() {
    std::lock_guard<std::mutex> lock(mu_);
    if (child_pid_ <= 0 || manual_reap_) { return; }
    int status = 0;
    const pid_t result = waitpid(child_pid_, &status, WNOHANG);
    if (result == 0) { return; }
    if (result < 0) {
      if (errno != ECHILD) {
        last_error_ = std::string("waitpid failed: ") + std::strerror(errno);
      }
      child_pid_ = -1;
      lifecycle_ = "error";
      return;
    }

    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) :
                                             128 + WTERMSIG(status);
    child_pid_ = -1;
    last_exit_code_ = exit_code;
    if (operator_stop_requested_) {
      operator_stop_requested_ = false;
      lifecycle_ = "stopped";
      last_error_.clear();
    } else if (exit_code == kRestartExitCode && !stopping_) {
      lifecycle_ = "restarting";
      restart_due_ = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(static_cast<int64_t>(
                         restart_delay_seconds_ * 1000.0));
      restart_scheduled_ = true;
      if (std::filesystem::is_regular_file(runtime_config_)) {
        current_child_config_ = runtime_config_;
      } else {
        lifecycle_ = "error";
        last_error_ = "restart config is missing: " + runtime_config_;
        restart_scheduled_ = false;
      }
    } else {
      lifecycle_ = exit_code == 0 ? "stopped" : "error";
      if (exit_code != 0) {
        last_error_ = "RX exited with status " + std::to_string(exit_code);
      }
    }
  }

  void launch_scheduled_restart() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!restart_scheduled_ || child_pid_ > 0 ||
        std::chrono::steady_clock::now() < restart_due_) {
      return;
    }
    restart_scheduled_ = false;
    start_child_locked(current_child_config_);
  }

  std::string handle(const std::string& request_text) {
    try {
      const YAML::Node request = YAML::Load(request_text);
      const std::string command = request["command"].as<std::string>(
          std::string("get_state"));
      if (command == "get_state") { return state(); }
      if (command == "start_acquisition") {
        std::lock_guard<std::mutex> lock(mu_);
        if (child_pid_ > 0 || restart_scheduled_) {
          throw std::runtime_error("acquisition is already running or restarting");
        }
        start_child_locked(current_child_config_);
        return state_locked("acquisition start requested");
      }
      if (command == "stop_acquisition" || command == "shutdown") {
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (child_pid_ <= 0) {
            lifecycle_ = "stopped";
            return state_locked("acquisition is already stopped");
          }
          operator_stop_requested_ = true;
          lifecycle_ = "stopping";
        }
        try {
          const std::string response = forward_to_child(
              "{\"command\":\"shutdown\"}", 2000);
          std::lock_guard<std::mutex> lock(mu_);
          return decorate_state_locked(response, "acquisition stop requested");
        } catch (const std::exception&) {
          std::lock_guard<std::mutex> lock(mu_);
          if (child_pid_ <= 0) {
            lifecycle_ = "stopped";
            operator_stop_requested_ = false;
            return state_locked("acquisition is already stopped");
          }
          if (kill(child_pid_, SIGINT) != 0 && errno != ESRCH) {
            throw std::runtime_error(
                std::string("failed to signal starting RX child: ") +
                std::strerror(errno));
          }
          lifecycle_ = "stopping";
          return state_locked(
              "acquisition stop requested while RX was starting");
        }
      }
      if (command == "shutdown_supervisor") {
        g_stop = 1;
        stop_child(true);
        return state("supervisor shutdown requested");
      }

      std::string child_request = request_text;
      if (command == "stage_restart" && request["updates"]) {
        YAML::Node forwarded = YAML::Clone(request);
        forwarded["updates"]["control"]["enabled"] = true;
        forwarded["updates"]["control"]["endpoint"] = child_endpoint_;
        forwarded["updates"]["control"]["runtime_config_path"] =
            runtime_config_;
        YAML::Emitter emitter;
        emitter << forwarded;
        child_request = emitter.c_str();
      } else {
        replace_all(child_request, public_endpoint_, child_endpoint_);
      }
      const std::string response = forward_to_child(child_request, 2500);
      {
        std::lock_guard<std::mutex> lock(mu_);
        cache_if_state_locked(response);
        if (command == "restart") { lifecycle_ = "restarting"; }
      }
      return state_from_response(response);
    } catch (const std::exception& error) {
      return std::string("{\"ok\":false,\"error\":\"") +
             json_escape(error.what()) + "\"}";
    }
  }

  void stop_child(bool supervisor_shutdown) {
    pid_t pid = -1;
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = supervisor_shutdown;
      restart_scheduled_ = false;
      manual_reap_ = true;
      pid = child_pid_;
    }
    if (pid <= 0) {
      std::lock_guard<std::mutex> lock(mu_);
      manual_reap_ = false;
      return;
    }
    try {
      forward_to_child("{\"command\":\"shutdown\"}", 1000);
    } catch (...) {
      // RX installs a SIGINT handler that drains and releases DAQIRI resources.
      kill(pid, SIGINT);
    }
    for (int attempt = 0; attempt < 300; ++attempt) {
      int status = 0;
      const pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid || (result < 0 && errno == ECHILD)) {
        std::lock_guard<std::mutex> lock(mu_);
        child_pid_ = -1;
        lifecycle_ = "stopped";
        manual_reap_ = false;
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    std::lock_guard<std::mutex> lock(mu_);
    child_pid_ = -1;
    lifecycle_ = "stopped";
    manual_reap_ = false;
  }

 private:
  std::string initial_state(const YAML::Node& root) const {
    const YAML::Node burst = root["burst_writer"];
    const YAML::Node thinned = root["thinned_stream"];
    const YAML::Node burst_threshold = yaml_map_node(burst, "threshold");
    const YAML::Node thinned_threshold = yaml_map_node(thinned, "threshold");
    const bool burst_enabled = yaml_map_value(burst, "enabled", false);
    const std::string burst_stage = yaml_map_value<std::string>(
        burst, "processing_stage", "corrected");
    const uint32_t burst_buckets =
        yaml_map_value<uint32_t>(burst, "buckets_per_capture", 1);
    const bool thinned_enabled = yaml_map_value(thinned, "enabled", false);
    std::ostringstream state;
    state << "{\"ok\":true,\"schema\":\"stem.control.v1\",\"message\":\"\""
          << ",\"acquisition\":{\"running\":false,\"restart_pending\":false,"
             "\"restart_requested\":false}"
          << ",\"burst_writer\":{\"capability_enabled\":"
          << (burst_enabled ? "true" : "false")
          << ",\"armed\":false,\"busy\":false,\"output_float32\":"
          << (burst_stage == "raw" ? "false" : "true")
          << ",\"capacity_buckets\":"
          << burst_buckets
          << ",\"processing_stage\":\"" << json_escape(burst_stage) << "\""
          << ",\"filepath_template\":\""
          << json_escape(yaml_map_value<std::string>(
                 burst, "filepath_template",
                 "/data/stem_burst_rx{receiver}_{capture}_{stage}.h5"))
          << "\",\"dataset_name\":\""
          << json_escape(
                 yaml_map_value<std::string>(burst, "dataset_name", "/frames"))
          << "\",\"buckets_per_capture\":"
          << burst_buckets
          << ",\"capture_count\":"
          << yaml_map_value<uint64_t>(burst, "capture_count", 1)
          << ",\"rearm_after_write\":"
          << (yaml_map_value(burst, "rearm_after_write", true) ? "true"
                                                                  : "false")
          << ",\"strict_complete\":"
          << (yaml_map_value(burst, "strict_complete", true) ? "true"
                                                                 : "false")
          << ",\"threshold\":{\"zlp\":"
          << yaml_map_value(burst_threshold, "zlp", 0.0)
          << ",\"core_loss\":"
          << yaml_map_value(burst_threshold, "core_loss", 0.0)
          << "},\"stats\":{\"captures_started\":0,\"captures_written\":0,"
             "\"buckets_captured\":0,\"buckets_skipped_busy\":0,"
             "\"rejected_incomplete\":0,\"aborted\":0,\"errors\":0}}"
          << ",\"thinned_stream\":{\"capability_enabled\":"
          << (thinned_enabled ? "true" : "false")
          << ",\"publishing\":"
          << (yaml_map_value(thinned, "start_publishing", true) ? "true"
                                                                    : "false")
          << ",\"endpoint\":\""
          << json_escape(yaml_map_value<std::string>(
                 thinned, "endpoint", "tcp://*:5556"))
          << "\",\"queue_depth\":"
          << yaml_map_value<uint32_t>(thinned, "queue_depth", 2)
          << ",\"processing_stage\":\""
          << json_escape(yaml_map_value<std::string>(
                 thinned, "processing_stage", "corrected"))
          << "\",\"topic_prefix\":\""
          << json_escape(
                 yaml_map_value<std::string>(thinned, "topic_prefix", "stem"))
          << "\",\"total_refresh_hz\":"
          << yaml_map_value(thinned, "total_refresh_hz", 10.0)
          << ",\"representative_frame_index\":"
          << yaml_map_value<uint32_t>(thinned, "representative_frame_index", 64)
          << ",\"include_representative_frame\":"
          << (yaml_map_value(thinned, "include_representative_frame", true)
                  ? "true"
                  : "false")
          << ",\"include_bucket_sum\":"
          << (yaml_map_value(thinned, "include_bucket_sum", true) ? "true"
                                                                      : "false")
          << ",\"threshold\":{\"zlp\":"
          << yaml_map_value(thinned_threshold, "zlp", 0.0)
          << ",\"core_loss\":"
          << yaml_map_value(thinned_threshold, "core_loss", 0.0)
          << "},\"stats\":{\"products_queued\":0,\"products_published\":0,"
             "\"products_coalesced\":0,\"dropped_no_buffer\":0,"
             "\"send_errors\":0}}"
          << ",\"effective_config\":" << yaml_to_json(root)
          << ",\"pending_config\":" << yaml_to_json(root) << "}";
    return state.str();
  }

  void prepare_child_config(const std::string& source,
                            const std::string& destination) {
    YAML::Node child = YAML::LoadFile(source);
    child["control"]["enabled"] = true;
    child["control"]["endpoint"] = child_endpoint_;
    child["control"]["runtime_config_path"] = runtime_config_;
    write_yaml_atomic(child, destination);
  }

  void remove_stale_ipc_socket() const {
    constexpr const char* prefix = "ipc://";
    if (child_endpoint_.rfind(prefix, 0) == 0) {
      std::error_code error;
      std::filesystem::remove(child_endpoint_.substr(std::strlen(prefix)), error);
    }
  }

  void start_child(const std::string& config) {
    std::lock_guard<std::mutex> lock(mu_);
    start_child_locked(config);
  }

  void start_child_locked(const std::string& config) {
    if (child_pid_ > 0) {
      throw std::runtime_error("acquisition is already running");
    }
    remove_stale_ipc_socket();
    std::vector<std::string> arguments;
    arguments.reserve(rx_arguments_.size() + 2);
    arguments.push_back(rx_binary_);
    arguments.push_back(config);
    arguments.insert(arguments.end(), rx_arguments_.begin(), rx_arguments_.end());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (auto& argument : arguments) { argv.push_back(argument.data()); }
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int result = posix_spawn(&pid, rx_binary_.c_str(), nullptr, nullptr,
                                   argv.data(), environ);
    if (result != 0) {
      lifecycle_ = "error";
      last_error_ = std::string("posix_spawn failed: ") + std::strerror(result);
      throw std::runtime_error(last_error_);
    }
    child_pid_ = pid;
    operator_stop_requested_ = false;
    lifecycle_ = "starting";
    last_error_.clear();
    std::cerr << "DAQ supervisor started RX pid " << child_pid_ << " using "
              << config << "\n";
  }

  std::string forward_to_child(const std::string& request,
                               int timeout_ms) const {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (child_pid_ <= 0) {
        throw std::runtime_error("acquisition is stopped");
      }
    }
    void* context = zmq_ctx_new();
    if (!context) { throw std::runtime_error("child control context creation failed"); }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
      zmq_ctx_term(context);
      throw std::runtime_error("child control socket creation failed");
    }
    const int linger = 0;
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    if (zmq_connect(socket, child_endpoint_.c_str()) != 0 ||
        zmq_send(socket, request.data(), request.size(), 0) < 0) {
      const std::string error = zmq_strerror(zmq_errno());
      zmq_close(socket);
      zmq_ctx_term(context);
      throw std::runtime_error("child control send failed: " + error);
    }
    std::vector<char> buffer(4 * 1024 * 1024);
    const int received = zmq_recv(socket, buffer.data(), buffer.size(), 0);
    if (received < 0) {
      const std::string error = zmq_strerror(zmq_errno());
      zmq_close(socket);
      zmq_ctx_term(context);
      throw std::runtime_error("child control response failed: " + error);
    }
    if (static_cast<size_t>(received) >= buffer.size()) {
      zmq_close(socket);
      zmq_ctx_term(context);
      throw std::runtime_error("child control response exceeds 4 MiB");
    }
    std::string response(buffer.data(), static_cast<size_t>(received));
    zmq_close(socket);
    zmq_ctx_term(context);
    return response;
  }

  std::string state(const std::string& message = {}) {
    try {
      const std::string response = forward_to_child(
          "{\"command\":\"get_state\"}", 1200);
      std::lock_guard<std::mutex> lock(mu_);
      cache_if_state_locked(response);
      if (lifecycle_ == "starting") { lifecycle_ = "running"; }
      return decorate_state_locked(response, message);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mu_);
      if (cached_state_.empty()) {
        return std::string("{\"ok\":false,\"error\":\"") +
               (child_pid_ > 0 ? "acquisition is starting" :
                                 "no acquisition state has been cached") +
               "\"}";
      }
      return decorate_state_locked(cached_state_, message);
    }
  }

  std::string state_from_response(const std::string& response) {
    std::lock_guard<std::mutex> lock(mu_);
    return decorate_state_locked(response, {});
  }

  void cache_if_state_locked(const std::string& response) {
    if (response.find("\"schema\":\"stem.control.v1\"") !=
        std::string::npos) {
      cached_state_ = response;
    }
  }

  std::string decorate_state_locked(const std::string& raw,
                                    const std::string& message) const {
    std::string response = raw;
    replace_all(response, child_endpoint_, public_endpoint_);
    const bool running = child_pid_ > 0 &&
                         (lifecycle_ == "starting" || lifecycle_ == "running");
    if (!running) {
      replace_all(response, "\"acquisition\":{\"running\":true",
                  "\"acquisition\":{\"running\":false");
    } else {
      replace_all(response, "\"acquisition\":{\"running\":false",
                  "\"acquisition\":{\"running\":true");
    }
    if (response.empty() || response.back() != '}') { return response; }
    response.pop_back();
    response += ",\"supervisor\":{\"state\":\"" +
                json_escape(lifecycle_) + "\",\"acquisition_running\":" +
                (running ? "true" : "false") + ",\"pid\":" +
                std::to_string(child_pid_) + ",\"public_endpoint\":\"" +
                json_escape(public_endpoint_) + "\",\"last_exit_code\":" +
                std::to_string(last_exit_code_) + ",\"last_error\":\"" +
                json_escape(last_error_) + "\"}";
    if (!message.empty()) {
      response += ",\"supervisor_message\":\"" + json_escape(message) + "\"";
    }
    response += "}";
    return response;
  }

  std::string state_locked(const std::string& message) const {
    if (cached_state_.empty()) {
      const bool running = child_pid_ > 0 &&
                           (lifecycle_ == "starting" || lifecycle_ == "running");
      return std::string("{\"ok\":true,\"schema\":\"stem.supervisor.v1\",") +
             "\"message\":\"" + json_escape(message) +
             "\",\"supervisor\":{\"state\":\"" +
             json_escape(lifecycle_) + "\",\"acquisition_running\":" +
             (running ? "true" : "false") + "," +
             "\"pid\":" + std::to_string(child_pid_) + "}}";
    }
    return decorate_state_locked(cached_state_, message);
  }

  std::vector<std::string> rx_arguments_;
  std::string rx_binary_;
  std::string public_endpoint_;
  std::string child_endpoint_;
  std::string child_config_;
  std::string runtime_config_;
  double restart_delay_seconds_ = 1.0;
  mutable std::mutex mu_;
  pid_t child_pid_ = -1;
  std::string lifecycle_ = "stopped";
  std::string current_child_config_;
  std::string cached_state_;
  std::string last_error_;
  int last_exit_code_ = 0;
  bool stopping_ = false;
  bool manual_reap_ = false;
  bool restart_scheduled_ = false;
  bool start_acquisition_on_launch_ = true;
  bool operator_stop_requested_ = false;
  std::chrono::steady_clock::time_point restart_due_{};
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: stem_daqiri_supervisor CONFIG [RX_ARGUMENTS...]\n";
    return 2;
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    std::vector<std::string> rx_arguments;
    for (int index = 2; index < argc; ++index) {
      rx_arguments.emplace_back(argv[index]);
    }
    PersistentSupervisor supervisor(argv[1], std::move(rx_arguments));
    supervisor.start_initial();
    stem::ControlServerConfig control;
    control.enabled = true;
    control.endpoint = supervisor.public_endpoint();
    stem::ControlServer server(control, [&supervisor](const std::string& request) {
      return supervisor.handle(request);
    });
    std::cout << "persistent DAQ supervisor listening on " << control.endpoint
              << "\n";
    while (!g_stop) {
      supervisor.poll();
      supervisor.launch_scheduled_restart();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    server.stop();
    supervisor.stop_child(true);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stem_daqiri_supervisor failed: " << error.what() << "\n";
    return 1;
  }
}
