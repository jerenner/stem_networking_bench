/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "stem_control_server.h"

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef STEM_DAQIRI_HAVE_ZMQ
#include <zmq.h>
#endif

namespace stem {

struct ControlServer::Impl {
  Impl(const ControlServerConfig& value, Handler callback)
      : config(value), handler(std::move(callback)) {
    if (!config.enabled) { return; }
#ifndef STEM_DAQIRI_HAVE_ZMQ
    throw std::runtime_error(
        "control.enabled=true requires ZeroMQ support in stem_daqiri_rx");
#else
    if (config.endpoint.empty()) {
      throw std::runtime_error("control.endpoint cannot be empty");
    }
    worker = std::thread(&Impl::run, this);
    std::unique_lock<std::mutex> lock(mu);
    initialized_cv.wait(lock, [&] { return initialized; });
    if (!initialization_error.empty()) {
      lock.unlock();
      if (worker.joinable()) { worker.join(); }
      throw std::runtime_error(initialization_error);
    }
#endif
  }

  ~Impl() { stop(); }

  void stop() {
    if (!config.enabled || stopping.exchange(true)) { return; }
    if (worker.joinable()) { worker.join(); }
  }

  void set_initialized(const std::string& error = {}) {
    {
      std::lock_guard<std::mutex> lock(mu);
      initialized = true;
      initialization_error = error;
    }
    initialized_cv.notify_all();
  }

  void run() {
#ifndef STEM_DAQIRI_HAVE_ZMQ
    set_initialized("ZeroMQ support is not compiled");
#else
    void* context = zmq_ctx_new();
    if (!context) {
      set_initialized("control zmq_ctx_new failed");
      return;
    }
    void* socket = zmq_socket(context, ZMQ_REP);
    if (!socket) {
      set_initialized(std::string("control zmq_socket failed: ") +
                      zmq_strerror(zmq_errno()));
      zmq_ctx_term(context);
      return;
    }
    const int linger = 0;
    const int timeout_ms = 200;
    zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
    if (zmq_bind(socket, config.endpoint.c_str()) != 0) {
      const std::string error =
          std::string("ZeroMQ control bind failed for ") + config.endpoint +
          ": " + zmq_strerror(zmq_errno());
      zmq_close(socket);
      zmq_ctx_term(context);
      set_initialized(error);
      return;
    }
    set_initialized();

    while (!stopping.load()) {
      zmq_msg_t request;
      zmq_msg_init(&request);
      const int received = zmq_msg_recv(&request, socket, 0);
      if (received < 0) {
        zmq_msg_close(&request);
        if (zmq_errno() == EAGAIN || zmq_errno() == EINTR) { continue; }
        break;
      }
      const std::string input(
          static_cast<const char*>(zmq_msg_data(&request)),
          static_cast<size_t>(zmq_msg_size(&request)));
      zmq_msg_close(&request);

      std::string response;
      try {
        response = handler(input);
      } catch (const std::exception& error) {
        response =
            std::string("{\"ok\":false,\"error\":\"control handler: ") +
            error.what() + "\"}";
      }
      if (zmq_send(socket, response.data(), response.size(), 0) < 0 &&
          zmq_errno() != EAGAIN && zmq_errno() != EINTR) {
        break;
      }
    }
    zmq_close(socket);
    zmq_ctx_term(context);
#endif
  }

  ControlServerConfig config;
  Handler handler;
  std::atomic<bool> stopping{false};
  std::thread worker;
  std::mutex mu;
  std::condition_variable initialized_cv;
  bool initialized = false;
  std::string initialization_error;
};

ControlServer::ControlServer(const ControlServerConfig& config, Handler handler)
    : impl_(std::make_unique<Impl>(config, std::move(handler))) {}

ControlServer::~ControlServer() = default;

bool ControlServer::enabled() const { return impl_->config.enabled; }

void ControlServer::stop() { impl_->stop(); }

}  // namespace stem
