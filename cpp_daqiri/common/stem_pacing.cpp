/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "stem_pacing.h"

#include <thread>

namespace stem {

Pacer::Pacer(double target_rate_gbps)
    : target_rate_bps_(target_rate_gbps > 0.0 ? target_rate_gbps * 1e9 : 0.0) {
  start_ = clock::now();
}

void Pacer::start() {
  start_ = clock::now();
  total_bytes_ = 0;
}

void Pacer::record_burst(uint64_t burst_bytes) {
  total_bytes_ += burst_bytes;
}

void Pacer::wait_for_next_burst() const {
  if (target_rate_bps_ <= 0.0) { return; }

  // Time we should have spent to emit total_bytes_ at target_rate_bps_.
  const double target_elapsed_s =
      static_cast<double>(total_bytes_) * 8.0 / target_rate_bps_;
  const auto target_tp = start_ +
      std::chrono::nanoseconds(static_cast<int64_t>(target_elapsed_s * 1e9));

  const auto now = clock::now();
  if (now >= target_tp) { return; }
  std::this_thread::sleep_until(target_tp);
}

double Pacer::elapsed_seconds() const {
  const auto now = clock::now();
  return std::chrono::duration<double>(now - start_).count();
}

bool should_stop(const PacingConfig& cfg, const Pacer& pacer) {
  if (cfg.total_time_to_send_s >= 0.0 &&
      pacer.elapsed_seconds() >= cfg.total_time_to_send_s) {
    return true;
  }
  if (cfg.total_bytes_cap > 0 &&
      pacer.total_bytes_emitted() >= cfg.total_bytes_cap) {
    return true;
  }
  return false;
}

}  // namespace stem
