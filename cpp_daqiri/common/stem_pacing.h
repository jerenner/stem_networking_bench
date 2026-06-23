/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Token-bucket pacing for the daqiri-based STEM TX. Given a target rate in
 * Gbps and an emitted byte count, returns the wall-clock instant at which
 * the next burst is allowed to ship. A target_rate_gbps <= 0 disables
 * pacing (unbounded send).
 *
 * Pacing model:
 *
 *   per-burst sleep = max(0, bytes_in_burst / target_bps - elapsed_since_start)
 *
 * This is a simple leaky-bucket style scheme: each burst consumes
 * bytes_in_burst / target_bps seconds of token budget; the next burst waits
 * until enough tokens have refilled. The total emitted bytes is checked
 * against either a byte limit or a wall-clock duration limit (whichever
 * stop condition fires first).
 */
#pragma once

#include <chrono>
#include <cstdint>

namespace stem {

struct PacingConfig {
  // Target TX rate in Gbps (gigabits per second of on-wire bytes including
  // Eth+IP+UDP headers + STEM payload). <= 0 means "send as fast as possible".
  double target_rate_gbps = 0.0;

  // Stop after this many wall-clock seconds. < 0 means "run forever".
  // 0 means "stop immediately" (useful for self-test).
  double total_time_to_send_s = 10.0;

  // Optional hard byte cap on top of the time cap. <= 0 means "no byte cap".
  uint64_t total_bytes_cap = 0;
};

class Pacer {
 public:
  using clock = std::chrono::steady_clock;
  using time_point = clock::time_point;

  // `target_rate_gbps` <= 0 disables sleep entirely.
  explicit Pacer(double target_rate_gbps);

  // Reset the start time and emitted-bytes counter.
  void start();

  // Returns true if pacing is enabled (target_rate_gbps > 0).
  bool enabled() const { return target_rate_bps_ > 0; }

  // Account for a burst that just emitted `burst_bytes` bytes. Caller must
  // also call wait_for_next_burst() if pacing is enabled.
  void record_burst(uint64_t burst_bytes);

  // Sleep until the next burst is allowed. No-op if pacing is disabled.
  void wait_for_next_burst() const;

  uint64_t total_bytes_emitted() const { return total_bytes_; }
  double elapsed_seconds() const;

 private:
  double   target_rate_bps_;  // bits/sec on the wire
  time_point start_;
  uint64_t total_bytes_ = 0;
};

// Returns true once any stop condition in the PacingConfig has fired.
bool should_stop(const PacingConfig& cfg, const Pacer& pacer);

}  // namespace stem
