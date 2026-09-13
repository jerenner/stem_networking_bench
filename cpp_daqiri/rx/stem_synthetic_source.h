/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "stem_synthetic.h"

namespace stem {

// Immutable full-cycle packets. Each worker owns a distinct pool, retained
// until its assembler and all GPU readers have drained.
class SyntheticPacketPool {
 public:
  explicit SyntheticPacketPool(const SyntheticLayout& layout);
  ~SyntheticPacketPool();
  SyntheticPacketPool(const SyntheticPacketPool&) = delete;
  SyntheticPacketPool& operator=(const SyntheticPacketPool&) = delete;
  uint8_t* const* packets() const { return pointers_.data(); }
  uint32_t size() const { return static_cast<uint32_t>(pointers_.size()); }
  uint64_t bytes() const { return bytes_; }

 private:
  uint8_t* device_ = nullptr;
  uint64_t bytes_ = 0;
  std::vector<uint8_t*> pointers_;
};

// Checks the assembled uint16 plane before processing, including unused tiles
// for partial source masks. The counter must be zeroed by the caller.
void validate_synthetic_frames(const uint16_t* frames, uint64_t first_frame,
                               uint32_t count, const SyntheticLayout& layout,
                               unsigned long long* mismatches, cudaStream_t stream);

}  // namespace stem
