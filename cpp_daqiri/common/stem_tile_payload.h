/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>
#include <stdexcept>

#include "stem_packet.h"

namespace stem {

// Zero denotes an omitted payload_size; explicit YAML values must be positive.
constexpr uint16_t resolve_tile_payload_size(uint16_t configured_size,
                                            bool duplicate_prefix) {
  return configured_size != 0
             ? configured_size
             : static_cast<uint16_t>(duplicate_prefix ? STEM_PAYLOAD_SIZE
                                                      : TILE_PAYLOAD_BYTES);
}

inline void validate_tile_payload_size(uint16_t size) {
  if (size == 0 || size > TILE_PAYLOAD_BYTES || size % sizeof(uint16_t) != 0) {
    throw std::runtime_error(
        "stem_rx.payload_size must be a positive even number of detector-data "
        "bytes after the STEM header, no greater than 8192");
  }
}

}  // namespace stem
