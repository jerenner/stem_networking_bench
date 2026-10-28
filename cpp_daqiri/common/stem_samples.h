/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace stem {

enum class SampleEncoding { kUint16, kTwosComplement12 };
enum class SampleType { kUint16, kInt16, kFloat32 };

#ifdef __CUDACC__
__host__ __device__
#endif
inline int16_t decode_twos_complement_12(uint16_t word) {
  // Only bit 11 is the sign bit; ignore padding in bits 15:12.
  return static_cast<int16_t>(
      (static_cast<int32_t>(word & 0x0fffu) ^ 0x0800) - 0x0800);
}

inline SampleEncoding parse_sample_encoding(const std::string& value) {
  if (value == "uint16") { return SampleEncoding::kUint16; }
  if (value == "twos_complement_12") {
    return SampleEncoding::kTwosComplement12;
  }
  throw std::runtime_error(
      "stem_rx.sample_encoding must be uint16 or twos_complement_12");
}

constexpr const char* sample_encoding_name(SampleEncoding encoding) {
  return encoding == SampleEncoding::kTwosComplement12
             ? "twos_complement_12" : "uint16";
}

constexpr SampleType decoded_sample_type(SampleEncoding encoding) {
  return encoding == SampleEncoding::kTwosComplement12
             ? SampleType::kInt16 : SampleType::kUint16;
}

constexpr const char* sample_type_name(SampleType type) {
  return type == SampleType::kFloat32 ? "float32"
       : type == SampleType::kInt16 ? "int16" : "uint16";
}

constexpr size_t sample_type_bytes(SampleType type) {
  return type == SampleType::kFloat32 ? sizeof(float) : sizeof(uint16_t);
}

// Dispatch once per kernel launch, not per sample. All input types share the
// same templated correction and preview kernels.
template <typename Visitor>
void visit_samples(const void* input, SampleType type, Visitor visitor) {
  switch (type) {
    case SampleType::kUint16:
      visitor(static_cast<const uint16_t*>(input));
      return;
    case SampleType::kInt16:
      visitor(static_cast<const int16_t*>(input));
      return;
    case SampleType::kFloat32:
      visitor(static_cast<const float*>(input));
      return;
  }
  throw std::runtime_error("unsupported sample type");
}

}  // namespace stem
