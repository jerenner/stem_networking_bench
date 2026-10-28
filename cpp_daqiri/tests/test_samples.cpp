/* SPDX-License-Identifier: Apache-2.0 */
#include "stem_samples.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

void require(bool condition) {
  if (!condition) { throw std::runtime_error("sample encoding test failed"); }
}

int main() {
  using namespace stem;
  // All ADC codes and all padding patterns, including sign-extended input.
  for (uint32_t word = 0; word <= 0xffff; ++word) {
    const int32_t code = word % 4096;
    const int32_t expected = code < 2048 ? code : code - 4096;
    require(decode_twos_complement_12(static_cast<uint16_t>(word)) == expected);
  }
  require(parse_sample_encoding("uint16") == SampleEncoding::kUint16);
  require(parse_sample_encoding("twos_complement_12") ==
          SampleEncoding::kTwosComplement12);
  bool rejected = false;
  try { parse_sample_encoding("int12"); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected);
  require(decoded_sample_type(SampleEncoding::kUint16) == SampleType::kUint16);
  require(decoded_sample_type(SampleEncoding::kTwosComplement12) == SampleType::kInt16);
  require(sample_type_bytes(SampleType::kInt16) == 2);
  require(sample_type_bytes(SampleType::kFloat32) == 4);
  require(std::string(sample_type_name(SampleType::kInt16)) == "int16");

  const int16_t signed_values[] = {-2048, -1, 0, 2047};
  const uint16_t unsigned_values[] = {0, 32768, 65535};
  const float float_values[] = {-0.5f, 1.0f};
  const auto check_visit = [](const void* data, SampleType type,
                              uint32_t count, float expected) {
    float sum = 0;
    visit_samples(data, type, [&](const auto* values) {
      using Value = std::remove_cv_t<std::remove_pointer_t<decltype(values)>>;
      require(type == SampleType::kInt16 ? std::is_same_v<Value, int16_t>
            : type == SampleType::kUint16 ? std::is_same_v<Value, uint16_t>
                                         : std::is_same_v<Value, float>);
      for (uint32_t i = 0; i < count; ++i) { sum += values[i]; }
    });
    require(sum == expected);
  };
  check_visit(signed_values, SampleType::kInt16, 4, -2);
  check_visit(unsigned_values, SampleType::kUint16, 3, 98303);
  check_visit(float_values, SampleType::kFloat32, 2, 0.5f);
  std::cout << "PASS all 65536 ADC words, sample types and signed dispatch\n";
  return EXIT_SUCCESS;
}
