#pragma once

#include <cstdint>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define STEM_NETWORKING_BENCH_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define STEM_NETWORKING_BENCH_HAS_NVTX 1
#else
#define STEM_NETWORKING_BENCH_HAS_NVTX 0
#endif

namespace holoscan::ops::profiling {

namespace color {
constexpr uint32_t kReceiver = 0xFF4E79A7;
constexpr uint32_t kProcessor = 0xFFE15759;
constexpr uint32_t kWriter = 0xFF76B7B2;
constexpr uint32_t kIo = 0xFFF28E2B;
constexpr uint32_t kCopy = 0xFF59A14F;
constexpr uint32_t kCompute = 0xFFEDC948;
constexpr uint32_t kBackpressure = 0xFFB07AA1;
}  // namespace color

class ScopedRange {
 public:
  ScopedRange(const char* message, uint32_t argb = color::kCompute) noexcept {
#if STEM_NETWORKING_BENCH_HAS_NVTX
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = argb;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = message;
    nvtxRangePushEx(&event);
    active_ = true;
#else
    (void)message;
    (void)argb;
#endif
  }

  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;

  ~ScopedRange() noexcept {
#if STEM_NETWORKING_BENCH_HAS_NVTX
    if (active_) { nvtxRangePop(); }
#endif
  }

 private:
  bool active_ = false;
};

}  // namespace holoscan::ops::profiling
