/* SPDX-License-Identifier: Apache-2.0 */
#include "stem_synthetic.h"

#include <array>
#include <iostream>
#include <limits>
#include <vector>

namespace {
void require(bool ok, const char* message) {
  if (!ok) { throw std::runtime_error(message); }
}
uint16_t le16(const uint8_t* p) { return p[0] | (p[1] << 8); }
uint16_t be16(const uint8_t* p) { return (p[0] << 8) | p[1]; }

void check_layout(uint32_t mask, bool legacy) {
  stem::SyntheticLayout layout;
  layout.source_mask = mask;
  layout.legacy_payload = legacy;
  layout.receiver = 7;
  std::vector<unsigned> ids;
  for (unsigned id = 0; id < 8; ++id) {
    if (mask & (1u << id)) { ids.push_back(id); }
  }
  require(layout.source_count() == ids.size(), "source count");
  require(layout.packet_bytes() == (legacy ? 7786u : 8298u), "packet length");
  require(layout.stride_bytes() % 64 == 0, "packet alignment");
  require(layout.stride_bytes() >= layout.packet_bytes(), "stride covers packet");
  require(layout.cycle_packets() == 128 * layout.packets_per_frame(), "cycle length");
  for (unsigned frame : {0u, 1u, 63u, 64u, 127u}) {
    std::vector<bool> seen(960, false);
    unsigned ignored = 0, kept = 0;
    for (unsigned local = 0; local < layout.packets_per_frame(); ++local) {
      const unsigned packet = frame * layout.packets_per_frame() + local;
      std::array<uint8_t, 107> header{};
      header.back() = 0xcc;
      layout.write_header(header.data(), packet);
      const auto* p = header.data();
      require(header.back() == 0xcc, "header write exceeds 106 bytes");
      require(be16(p + 12) == 0x0800 && p[14] == 0x45 && p[23] == 17,
              "IPv4/UDP framing");
      require(be16(p + 16) == layout.packet_bytes() - 14, "IPv4 length");
      uint32_t checksum = 0;
      for (unsigned i = 14; i < 34; i += 2) { checksum += be16(p + i); }
      while (checksum >> 16) { checksum = (checksum & 65535u) + (checksum >> 16); }
      require(checksum == 65535u, "IPv4 checksum");
      require(be16(p + 36) == 23130 && be16(p + 38) == layout.packet_bytes() - 34,
              "UDP port/length");
      require(be16(p + 40) == 0, "IPv4 UDP checksum disabled");
      require(le16(p + 42) == 0x5a5a && le16(p + 50) == 0xa5a5, "STEM markers");
      require(le16(p + 44) == frame, "frame count");
      const auto row = le16(p + 46);
      const auto source = le16(p + 48);
      require(row / 128 == frame && row % 128 == local / ids.size(), "row field");
      require(source == ids[local % ids.size()], "sparse source ID ordering");
      for (unsigned i = 52; i < 106; i += 2) {
        require(le16(p + i) == frame, "16-bit loop counter fill");
      }
      if (row % 128 >= 120) { ++ignored; continue; }
      const unsigned tile = (local % ids.size()) * 120 + row % 128;
      require(layout.tile(packet) == tile && tile < 960 && !seen[tile], "unique tile mapping");
      seen[tile] = true;
      ++kept;
    }
    require(kept == ids.size() * 120, "kept tiles");
    require(ignored == (legacy ? ids.size() * 8 : 0), "legacy discard count");
    for (unsigned tile = 0; tile < 960; ++tile) {
      require(seen[tile] == (tile < ids.size() * 120), "compact sparse coverage");
    }
  }
}

void check_patterns_and_geometry() {
  stem::SyntheticLayout a, b;
  b.receiver = 1;
  require(a.sample(0, 0, 0) != b.sample(0, 0, 0), "receiver pattern independence");
  require(a.sample(0, 0, 0) != a.sample(1, 0, 0), "frame pattern independence");
  require(a.sample(0, 0, 0) != a.sample(0, 1, 0), "tile pattern independence");
  a.pattern = stem::SyntheticPattern::kWalkingDot;
  unsigned dots = 0;
  for (unsigned i = 0; i < 4096; ++i) { dots += a.sample(1, 2, i) == 20000; }
  require(dots == 1, "one walking dot per native tile");

  std::vector<uint8_t> coverage(1024 * 3840, 0);
  for (unsigned tile = 0; tile < 960; ++tile) {
    const bool zlp = tile < 192;
    const unsigned local = zlp ? tile : tile - 192;
    const unsigned width = zlp ? 32 : 128;
    const unsigned height = zlp ? 128 : 32;
    const unsigned row = (local / 24) * height;
    const unsigned col = (local % 24) * width + (zlp ? 0 : 768);
    for (unsigned i = 0; i < 4096; ++i) {
      const unsigned index = (row + i / width) * 3840 + col + i % width;
      require(index < coverage.size() && !coverage[index], "tile bounds/overlap");
      coverage[index] = 1;
    }
  }
  for (auto covered : coverage) { require(covered == 1, "full sensor coverage"); }
}

void check_config_and_clock() {
  stem::SyntheticConfig cfg;
  cfg.validate();
  const double bucket_time = cfg.scheduled_seconds(128 * 960);
  require(std::abs(bucket_time - (128.0 * 960 * 8298 * 8 / 100e9)) < 1e-12,
          "rate counts full packets, not payload alone");
  require(cfg.lag_seconds(128 * 960, bucket_time / 2) == 0, "ahead of schedule");
  require(std::abs(cfg.lag_seconds(128 * 960, bucket_time + 0.2) - 0.2) < 1e-12,
          "backpressure must remain visible as schedule debt");
  cfg.maximum_rate = true;
  require(cfg.lag_seconds(0, 10) == 0, "unpaced mode has no schedule debt");
  auto rejects = [](stem::SyntheticConfig bad) {
    try { bad.validate(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid configuration accepted");
  };
  auto bad = cfg; bad.layout.source_mask = 0; rejects(bad);
  bad = cfg; bad.layout.source_mask = 256; rejects(bad);
  bad = cfg; bad.frames_per_tensor = 0; rejects(bad);
  bad = cfg; bad.frames_per_tensor = 129; rejects(bad);
  bad = cfg; bad.packets_per_burst = 0; rejects(bad);
  bad = cfg; bad.packets_per_burst = 32 * 960 + 1; rejects(bad);
  bad = cfg; bad.target_gbps = 0; rejects(bad);
  bad = cfg; bad.target_gbps = std::numeric_limits<double>::quiet_NaN(); rejects(bad);
  bad = cfg; bad.duration_seconds = -0.5; rejects(bad);
  bad = cfg; bad.duration_seconds = std::numeric_limits<double>::infinity(); rejects(bad);
  bad = cfg; bad.report_interval_seconds = 0; rejects(bad);
  bad = cfg; bad.gpu_device = 1; rejects(bad);
  cfg.duration_seconds = -1; cfg.frames_per_tensor = 1; cfg.validate();
}
}  // namespace

int main() {
  try {
    for (unsigned mask = 1; mask <= 255; ++mask) {
      check_layout(mask, false);
      check_layout(mask, true);
    }
    check_patterns_and_geometry();
    check_config_and_clock();
    std::cout << "PASS: 510 source layouts, packet headers/checksums, tile coverage, "
                 "patterns, invalid settings, and pacing debt\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
