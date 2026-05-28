/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Phase 1 stem_daqiri_tx: paced STEM-format TX over daqiri/DPDK on a single
 * ConnectX-7 NIC. Modeled on
 *   third_party/daqiri/examples/raw_gpudirect_bench.cpp::tx_worker
 *
 * The binary loads a YAML config (daqiri.cfg block + stem_tx app block),
 * calls daqiri_init, then runs a single-threaded TX loop that:
 *
 *   1. Builds STEM-format packet templates on host (Eth/IPv4/UDP headers
 *      plus the 64-byte STEM custom header + 7680-byte payload row).
 *   2. For each unique GPU TX buffer that daqiri hands out, cudaMemcpy's
 *      one template into it (cycles through (source_id, row_offset) per
 *      packet position in a burst). After this one-time fill, the same
 *      buffer is re-sent on every subsequent burst.
 *   3. Optionally launches a CUDA kernel to update STEM headers on the GPU
 *      (Phase 3 latency stamping path; off by default in Phase 1).
 *   4. send_tx_burst.
 *   5. Token-bucket sleep until enough wall-clock has elapsed for the
 *      target_rate_gbps; stop when total_time_to_send_s elapses.
 *
 * Prints a final summary: bytes sent, packets sent, achieved Gbps,
 * achieved pps. Exits 0 on clean shutdown.
 */
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <daqiri/daqiri.h>

#include "stem_kernels.h"
#include "stem_packet.h"
#include "stem_pacing.h"

namespace {

// ===========================================================================
// stem_tx YAML block parser. Mirrors daqiri::bench::RawBenchTxConfig but
// extended with STEM-specific knobs.
// ===========================================================================
struct StemTxConfig {
  std::string interface_name = "tx_port";

  // daqiri TX wire layout: header_size = Eth+IPv4+UDP = 42; payload_size
  // = STEM 64B header + 7680B row payload = 7744.
  uint32_t batch_size   = 1024;
  uint32_t header_size  = stem::TX_DAQIRI_HEADER_SIZE;
  uint32_t payload_size = stem::TX_DAQIRI_PAYLOAD_SIZE;

  std::string ip_src_addr  = "169.254.100.253";
  std::string ip_dst_addr  = "169.254.95.47";
  std::string eth_src_addr = "00:00:00:00:00:00";
  std::string eth_dst_addr = "00:00:00:00:00:00";
  uint16_t    udp_src_port = 4096;
  uint16_t    udp_dst_port = 4096;

  // STEM knobs
  uint32_t num_sources_active = stem::NUM_SOURCES_MAX;   // 8
  uint32_t rows_per_source    = stem::ROWS_PER_SOURCE;   // 128
  // Phase 3: update STEM headers on the GPU each burst so row_number/
  // source_id varies across bursts and the RX can frame-assemble properly.
  bool     update_headers_per_burst = false;
  // Phase 3 latency stamping: write epoch_us into the STEM header of the
  // first packet of every burst.
  bool     stamp_epoch_us = false;

  // Pacing
  stem::PacingConfig pacing;
};

StemTxConfig parse_stem_tx_cfg(const YAML::Node& root) {
  StemTxConfig cfg;
  if (!root["stem_tx"]) {
    std::cerr << "config missing top-level 'stem_tx' block\n";
    return cfg;
  }
  const auto tx = root["stem_tx"];
  cfg.interface_name = tx["interface_name"].as<std::string>(cfg.interface_name);
  cfg.batch_size     = tx["batch_size"].as<uint32_t>(cfg.batch_size);
  cfg.header_size    = tx["header_size"].as<uint32_t>(cfg.header_size);
  cfg.payload_size   = tx["payload_size"].as<uint32_t>(cfg.payload_size);
  cfg.ip_src_addr    = tx["ip_src_addr"].as<std::string>(cfg.ip_src_addr);
  cfg.ip_dst_addr    = tx["ip_dst_addr"].as<std::string>(cfg.ip_dst_addr);
  cfg.eth_src_addr   = tx["eth_src_addr"].as<std::string>(cfg.eth_src_addr);
  cfg.eth_dst_addr   = tx["eth_dst_addr"].as<std::string>(cfg.eth_dst_addr);
  cfg.udp_src_port   = tx["udp_src_port"].as<uint16_t>(cfg.udp_src_port);
  cfg.udp_dst_port   = tx["udp_dst_port"].as<uint16_t>(cfg.udp_dst_port);
  cfg.num_sources_active = tx["num_sources_active"].as<uint32_t>(cfg.num_sources_active);
  cfg.rows_per_source    = tx["rows_per_source"].as<uint32_t>(cfg.rows_per_source);
  cfg.update_headers_per_burst =
      tx["update_headers_per_burst"].as<bool>(cfg.update_headers_per_burst);
  cfg.stamp_epoch_us = tx["stamp_epoch_us"].as<bool>(cfg.stamp_epoch_us);
  cfg.pacing.target_rate_gbps =
      tx["target_rate_gbps"].as<double>(cfg.pacing.target_rate_gbps);
  cfg.pacing.total_time_to_send_s =
      tx["total_time_to_send"].as<double>(cfg.pacing.total_time_to_send_s);
  cfg.pacing.total_bytes_cap =
      tx["total_bytes_cap"].as<uint64_t>(cfg.pacing.total_bytes_cap);
  return cfg;
}

// ---------------------------------------------------------------------------
// Signal handling for clean Ctrl-C.
// ---------------------------------------------------------------------------
volatile std::sig_atomic_t g_stop_requested = 0;
void on_sigint(int) { g_stop_requested = 1; }

// ---------------------------------------------------------------------------
// Build STEM-format packet templates for one full burst.
//
// We pre-compute kBatchSize host-side packet templates -- enough variety
// that each packet position in a burst has a unique (source_id, row_offset)
// pair that cycles through the 8x128 = 1024 distinct possibilities. When
// daqiri hands out a GPU buffer for packet position i in any future burst,
// we memcpy template[i % template_count] into it.
// ---------------------------------------------------------------------------
void build_templates(const StemTxConfig& cfg,
                     uint32_t          burst_size,
                     std::vector<std::vector<uint8_t>>* templates,
                     uint16_t hdr_eth_offset) {
  templates->clear();
  templates->reserve(burst_size);

  char eth_dst[6] = {0};
  char eth_src[6] = {0};
  daqiri::format_eth_addr(eth_src, cfg.eth_src_addr);
  daqiri::format_eth_addr(eth_dst, cfg.eth_dst_addr);

  uint32_t ip_src_n = 0;
  uint32_t ip_dst_n = 0;
  inet_pton(AF_INET, cfg.ip_src_addr.c_str(), &ip_src_n);
  inet_pton(AF_INET, cfg.ip_dst_addr.c_str(), &ip_dst_n);
  const uint32_t ip_src_host = ntohl(ip_src_n);
  const uint32_t ip_dst_host = ntohl(ip_dst_n);

  // NOTE: do NOT bake an epoch_us value into any template. If we did, the
  // RX would observe stale stamps any time a DPDK mbuf rotates through
  // position 0 and back out. The per-burst host-stamp path is the only
  // place epoch_us is written.

  // For each packet position in the burst, pick the matching (source_id,
  // row_offset). The cycle wraps every num_sources_active * rows_per_source
  // packets and matches one full frame of STEM data when set to (8, 128).
  for (uint32_t i = 0; i < burst_size; ++i) {
    std::vector<uint8_t> pkt(cfg.header_size + cfg.payload_size, 0);

    // We can't call daqiri::bench::populate_udp_ipv4_headers() because
    // raw_bench_common.cpp lives in the daqiri benchmarks helper lib which
    // we explicitly exclude from build (DAQIRI_BUILD_EXAMPLES=OFF). Build
    // Eth/IPv4/UDP headers inline here.
    (void)hdr_eth_offset;
    {
      auto* p = pkt.data();

      // Eth (14 bytes): dst (6) + src (6) + ethertype (2)
      std::memcpy(p + 0, eth_dst, 6);
      std::memcpy(p + 6, eth_src, 6);
      p[12] = 0x08; p[13] = 0x00;  // ETH_P_IP

      // IPv4 (20 bytes)
      p[14] = 0x45;  // version=4, ihl=5
      p[15] = 0x00;  // tos
      const uint16_t ip_total_len =
          static_cast<uint16_t>(cfg.header_size + cfg.payload_size - 14);
      p[16] = static_cast<uint8_t>((ip_total_len >> 8) & 0xff);
      p[17] = static_cast<uint8_t>(ip_total_len & 0xff);
      p[18] = 0; p[19] = 0;          // identification
      p[20] = 0x40; p[21] = 0x00;     // flags=DF, frag offset = 0
      p[22] = 64;                      // ttl
      p[23] = 17;                      // udp
      p[24] = 0; p[25] = 0;            // header checksum (computed below)
      p[26] = static_cast<uint8_t>((ip_src_host >> 24) & 0xff);
      p[27] = static_cast<uint8_t>((ip_src_host >> 16) & 0xff);
      p[28] = static_cast<uint8_t>((ip_src_host >> 8) & 0xff);
      p[29] = static_cast<uint8_t>(ip_src_host & 0xff);
      p[30] = static_cast<uint8_t>((ip_dst_host >> 24) & 0xff);
      p[31] = static_cast<uint8_t>((ip_dst_host >> 16) & 0xff);
      p[32] = static_cast<uint8_t>((ip_dst_host >> 8) & 0xff);
      p[33] = static_cast<uint8_t>(ip_dst_host & 0xff);

      // IPv4 header checksum (1's complement over IP header bytes 14..33).
      uint32_t sum = 0;
      for (int b = 14; b < 34; b += 2) {
        sum += (static_cast<uint32_t>(p[b]) << 8) |
               static_cast<uint32_t>(p[b + 1]);
      }
      while (sum >> 16) { sum = (sum & 0xffff) + (sum >> 16); }
      const uint16_t cksum = static_cast<uint16_t>(~sum & 0xffff);
      p[24] = static_cast<uint8_t>((cksum >> 8) & 0xff);
      p[25] = static_cast<uint8_t>(cksum & 0xff);

      // UDP (8 bytes)
      const uint16_t udp_len =
          static_cast<uint16_t>(cfg.header_size + cfg.payload_size - 14 - 20);
      p[34] = static_cast<uint8_t>((cfg.udp_src_port >> 8) & 0xff);
      p[35] = static_cast<uint8_t>(cfg.udp_src_port & 0xff);
      p[36] = static_cast<uint8_t>((cfg.udp_dst_port >> 8) & 0xff);
      p[37] = static_cast<uint8_t>(cfg.udp_dst_port & 0xff);
      p[38] = static_cast<uint8_t>((udp_len >> 8) & 0xff);
      p[39] = static_cast<uint8_t>(udp_len & 0xff);
      p[40] = 0; p[41] = 0;  // udp checksum (optional for IPv4; leave 0)
    }

    // STEM custom 64-byte header at offset = header_size (42).
    const uint32_t source_id = i % cfg.num_sources_active;
    const uint32_t row_offset = (i / cfg.num_sources_active) % cfg.rows_per_source;
    const uint16_t row_number = static_cast<uint16_t>(row_offset);
    stem::stem_tx_stamp_packet(pkt.data() + cfg.header_size, row_number,
                               static_cast<uint16_t>(source_id),
                               /*epoch_us=*/0);

    // STEM payload: deterministic uint16 ramp so loss is visible in dumps.
    auto* payload16 = reinterpret_cast<uint16_t*>(
        pkt.data() + cfg.header_size + stem::STEM_HEADER_SIZE);
    const uint32_t samples = stem::STEM_PAYLOAD_SIZE / sizeof(uint16_t);
    for (uint32_t s = 0; s < samples; ++s) {
      payload16[s] = static_cast<uint16_t>((source_id << 12) |
                                           (row_offset & 0xff) |
                                           ((s & 0xf) << 8));
    }

    templates->push_back(std::move(pkt));
  }
}

// ---------------------------------------------------------------------------
// TX worker.
// ---------------------------------------------------------------------------
void tx_worker(const StemTxConfig& cfg, std::atomic<bool>& stop) {
  const int port_id = daqiri::get_port_id(cfg.interface_name);
  if (port_id < 0) {
    std::cerr << "Invalid TX interface_name: " << cfg.interface_name << "\n";
    stop.store(true);
    return;
  }

  std::vector<std::vector<uint8_t>> templates;
  build_templates(cfg, cfg.batch_size, &templates, 0);

  // Cache: maps each unique GPU buffer ptr we've ever seen to the index
  // of the template that was copied into it. After all unique buffers have
  // been initialized, we never memcpy again -- daqiri recycles the same
  // pool of buffers and we just keep sending them.
  std::unordered_map<void*, uint32_t> initialized_buffers;
  initialized_buffers.reserve(cfg.batch_size * 4);

  cudaStream_t header_stream = nullptr;
  cudaStreamCreateWithFlags(&header_stream, cudaStreamNonBlocking);

  // GPU scratch for per-burst header updates (Phase 3 path). Allocated once.
  uint16_t* dev_row_numbers = nullptr;
  uint16_t* dev_source_ids = nullptr;
  uint8_t** dev_pkt_ptrs = nullptr;
  if (cfg.update_headers_per_burst) {
    cudaMalloc(&dev_row_numbers, cfg.batch_size * sizeof(uint16_t));
    cudaMalloc(&dev_source_ids, cfg.batch_size * sizeof(uint16_t));
    cudaMalloc(&dev_pkt_ptrs, cfg.batch_size * sizeof(uint8_t*));
  }
  std::vector<uint16_t> host_row_numbers(cfg.batch_size, 0);
  std::vector<uint16_t> host_source_ids(cfg.batch_size, 0);
  std::vector<uint8_t*> host_pkt_ptrs(cfg.batch_size, nullptr);

  uint64_t global_pkt_counter = 0;
  uint64_t total_packets = 0;
  stem::Pacer pacer(cfg.pacing.target_rate_gbps);
  pacer.start();

  while (!stop.load() && !g_stop_requested) {
    if (stem::should_stop(cfg.pacing, pacer)) { break; }

    auto* msg = daqiri::create_tx_burst_params();
    daqiri::set_header(msg, static_cast<uint16_t>(port_id), 0, cfg.batch_size, 1);

    if (!daqiri::is_tx_burst_available(msg)) {
      daqiri::free_tx_metadata(msg);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    if (daqiri::get_tx_packet_burst(msg) != daqiri::Status::SUCCESS) {
      daqiri::free_tx_metadata(msg);
      continue;
    }

    const auto num_pkts = static_cast<int>(daqiri::get_num_packets(msg));
    bool failed = false;
    uint64_t burst_bytes = 0;

    for (int i = 0; i < num_pkts; ++i) {
      auto* gpu_pkt = static_cast<uint8_t*>(daqiri::get_segment_packet_ptr(msg, 0, i));
      const uint32_t tpl_idx = static_cast<uint32_t>(i) % templates.size();
      const auto& tpl = templates[tpl_idx];

      auto it = initialized_buffers.find(gpu_pkt);
      if (it == initialized_buffers.end()) {
        if (cudaMemcpy(gpu_pkt, tpl.data(), tpl.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          failed = true;
          break;
        }
        initialized_buffers.emplace(gpu_pkt, tpl_idx);
      }

      if (daqiri::set_packet_lengths(
              msg, i,
              {static_cast<int>(cfg.header_size + cfg.payload_size)}) !=
          daqiri::Status::SUCCESS) {
        failed = true;
        break;
      }
      burst_bytes += cfg.header_size + cfg.payload_size;

      if (cfg.update_headers_per_burst) {
        const uint64_t k = global_pkt_counter + static_cast<uint64_t>(i);
        const uint32_t source_id = static_cast<uint32_t>(
            k % cfg.num_sources_active);
        const uint64_t row_global =
            (k / cfg.num_sources_active) % stem::ROW_NUMBER_WRAP;
        host_row_numbers[i] = static_cast<uint16_t>(row_global);
        host_source_ids[i]  = static_cast<uint16_t>(source_id);
        host_pkt_ptrs[i]    = gpu_pkt + cfg.header_size;  // start of STEM hdr
      }
    }

    if (failed) {
      daqiri::free_all_packets_and_burst_tx(msg);
      continue;
    }

    if (cfg.update_headers_per_burst) {
      cudaMemcpyAsync(dev_row_numbers, host_row_numbers.data(),
                      num_pkts * sizeof(uint16_t),
                      cudaMemcpyHostToDevice, header_stream);
      cudaMemcpyAsync(dev_source_ids, host_source_ids.data(),
                      num_pkts * sizeof(uint16_t),
                      cudaMemcpyHostToDevice, header_stream);
      cudaMemcpyAsync(dev_pkt_ptrs, host_pkt_ptrs.data(),
                      num_pkts * sizeof(uint8_t*),
                      cudaMemcpyHostToDevice, header_stream);
      const uint64_t epoch_us = cfg.stamp_epoch_us
          ? std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()
          : 0;
      // The kernel writes row_number/source_id for every packet and
      // additionally zeroes epoch_us for every pkt_idx > 0 so a buffer
      // that DPDK rotated out of position 0 doesn't surface a stale
      // stamp at the RX.
      stem::stem_tx_update_burst_headers(
          dev_pkt_ptrs, dev_row_numbers, dev_source_ids,
          static_cast<uint32_t>(num_pkts), 0, epoch_us, header_stream);
      cudaStreamSynchronize(header_stream);
    } else if (cfg.stamp_epoch_us) {
      // Phase 3 latency-stamp path that does NOT update row_numbers per
      // burst. daqiri's memory regions on Spark are kind: host_pinned --
      // we can write directly with host stores; no cudaMemcpy round-trip
      // is required.
      //
      // Crucially we ALSO zero epoch_us in positions 1..N-1 so a buffer
      // that DPDK rotated out of position 0 doesn't surface a stale stamp
      // at the RX.
      const uint64_t epoch_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
      static const uint64_t kZeroEpoch = 0;
      for (int i = 0; i < num_pkts; ++i) {
        auto* pkt = static_cast<uint8_t*>(
            daqiri::get_segment_packet_ptr(msg, 0, i));
        if (pkt == nullptr) { continue; }
        std::memcpy(
            pkt + cfg.header_size + stem::STEM_HDR_OFF_EPOCH_US,
            (i == 0) ? &epoch_us : &kZeroEpoch,
            sizeof(uint64_t));
      }
    }

    daqiri::send_tx_burst(msg);
    pacer.record_burst(burst_bytes);
    total_packets += num_pkts;
    global_pkt_counter += num_pkts;
    pacer.wait_for_next_burst();
  }

  if (cfg.update_headers_per_burst) {
    cudaFree(dev_row_numbers);
    cudaFree(dev_source_ids);
    cudaFree(dev_pkt_ptrs);
  }
  cudaStreamDestroy(header_stream);

  const double secs = pacer.elapsed_seconds();
  const double gbps = secs > 0
      ? static_cast<double>(pacer.total_bytes_emitted()) * 8.0 / (secs * 1e9)
      : 0.0;
  const double pps = secs > 0
      ? static_cast<double>(total_packets) / secs
      : 0.0;
  std::printf("stem_daqiri_tx complete:\n");
  std::printf("  duration       : %.3f s\n", secs);
  std::printf("  bytes sent     : %lu\n",
              static_cast<unsigned long>(pacer.total_bytes_emitted()));
  std::printf("  packets sent   : %lu\n",
              static_cast<unsigned long>(total_packets));
  std::printf("  achieved Gbps  : %.3f\n", gbps);
  std::printf("  achieved pps   : %.3f\n", pps);
  std::printf("  target Gbps    : %.3f (0 = unbounded)\n",
              cfg.pacing.target_rate_gbps);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <config.yaml> [--seconds N] [--rate GBPS]\n";
    return 1;
  }

  std::signal(SIGINT, on_sigint);

  // CLI override for --seconds / --rate so the parity sweep in Phase 3 can
  // reuse one YAML across multiple target rates.
  double cli_seconds = -1.0;
  double cli_rate    = -1.0;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--seconds") { cli_seconds = std::stod(argv[i + 1]); }
    else if (flag == "--rate") { cli_rate = std::stod(argv[i + 1]); }
  }

  const auto root = YAML::LoadFile(argv[1]);
  if (daqiri::daqiri_init(argv[1]) != daqiri::Status::SUCCESS) {
    std::cerr << "daqiri_init failed for config " << argv[1] << "\n";
    return 1;
  }

  StemTxConfig cfg = parse_stem_tx_cfg(root);
  if (cli_seconds >= 0.0) { cfg.pacing.total_time_to_send_s = cli_seconds; }
  if (cli_rate    >= 0.0) { cfg.pacing.target_rate_gbps     = cli_rate; }

  std::cout << "stem_daqiri_tx starting on '" << cfg.interface_name
            << "' -> target=" << cfg.pacing.target_rate_gbps
            << " Gbps duration=" << cfg.pacing.total_time_to_send_s
            << " s batch=" << cfg.batch_size
            << " packet=" << (cfg.header_size + cfg.payload_size)
            << "B\n";

  std::atomic<bool> stop{false};
  std::thread t(tx_worker, cfg, std::ref(stop));
  t.join();

  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
