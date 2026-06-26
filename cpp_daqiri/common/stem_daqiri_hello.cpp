/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Hello/link-check build/run gate. Loads a YAML config, calls daqiri_init,
 * prints daqiri stats, then calls daqiri::shutdown. Used to verify that
 * the cpp_daqiri build links cleanly against daqiri and that daqiri can
 * initialize against a minimal config inside the container.
 *
 * Usage:
 *   stem_daqiri_hello <config.yaml>
 *   stem_daqiri_hello --self-test
 *
 * In --self-test mode the binary just prints "ok" and exits 0 -- this is
 * the safe path that does not touch any NIC and works on any host. Run
 * with a real config to also exercise the daqiri::daqiri_init code path.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <daqiri/daqiri.h>

#include "stem_packet.h"

namespace {

void print_usage(const char* argv0) {
  std::cout << "Usage:\n"
            << "  " << argv0 << " <config.yaml>     # init/shutdown daqiri "
                                "against the given config\n"
            << "  " << argv0 << " --self-test        # link-check only, "
                                "no NIC access, exits 0\n"
            << "  " << argv0 << " --print-layout     # dump STEM wire "
                                "constants and exit 0\n";
}

void print_layout() {
  std::cout << "STEM wire layout:\n"
            << "  L4 header:           " << stem::L4_HEADER_SIZE << " bytes\n"
            << "  STEM custom header:  " << stem::STEM_HEADER_SIZE << " bytes\n"
            << "  STEM payload (row):  " << stem::STEM_PAYLOAD_SIZE
            << " bytes (" << stem::FRAME_WIDTH << " * uint16)\n"
            << "  Total per packet:    " << stem::STEM_PACKET_BYTES << " bytes\n"
            << "Frame geometry:\n"
            << "  FRAME_WIDTH:         " << stem::FRAME_WIDTH << "\n"
            << "  FRAME_HEIGHT:        " << stem::FRAME_HEIGHT << "\n"
            << "  ROWS_PER_SOURCE:     " << stem::ROWS_PER_SOURCE << "\n"
            << "  NUM_SOURCES_MAX:     " << stem::NUM_SOURCES_MAX << "\n"
            << "  PACKETS_PER_FRAME:   " << stem::PACKETS_PER_FRAME_FULL << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string arg1 = argv[1];

  if (arg1 == "--help" || arg1 == "-h") {
    print_usage(argv[0]);
    return 0;
  }

  if (arg1 == "--print-layout") {
    print_layout();
    return 0;
  }

  if (arg1 == "--self-test") {
    std::cout << "stem_daqiri_hello: self-test ok (built against daqiri "
                 "headers, link-check passed)\n";
    print_layout();
    return 0;
  }

  // Real init path: requires a daqiri-formatted YAML.
  std::cout << "stem_daqiri_hello: calling daqiri_init(\"" << arg1 << "\")\n";
  if (daqiri::daqiri_init(arg1) != daqiri::Status::SUCCESS) {
    std::cerr << "stem_daqiri_hello: daqiri_init failed\n";
    return 1;
  }
  std::cout << "stem_daqiri_hello: daqiri_init succeeded; calling shutdown\n";
  daqiri::print_stats();
  daqiri::shutdown();
  std::cout << "stem_daqiri_hello: clean exit\n";
  return 0;
}
