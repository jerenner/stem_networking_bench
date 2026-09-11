// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4VUserActionInitialization.hh"

namespace eels_sim {

class ActionInitialization final : public G4VUserActionInitialization {
 public:
  void Build() const override;
};

}  // namespace eels_sim
