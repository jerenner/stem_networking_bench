// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4UserEventAction.hh"
#include "globals.hh"

class G4Event;

namespace eels_sim {

class EventAction final : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event* event) override;
  void EndOfEventAction(const G4Event* event) override;

  void AddDeposit(G4double energy) {
    total_edep_ += energy;
    ++deposit_count_;
  }

 private:
  G4double total_edep_ = 0.;
  G4int deposit_count_ = 0;
};

}  // namespace eels_sim
