// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4UserSteppingAction.hh"

class G4LogicalVolume;
class G4Step;

namespace eels_sim {

class EventAction;

class SteppingAction final : public G4UserSteppingAction {
 public:
  explicit SteppingAction(EventAction* event_action);
  void UserSteppingAction(const G4Step* step) override;

 private:
  EventAction* event_action_;
  G4LogicalVolume* sensor_volume_ = nullptr;
};

}  // namespace eels_sim
