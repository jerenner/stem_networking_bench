// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/ActionInitialization.hh"

#include "eels_sim/EventAction.hh"
#include "eels_sim/PrimaryGeneratorAction.hh"
#include "eels_sim/RunAction.hh"
#include "eels_sim/SteppingAction.hh"

namespace eels_sim {

void ActionInitialization::Build() const {
  SetUserAction(new RunAction());
  SetUserAction(new PrimaryGeneratorAction());
  auto* event_action = new EventAction();
  SetUserAction(event_action);
  SetUserAction(new SteppingAction(event_action));
}

}  // namespace eels_sim
