// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/ActionInitialization.hh"
#include "eels_sim/DetectorConstruction.hh"

#include "FTFP_BERT.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4RunManager.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4UImanager.hh"
#include "Randomize.hh"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <macro.mac>\n";
    return 2;
  }

  G4Random::setTheEngine(new CLHEP::RanecuEngine());

  auto* run_manager = new G4RunManager();
  run_manager->SetUserInitialization(new eels_sim::DetectorConstruction());

  auto* physics = new FTFP_BERT();
  physics->ReplacePhysics(new G4EmLivermorePhysics());
  physics->RegisterPhysics(new G4StepLimiterPhysics());
  physics->SetDefaultCutValue(10. * nm);
  run_manager->SetUserInitialization(physics);
  run_manager->SetUserInitialization(new eels_sim::ActionInitialization());

  const auto status =
      G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);
  delete run_manager;
  return status == 0 ? 0 : 1;
}
