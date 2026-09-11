// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

#include <vector>

class G4Event;
class G4GenericMessenger;
class G4ParticleGun;

namespace eels_sim {

class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction {
 public:
  PrimaryGeneratorAction();
  ~PrimaryGeneratorAction() override;

  void GeneratePrimaries(G4Event* event) override;

 private:
  struct SourceRecord {
    G4int event_id;
    G4int frame_id;
    G4int electron_id;
    G4double x;
    G4double y;
    G4double z;
    G4double dir_x;
    G4double dir_y;
    G4double dir_z;
    G4double kinetic_energy;
    G4double time;
    G4double weight;
    G4int loss_channel;
  };

  void DefineCommands();
  void LoadInputFile();

  G4ParticleGun* particle_gun_ = nullptr;
  G4GenericMessenger* messenger_ = nullptr;
  G4double energy_;
  G4double energy_sigma_;
  G4double sigma_x_;
  G4double sigma_y_;
  G4double uniform_half_x_;
  G4double uniform_half_y_;
  G4double angular_sigma_;
  G4double source_z_;
  G4String input_file_;
  std::vector<SourceRecord> source_records_;
  G4bool input_loaded_ = false;
};

}  // namespace eels_sim
