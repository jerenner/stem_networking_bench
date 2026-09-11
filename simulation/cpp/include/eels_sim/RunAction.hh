// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4UserRunAction.hh"
#include "globals.hh"

class G4GenericMessenger;
class G4Run;

namespace eels_sim {

class RunAction final : public G4UserRunAction {
 public:
  RunAction();
  ~RunAction() override;

  void BeginOfRunAction(const G4Run* run) override;
  void EndOfRunAction(const G4Run* run) override;

 private:
  void CreateNtuples();

  G4String output_file_;
  G4GenericMessenger* messenger_ = nullptr;
};

}  // namespace eels_sim
