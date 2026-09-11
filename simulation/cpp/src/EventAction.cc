// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/EventAction.hh"

#include "eels_sim/AnalysisSchema.hh"

#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

namespace eels_sim {

void EventAction::BeginOfEventAction(const G4Event*) {
  total_edep_ = 0.;
  deposit_count_ = 0;
}

void EventAction::EndOfEventAction(const G4Event* event) {
  auto* analysis = G4AnalysisManager::Instance();
  analysis->FillNtupleIColumn(schema::kEvents, 0, event->GetEventID());
  analysis->FillNtupleIColumn(schema::kEvents, 1, deposit_count_);
  analysis->FillNtupleDColumn(schema::kEvents, 2, total_edep_ / eV);
  analysis->AddNtupleRow(schema::kEvents);
}

}  // namespace eels_sim
