// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/SteppingAction.hh"

#include "eels_sim/AnalysisSchema.hh"
#include "eels_sim/DetectorConstruction.hh"
#include "eels_sim/EventAction.hh"

#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

namespace eels_sim {

SteppingAction::SteppingAction(EventAction* event_action) : event_action_(event_action) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {
  if (sensor_volume_ == nullptr) {
    const auto* detector = static_cast<const DetectorConstruction*>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    sensor_volume_ = detector->GetSensorVolume();
  }

  const auto* pre = step->GetPreStepPoint();
  if (pre->GetTouchableHandle()->GetVolume()->GetLogicalVolume() != sensor_volume_) {
    return;
  }

  const auto edep = step->GetTotalEnergyDeposit();
  if (edep <= 0.) {
    return;
  }

  event_action_->AddDeposit(edep);
  const auto* post = step->GetPostStepPoint();
  const auto position = 0.5 * (pre->GetPosition() + post->GetPosition());
  const auto* track = step->GetTrack();
  const auto* process = post->GetProcessDefinedStep();
  const auto* event = G4RunManager::GetRunManager()->GetCurrentEvent();

  auto* analysis = G4AnalysisManager::Instance();
  analysis->FillNtupleIColumn(schema::kDeposits, 0, event->GetEventID());
  analysis->FillNtupleIColumn(schema::kDeposits, 1, track->GetTrackID());
  analysis->FillNtupleIColumn(schema::kDeposits, 2, track->GetParentID());
  analysis->FillNtupleIColumn(schema::kDeposits, 3,
                              track->GetParticleDefinition()->GetPDGEncoding());
  analysis->FillNtupleIColumn(schema::kDeposits, 4,
                              process == nullptr ? -1 : process->GetProcessType());
  analysis->FillNtupleIColumn(schema::kDeposits, 5,
                              process == nullptr ? -1 : process->GetProcessSubType());
  analysis->FillNtupleDColumn(schema::kDeposits, 6, position.x() / um);
  analysis->FillNtupleDColumn(schema::kDeposits, 7, position.y() / um);
  analysis->FillNtupleDColumn(schema::kDeposits, 8, position.z() / um);
  analysis->FillNtupleDColumn(schema::kDeposits, 9, pre->GetGlobalTime() / ns);
  analysis->FillNtupleDColumn(schema::kDeposits, 10, edep / eV);
  analysis->FillNtupleDColumn(schema::kDeposits, 11, pre->GetKineticEnergy() / keV);
  analysis->AddNtupleRow(schema::kDeposits);
}

}  // namespace eels_sim
