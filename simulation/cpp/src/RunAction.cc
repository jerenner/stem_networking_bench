// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/RunAction.hh"

#include "eels_sim/AnalysisSchema.hh"
#include "eels_sim/DetectorConstruction.hh"

#include "G4AnalysisManager.hh"
#include "G4GenericMessenger.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

namespace eels_sim {

RunAction::RunAction() : output_file_("output/deposits") {
  messenger_ = new G4GenericMessenger(this, "/eels/output/", "Simulation output");
  messenger_->DeclareProperty("file", output_file_,
                              "Output base path; one CSV file is written per ntuple.");
  CreateNtuples();
}

RunAction::~RunAction() { delete messenger_; }

void RunAction::CreateNtuples() {
  auto* analysis = G4AnalysisManager::Instance();
  // CSV avoids binding this transport stage to one particular HDF5 ABI.  The
  // Python digitizer writes the stable, self-describing HDF5 product.
  analysis->SetDefaultFileType("csv");
  analysis->SetVerboseLevel(1);

  analysis->CreateNtuple("run_info", "Detector configuration");
  analysis->CreateNtupleIColumn("rows");
  analysis->CreateNtupleIColumn("pixel_columns");
  analysis->CreateNtupleDColumn("pixel_pitch_um");
  analysis->CreateNtupleDColumn("sensor_thickness_um");
  analysis->CreateNtupleDColumn("max_step_um");
  analysis->FinishNtuple();

  analysis->CreateNtuple("primaries", "Incident electrons");
  analysis->CreateNtupleIColumn("event_id");
  analysis->CreateNtupleIColumn("frame_id");
  analysis->CreateNtupleIColumn("electron_id");
  analysis->CreateNtupleDColumn("x_um");
  analysis->CreateNtupleDColumn("y_um");
  analysis->CreateNtupleDColumn("z_um");
  analysis->CreateNtupleDColumn("dir_x");
  analysis->CreateNtupleDColumn("dir_y");
  analysis->CreateNtupleDColumn("dir_z");
  analysis->CreateNtupleDColumn("kinetic_energy_keV");
  analysis->CreateNtupleDColumn("time_ns");
  analysis->CreateNtupleDColumn("weight");
  analysis->CreateNtupleIColumn("loss_channel");
  analysis->FinishNtuple();

  analysis->CreateNtuple("deposits", "Energy-deposition steps in active silicon");
  analysis->CreateNtupleIColumn("event_id");
  analysis->CreateNtupleIColumn("track_id");
  analysis->CreateNtupleIColumn("parent_id");
  analysis->CreateNtupleIColumn("pdg");
  analysis->CreateNtupleIColumn("process_type");
  analysis->CreateNtupleIColumn("process_subtype");
  analysis->CreateNtupleDColumn("x_um");
  analysis->CreateNtupleDColumn("y_um");
  analysis->CreateNtupleDColumn("z_um");
  analysis->CreateNtupleDColumn("time_ns");
  analysis->CreateNtupleDColumn("edep_eV");
  analysis->CreateNtupleDColumn("pre_kinetic_energy_keV");
  analysis->FinishNtuple();

  analysis->CreateNtuple("events", "Per-primary detector totals");
  analysis->CreateNtupleIColumn("event_id");
  analysis->CreateNtupleIColumn("deposit_count");
  analysis->CreateNtupleDColumn("total_edep_eV");
  analysis->FinishNtuple();
}

void RunAction::BeginOfRunAction(const G4Run*) {
  auto* analysis = G4AnalysisManager::Instance();
  if (!analysis->OpenFile(output_file_)) {
    G4Exception("RunAction::BeginOfRunAction", "EELS_SIM_OUTPUT", FatalException,
                "Could not open transport output files.");
  }

  const auto* detector = static_cast<const DetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  analysis->FillNtupleIColumn(schema::kRunInfo, 0, detector->GetRows());
  analysis->FillNtupleIColumn(schema::kRunInfo, 1, detector->GetColumns());
  analysis->FillNtupleDColumn(schema::kRunInfo, 2, detector->GetPixelPitch() / um);
  analysis->FillNtupleDColumn(schema::kRunInfo, 3, detector->GetSensorThickness() / um);
  analysis->FillNtupleDColumn(schema::kRunInfo, 4, detector->GetMaxStep() / um);
  analysis->AddNtupleRow(schema::kRunInfo);
}

void RunAction::EndOfRunAction(const G4Run*) {
  auto* analysis = G4AnalysisManager::Instance();
  analysis->Write();
  analysis->CloseFile();
}

}  // namespace eels_sim
