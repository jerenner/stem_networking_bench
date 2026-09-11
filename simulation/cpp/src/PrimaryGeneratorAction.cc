// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/PrimaryGeneratorAction.hh"

#include "eels_sim/AnalysisSchema.hh"

#include "G4AnalysisManager.hh"
#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4GenericMessenger.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace eels_sim {

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : particle_gun_(new G4ParticleGun(1)),
      energy_(300. * keV),
      energy_sigma_(0.),
      sigma_x_(0.),
      sigma_y_(0.),
      uniform_half_x_(0.),
      uniform_half_y_(0.),
      angular_sigma_(0.),
      source_z_(-5. * um),
      input_file_("") {
  particle_gun_->SetParticleDefinition(G4Electron::Definition());
  DefineCommands();
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete messenger_;
  delete particle_gun_;
}

void PrimaryGeneratorAction::DefineCommands() {
  messenger_ = new G4GenericMessenger(this, "/eels/source/", "Incident electron source");
  messenger_->DeclarePropertyWithUnit("energy", "keV", energy_, "Mean kinetic energy.");
  messenger_->DeclarePropertyWithUnit("energySigma", "keV", energy_sigma_,
                                      "Gaussian kinetic-energy spread.");
  messenger_->DeclarePropertyWithUnit("sigmaX", "um", sigma_x_, "Gaussian x width.");
  messenger_->DeclarePropertyWithUnit("sigmaY", "um", sigma_y_, "Gaussian y width.");
  messenger_->DeclarePropertyWithUnit(
      "uniformHalfX", "um", uniform_half_x_,
      "Uniform x half-width; when positive, overrides sigmaX.");
  messenger_->DeclarePropertyWithUnit(
      "uniformHalfY", "um", uniform_half_y_,
      "Uniform y half-width; when positive, overrides sigmaY.");
  messenger_->DeclarePropertyWithUnit("angularSigma", "mrad", angular_sigma_,
                                      "Gaussian divergence in each transverse axis.");
  messenger_->DeclarePropertyWithUnit("z", "um", source_z_, "Source-plane z position.");
  messenger_->DeclareProperty(
      "inputFile", input_file_,
      "Optional individual-electron CSV from the spectrometer transfer stage.");
}

void PrimaryGeneratorAction::LoadInputFile() {
  std::ifstream stream(input_file_);
  if (!stream) {
    const auto message = "Could not open spectrometer source CSV: " + input_file_;
    G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                FatalException, message.c_str());
    return;
  }

  std::string line;
  if (!std::getline(stream, line)) {
    G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                FatalException, "Spectrometer source CSV is empty.");
    return;
  }
  const std::string expected_header =
      "event_id,frame_id,electron_id,x_um,y_um,z_um,dir_x,dir_y,dir_z,"
      "kinetic_energy_eV,time_ns,weight,loss_channel";
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != expected_header) {
    G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                FatalException, "Unexpected spectrometer source CSV header.");
    return;
  }

  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::stringstream line_stream(line);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(line_stream, field, ',')) {
      fields.push_back(field);
    }
    if (fields.size() != 13) {
      G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                  FatalException, "Malformed spectrometer source CSV row.");
      return;
    }
    SourceRecord record{
        static_cast<G4int>(std::stoll(fields[0])),
        static_cast<G4int>(std::stoll(fields[1])),
        static_cast<G4int>(std::stoll(fields[2])),
        std::stod(fields[3]) * um,
        std::stod(fields[4]) * um,
        std::stod(fields[5]) * um,
        std::stod(fields[6]),
        std::stod(fields[7]),
        std::stod(fields[8]),
        std::stod(fields[9]) * eV,
        std::stod(fields[10]) * ns,
        std::stod(fields[11]),
        static_cast<G4int>(std::stoll(fields[12])),
    };
    if (record.event_id != static_cast<G4int>(source_records_.size())) {
      G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                  FatalException, "Source event_id values must be contiguous from zero.");
      return;
    }
    if (std::abs(record.weight - 1.) > 1.e-12) {
      G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                  FatalException, "Geant4 source records must have unit weight.");
      return;
    }
    source_records_.push_back(record);
  }
  if (source_records_.empty()) {
    G4Exception("PrimaryGeneratorAction::LoadInputFile", "EELS_SIM_SOURCE",
                FatalException, "Spectrometer source CSV contains no electrons.");
    return;
  }
  input_loaded_ = true;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  G4double x;
  G4double y;
  G4double z;
  G4ThreeVector direction;
  G4double sampled_energy;
  G4double time = 0.;
  G4int frame_id = -1;
  G4int electron_id = event->GetEventID();
  G4int loss_channel = -1;

  if (!input_file_.empty()) {
    if (!input_loaded_) {
      LoadInputFile();
    }
    if (event->GetEventID() >= static_cast<G4int>(source_records_.size())) {
      G4Exception("PrimaryGeneratorAction::GeneratePrimaries", "EELS_SIM_SOURCE",
                  FatalException,
                  "beamOn requests more events than the spectrometer source contains.");
      return;
    }
    const auto& record = source_records_[event->GetEventID()];
    x = record.x;
    y = record.y;
    z = record.z;
    direction = G4ThreeVector(record.dir_x, record.dir_y, record.dir_z).unit();
    sampled_energy = record.kinetic_energy;
    time = record.time;
    frame_id = record.frame_id;
    electron_id = record.electron_id;
    loss_channel = record.loss_channel;
  } else {
    x = uniform_half_x_ > 0.
            ? (2. * G4UniformRand() - 1.) * uniform_half_x_
            : G4RandGauss::shoot(0., sigma_x_);
    y = uniform_half_y_ > 0.
            ? (2. * G4UniformRand() - 1.) * uniform_half_y_
            : G4RandGauss::shoot(0., sigma_y_);
    z = source_z_;
    const auto tx = G4RandGauss::shoot(0., angular_sigma_);
    const auto ty = G4RandGauss::shoot(0., angular_sigma_);
    direction = G4ThreeVector(tx, ty, 1.).unit();
    sampled_energy =
        std::max(1. * eV, G4RandGauss::shoot(energy_, energy_sigma_));
  }

  particle_gun_->SetParticlePosition({x, y, z});
  particle_gun_->SetParticleMomentumDirection(direction);
  particle_gun_->SetParticleEnergy(sampled_energy);
  particle_gun_->SetParticleTime(time);
  particle_gun_->GeneratePrimaryVertex(event);

  auto* analysis = G4AnalysisManager::Instance();
  analysis->FillNtupleIColumn(schema::kPrimaries, 0, event->GetEventID());
  analysis->FillNtupleIColumn(schema::kPrimaries, 1, frame_id);
  analysis->FillNtupleIColumn(schema::kPrimaries, 2, electron_id);
  analysis->FillNtupleDColumn(schema::kPrimaries, 3, x / um);
  analysis->FillNtupleDColumn(schema::kPrimaries, 4, y / um);
  analysis->FillNtupleDColumn(schema::kPrimaries, 5, z / um);
  analysis->FillNtupleDColumn(schema::kPrimaries, 6, direction.x());
  analysis->FillNtupleDColumn(schema::kPrimaries, 7, direction.y());
  analysis->FillNtupleDColumn(schema::kPrimaries, 8, direction.z());
  analysis->FillNtupleDColumn(schema::kPrimaries, 9, sampled_energy / keV);
  analysis->FillNtupleDColumn(schema::kPrimaries, 10, time / ns);
  analysis->FillNtupleDColumn(schema::kPrimaries, 11, 1.);
  analysis->FillNtupleIColumn(schema::kPrimaries, 12, loss_channel);
  analysis->AddNtupleRow(schema::kPrimaries);
}

}  // namespace eels_sim
