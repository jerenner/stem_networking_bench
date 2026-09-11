// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#include "eels_sim/DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4GenericMessenger.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"

namespace eels_sim {

DetectorConstruction::DetectorConstruction()
    : rows_(960),
      columns_(3840),
      pixel_pitch_(10. * um),
      sensor_thickness_(5. * um),
      max_step_(0.1 * um) {
  DefineCommands();
}

DetectorConstruction::~DetectorConstruction() { delete messenger_; }

void DetectorConstruction::DefineCommands() {
  messenger_ = new G4GenericMessenger(this, "/eels/detector/", "Silicon sensor geometry");
  messenger_->DeclareProperty("rows", rows_, "Number of pixel rows used by digitization.");
  messenger_->DeclareProperty("columns", columns_, "Number of pixel columns used by digitization.");
  messenger_->DeclarePropertyWithUnit("pixelPitch", "um", pixel_pitch_, "Square pixel pitch.");
  messenger_->DeclarePropertyWithUnit(
      "thickness", "um", sensor_thickness_, "Active silicon thickness.");
  messenger_->DeclarePropertyWithUnit(
      "maxStep", "um", max_step_, "Maximum transport step in silicon.");
}

G4VPhysicalVolume* DetectorConstruction::Construct() {
  auto* nist = G4NistManager::Instance();
  auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
  auto* silicon = nist->FindOrBuildMaterial("G4_Si");

  const auto sensor_x = columns_ * pixel_pitch_;
  const auto sensor_y = rows_ * pixel_pitch_;
  const auto world_x = sensor_x + 2. * mm;
  const auto world_y = sensor_y + 2. * mm;
  const auto world_z = sensor_thickness_ + 40. * um;

  auto* world_solid = new G4Box("World", world_x / 2., world_y / 2., world_z / 2.);
  auto* world_logical = new G4LogicalVolume(world_solid, vacuum, "World");
  auto* world_physical = new G4PVPlacement(nullptr, {}, world_logical, "World", nullptr,
                                           false, 0, true);

  auto* sensor_solid =
      new G4Box("ActiveSilicon", sensor_x / 2., sensor_y / 2., sensor_thickness_ / 2.);
  sensor_volume_ = new G4LogicalVolume(sensor_solid, silicon, "ActiveSilicon");
  sensor_volume_->SetUserLimits(new G4UserLimits(max_step_));
  new G4PVPlacement(nullptr, {}, sensor_volume_, "ActiveSilicon", world_logical, false, 0,
                    true);

  return world_physical;
}

}  // namespace eels_sim
