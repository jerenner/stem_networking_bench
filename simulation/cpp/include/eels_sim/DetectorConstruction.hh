// SPDX-FileCopyrightText: Copyright (c) 2026 EELS Simulation Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4GenericMessenger;
class G4LogicalVolume;
class G4VPhysicalVolume;

namespace eels_sim {

class DetectorConstruction final : public G4VUserDetectorConstruction {
 public:
  DetectorConstruction();
  ~DetectorConstruction() override;

  G4VPhysicalVolume* Construct() override;

  G4LogicalVolume* GetSensorVolume() const { return sensor_volume_; }
  G4int GetRows() const { return rows_; }
  G4int GetColumns() const { return columns_; }
  G4double GetPixelPitch() const { return pixel_pitch_; }
  G4double GetSensorThickness() const { return sensor_thickness_; }
  G4double GetMaxStep() const { return max_step_; }

 private:
  void DefineCommands();

  G4int rows_;
  G4int columns_;
  G4double pixel_pitch_;
  G4double sensor_thickness_;
  G4double max_step_;
  G4LogicalVolume* sensor_volume_ = nullptr;
  G4GenericMessenger* messenger_ = nullptr;
};

}  // namespace eels_sim
