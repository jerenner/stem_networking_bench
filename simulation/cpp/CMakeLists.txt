find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(eels_detector
  main.cc
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/EventAction.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
)

target_include_directories(eels_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(eels_detector PRIVATE cxx_std_17)
target_link_libraries(eels_detector PRIVATE ${Geant4_LIBRARIES})

install(TARGETS eels_detector RUNTIME DESTINATION bin)
