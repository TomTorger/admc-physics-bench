cmake_minimum_required(VERSION 3.20)
project(scalar_impulse_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
option(BUILD_TESTING "Build tests" ON)

FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3)
FetchContent_MakeAvailable(benchmark)

add_library(core
  src/math.hpp src/types.hpp
  src/contact_gen.cc src/scenes.cc src/metrics.cc
  src/solver_baseline_vec.cc src/solver_scalar_cached.cc src/solver_scalar_soa.cc)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(core PRIVATE $<$<CONFIG:Release>:NDEBUG>)

add_executable(bench bench/bench_main.cc)
target_link_libraries(bench PRIVATE core benchmark::benchmark)

if(BUILD_TESTING)
  enable_testing()

  add_executable(test_sanity tests/test_sanity.cc)
  target_link_libraries(test_sanity PRIVATE core)
  add_test(NAME test_sanity COMMAND test_sanity)

  add_executable(test_invariants tests/test_invariants.cc)
  target_link_libraries(test_invariants PRIVATE core)
  add_test(NAME test_invariants COMMAND test_invariants)

  add_executable(test_equivalence_scalar_vs_baseline
                 tests/test_equivalence_scalar_vs_baseline.cc)
  target_link_libraries(test_equivalence_scalar_vs_baseline PRIVATE core)
  add_test(NAME test_equivalence_scalar_vs_baseline
           COMMAND test_equivalence_scalar_vs_baseline)

  add_executable(test_friction_cone_and_energy tests/test_friction_cone_and_energy.cc)
  target_link_libraries(test_friction_cone_and_energy PRIVATE core)
  add_test(NAME test_friction_cone_and_energy COMMAND test_friction_cone_and_energy)
endif()
