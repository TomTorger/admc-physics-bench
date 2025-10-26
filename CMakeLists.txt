cmake_minimum_required(VERSION 3.20)
project(scalar_impulse_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmark binary" ON)

if(ENABLE_BENCHMARKS)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(benchmark)
endif()

add_library(core
  src/math.hpp src/types.hpp
  src/contact_gen.cc src/scenes.cc src/metrics.cc src/joints.cc
  src/solver_baseline_vec.cc src/solver_scalar_cached.cc src/solver_scalar_soa.cc)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(core PRIVATE $<$<CONFIG:Release>:NDEBUG>)

if(ENABLE_BENCHMARKS)
  add_executable(bench bench/bench_main.cc)
  target_link_libraries(bench PRIVATE core benchmark::benchmark)
endif()

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_invariants tests/test_invariants.cc)
  target_link_libraries(test_invariants PRIVATE core)
  add_executable(test_sanity tests/test_sanity.cc)
  target_link_libraries(test_sanity PRIVATE core)
  add_executable(test_solvers tests/test_solvers.cc)
  target_link_libraries(test_solvers PRIVATE core)
  add_executable(test_joints tests/test_joints.cc)
  target_link_libraries(test_joints PRIVATE core)

  add_custom_target(tests DEPENDS test_invariants test_sanity test_solvers test_joints)

  add_test(NAME test_invariants COMMAND test_invariants)
  add_test(NAME test_sanity COMMAND test_sanity)
  add_test(NAME test_solvers COMMAND test_solvers)
  add_test(NAME test_joints COMMAND test_joints)
endif()
