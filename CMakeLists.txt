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
  add_executable(tests tests/test_invariants.cc tests/test_sanity.cc)
  target_link_libraries(tests PRIVATE core)
  add_test(NAME all_tests COMMAND tests)
endif()
