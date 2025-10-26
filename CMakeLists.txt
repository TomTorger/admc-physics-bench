cmake_minimum_required(VERSION 3.20)
project(scalar_impulse_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Enable Google Benchmark integration" OFF)
option(ADMC_USE_AVX2 "Enable AVX2 kernels" ON)
option(ADMC_USE_NEON "Enable NEON kernels" ON)
option(ADMC_USE_THREADS "Enable multithreaded solver wrapper" ON)

include(CheckCXXCompilerFlag)

set(ADMC_ENABLE_AVX2 OFF)
if(ADMC_USE_AVX2)
  check_cxx_compiler_flag("-mavx2" ADMC_COMPILER_HAS_AVX2)
  if(ADMC_COMPILER_HAS_AVX2)
    set(ADMC_ENABLE_AVX2 ON)
  endif()
endif()

set(ADMC_ENABLE_NEON OFF)
if(ADMC_USE_NEON)
  check_cxx_compiler_flag("-mfpu=neon" ADMC_COMPILER_HAS_NEON)
  if(ADMC_COMPILER_HAS_NEON)
    set(ADMC_ENABLE_NEON ON)
  endif()
endif()

if(ENABLE_BENCHMARKS)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(benchmark)
endif()

add_library(core
  src/math.hpp src/types.hpp
  src/contact_gen.cc src/scenes.cc src/metrics.cc src/metrics_micro.cc src/joints.cc
  src/solver_baseline_vec.cc src/solver_scalar_cached.cc src/solver_scalar_soa.cc
  src/solver_scalar_soa_simd.cc src/solver_scalar_soa_mt.cc)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(core PRIVATE $<$<CONFIG:Release>:NDEBUG>)

if(ADMC_ENABLE_AVX2)
  target_compile_options(core PRIVATE -mavx2)
  target_compile_definitions(core PUBLIC ADMC_USE_AVX2)
endif()

if(ADMC_ENABLE_NEON)
  target_compile_options(core PRIVATE -mfpu=neon)
  target_compile_definitions(core PUBLIC ADMC_USE_NEON)
endif()

if(ADMC_USE_THREADS)
  target_compile_definitions(core PUBLIC ADMC_USE_THREADS)
endif()

add_executable(bench bench/bench_main.cc)
target_link_libraries(bench PRIVATE core)
set_target_properties(bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
if(ENABLE_BENCHMARKS)
  target_link_libraries(bench PRIVATE benchmark::benchmark)
  target_compile_definitions(bench PRIVATE ADMC_HAVE_GBENCH=1)
else()
  target_compile_definitions(bench PRIVATE ADMC_HAVE_GBENCH=0)
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
  add_executable(test_simd_parity tests/test_simd_parity.cc)
  target_link_libraries(test_simd_parity PRIVATE core)
  add_executable(test_thread_determinism tests/test_thread_determinism.cc)
  target_link_libraries(test_thread_determinism PRIVATE core)
  add_executable(test_quick_parity tests/test_quick_parity.cc)
  target_link_libraries(test_quick_parity PRIVATE core)
  add_executable(test_csv_header_only tests/test_csv_header_only.cc)
  target_link_libraries(test_csv_header_only PRIVATE core)

  add_custom_target(tests DEPENDS test_invariants test_sanity test_solvers test_joints
                                   test_simd_parity test_thread_determinism
                                   test_quick_parity test_csv_header_only)

  add_test(NAME test_invariants COMMAND test_invariants)
  add_test(NAME test_sanity COMMAND test_sanity)
  add_test(NAME test_solvers COMMAND test_solvers)
  add_test(NAME test_joints COMMAND test_joints)
  add_test(NAME test_simd_parity COMMAND test_simd_parity)
  add_test(NAME test_thread_determinism COMMAND test_thread_determinism)
  add_test(NAME test_quick_parity COMMAND test_quick_parity)
  add_test(NAME test_csv_header_only COMMAND test_csv_header_only)
endif()
