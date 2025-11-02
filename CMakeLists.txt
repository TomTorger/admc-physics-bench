cmake_minimum_required(VERSION 3.20)
project(scalar_impulse_lab LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Enable Google Benchmark integration" OFF)
option(ADMC_ENABLE_AVX2 "Enable AVX2 kernels" ON)
option(ADMC_ENABLE_AVX512 "Enable AVX-512 kernels" OFF)
option(ADMC_ENABLE_NEON "Enable NEON kernels" ON)
option(ADMC_USE_THREADS "Enable multithreaded solver wrapper" ON)
option(ADMC_ENABLE_PARALLEL "Enable multithreaded solver" ON)
option(ADMC_DETERMINISTIC "Force deterministic single-thread behavior" OFF)

include(CheckCXXCompilerFlag)

# --- SIMD feature detection ---
set(ADMC_SIMD_FLAGS "")
set(ADMC_COMPILE_AVX2 OFF)
set(ADMC_COMPILE_AVX512 OFF)
set(ADMC_COMPILE_NEON OFF)
set(ADMC_COMPILER_HAS_FMA OFF)
set(ADMC_COMPILER_HAS_AVX512F OFF)
set(ADMC_COMPILER_HAS_NEON OFF)

if(MSVC)
  add_compile_options(/permissive- /Zc:preprocessor /Zc:externConstexpr /Zc:__cplusplus
                      /EHsc /W4)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)

  if(ADMC_ENABLE_AVX512)
    list(APPEND ADMC_SIMD_FLAGS "/arch:AVX512")
    set(ADMC_COMPILE_AVX512 ON)
    set(ADMC_COMPILE_AVX2 ON)
    set(ADMC_COMPILER_HAS_FMA ON)
  elseif(ADMC_ENABLE_AVX2)
    list(APPEND ADMC_SIMD_FLAGS "/arch:AVX2")
    set(ADMC_COMPILE_AVX2 ON)
    set(ADMC_COMPILER_HAS_FMA ON)
  endif()
else()
  if(ADMC_ENABLE_AVX2)
    check_cxx_compiler_flag("-mavx2" ADMC_COMPILER_HAS_AVX2)
    check_cxx_compiler_flag("-mfma" ADMC_COMPILER_HAS_FMA)
    if(ADMC_COMPILER_HAS_AVX2)
      list(APPEND ADMC_SIMD_FLAGS "-mavx2" "-mfma")
      set(ADMC_COMPILE_AVX2 ON)
    endif()
  endif()

  if(ADMC_ENABLE_AVX512)
    check_cxx_compiler_flag("-mavx512f" ADMC_COMPILER_HAS_AVX512F)
    if(ADMC_COMPILER_HAS_AVX512F)
      list(APPEND ADMC_SIMD_FLAGS "-mavx512f")
      if(NOT ADMC_COMPILE_AVX2 AND ADMC_COMPILER_HAS_FMA)
        list(APPEND ADMC_SIMD_FLAGS "-mfma")
      endif()
      set(ADMC_COMPILE_AVX512 ON)
    endif()
  endif()

  if(ADMC_ENABLE_NEON)
    check_cxx_compiler_flag("-mfpu=neon" ADMC_COMPILER_HAS_NEON)
    if(ADMC_COMPILER_HAS_NEON)
      list(APPEND ADMC_SIMD_FLAGS "-mfpu=neon")
      set(ADMC_COMPILE_NEON ON)
    endif()
  endif()
endif()

# --- Optional Google Benchmark ---
if(ENABLE_BENCHMARKS)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(benchmark)
endif()

# --- Core library ---
add_library(core
  src/math.hpp src/types.hpp
  src/contact_gen.cc src/scenes.cc src/metrics.cc src/metrics_micro.cc src/joints.cc
  src/solver_baseline_vec.cc src/solver_scalar_cached.cc src/solver_scalar_soa.cc
  src/solver_scalar_soa.cpp src/solver_scalar_soa_spheres.cpp
  src/solver_scalar_soa_simd.cc src/solver_scalar_soa_mt.cc
  src/solver_scalar_soa_native.cc src/solver/islands.cc
  src/solver/solver_scalar_soa_parallel.cc
  src/solver_scalar_soa_vectorized.cc src/solver_vec_lane.cpp src/soa/tiler.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(core PRIVATE $<$<CONFIG:Release>:NDEBUG>)

if(ADMC_SIMD_FLAGS)
  list(REMOVE_DUPLICATES ADMC_SIMD_FLAGS)
  target_compile_options(core PRIVATE ${ADMC_SIMD_FLAGS})
endif()

if(ADMC_COMPILE_AVX2)
  target_compile_definitions(core PUBLIC ADMC_ENABLE_AVX2)
endif()
if(ADMC_COMPILE_AVX512)
  target_compile_definitions(core PUBLIC ADMC_ENABLE_AVX512)
endif()
if(ADMC_COMPILE_NEON)
  target_compile_definitions(core PUBLIC ADMC_ENABLE_NEON)
endif()
if(ADMC_USE_THREADS)
  target_compile_definitions(core PUBLIC ADMC_USE_THREADS)
endif()
if(ADMC_ENABLE_PARALLEL)
  target_compile_definitions(core PUBLIC ADMC_ENABLE_PARALLEL=1)
endif()
if(ADMC_DETERMINISTIC)
  target_compile_definitions(core PUBLIC ADMC_DETERMINISTIC=1)
endif()

# --- Modular Bench suite (new) ---
set(BENCH_SOURCES
  bench/bench_main.cc
  bench/bench_cli.hpp
  bench/bench_runner.hpp
  bench/bench_output.hpp
  bench/bench_scenes.hpp
  bench/bench_utils.hpp
  bench/bench_csv_schema.hpp
  bench/bench_parallel.hpp
)
add_library(bench_common INTERFACE)
target_include_directories(bench_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/bench)

add_executable(bench ${BENCH_SOURCES})
target_link_libraries(bench PRIVATE core bench_common)
set_target_properties(bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bench)

# --- Legacy alias (for CI compatibility) ---
set(ADMC_BENCH_LEGACY_PATH
    "${CMAKE_BINARY_DIR}/bench/bench_main${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(
  TARGET bench POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:bench>
          ${ADMC_BENCH_LEGACY_PATH}
  BYPRODUCTS ${ADMC_BENCH_LEGACY_PATH}
  COMMENT "Copying bench binary to legacy name bench_main"
)

# --- Benchmark integration ---
if(ENABLE_BENCHMARKS)
  target_link_libraries(bench PRIVATE benchmark::benchmark)
  target_compile_definitions(bench PRIVATE ADMC_HAVE_GBENCH=1)
else()
  target_compile_definitions(bench PRIVATE ADMC_HAVE_GBENCH=0)
endif()

# --- Tests ---
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
  add_executable(test_energy_and_penetration tests/test_energy_and_penetration.cc)
  target_link_libraries(test_energy_and_penetration PRIVATE core)
  add_executable(test_parity_soa_vs_baseline tests/parity_soa_vs_baseline.cc)
  target_link_libraries(test_parity_soa_vs_baseline PRIVATE core)

  add_custom_target(tests DEPENDS
    test_invariants test_sanity test_solvers test_joints
    test_simd_parity test_thread_determinism test_quick_parity
    test_csv_header_only test_energy_and_penetration
    test_parity_soa_vs_baseline
  )

  add_test(NAME test_invariants COMMAND test_invariants)
  add_test(NAME test_sanity COMMAND test_sanity)
  add_test(NAME test_solvers COMMAND test_solvers)
  add_test(NAME test_joints COMMAND test_joints)
  add_test(NAME test_simd_parity COMMAND test_simd_parity)
  add_test(NAME test_thread_determinism COMMAND test_thread_determinism)
  add_test(NAME test_quick_parity COMMAND test_quick_parity)
  add_test(NAME test_csv_header_only COMMAND test_csv_header_only)
  add_test(NAME test_energy_and_penetration COMMAND test_energy_and_penetration)
  add_test(NAME test_parity_soa_vs_baseline COMMAND test_parity_soa_vs_baseline)
endif()
