#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#endif

#if defined(__has_include)
#if __has_include(<immintrin.h>)
#include <immintrin.h>
#endif
#endif

#if defined(ADMC_ENABLE_AVX512) &&                                                     \
    (defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__)))
#define ADMC_HAS_AVX512 1
#endif

#if defined(ADMC_ENABLE_AVX2) &&                                                       \
    (defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)))
#define ADMC_HAS_AVX2 1
#endif

#if defined(ADMC_ENABLE_NEON) &&                                                         \
    (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define ADMC_HAS_NEON 1
#endif

#ifndef ADMC_HAS_AVX512
#define ADMC_HAS_AVX512 0
#endif

#ifndef ADMC_HAS_AVX2
#define ADMC_HAS_AVX2 0
#endif

#ifndef ADMC_HAS_NEON
#define ADMC_HAS_NEON 0
#endif

#if __cplusplus >= 202002L
#define ADMC_LIKELY(condition) (condition)
#define ADMC_UNLIKELY(condition) (condition)
#define ADMC_LIKELY_HINT [[likely]]
#define ADMC_UNLIKELY_HINT [[unlikely]]
#else
#if defined(__clang__) || defined(__GNUC__)
#define ADMC_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#define ADMC_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
#define ADMC_LIKELY(condition) (condition)
#define ADMC_UNLIKELY(condition) (condition)
#endif
#define ADMC_LIKELY_HINT
#define ADMC_UNLIKELY_HINT
#endif

#if defined(_MSC_VER)
#define ADMC_ASSUME(condition) __assume(condition)
#elif defined(__clang__)
#define ADMC_ASSUME(condition) __builtin_assume(condition)
#elif defined(__GNUC__)
#define ADMC_ASSUME(condition)                                   \
  do {                                                           \
    if (!(condition)) {                                          \
      __builtin_unreachable();                                   \
    }                                                            \
  } while (false)
#else
#define ADMC_ASSUME(condition) ((void)0)
#endif

inline void* admc_aligned_alloc(std::size_t size, std::size_t alignment) {
  if (size == 0) {
    return nullptr;
  }

  const std::size_t effective_alignment =
      alignment < alignof(void*) ? alignof(void*) : alignment;

#if defined(_MSC_VER)
  if (void* ptr = _aligned_malloc(size, effective_alignment)) {
    return ptr;
  }
  throw std::bad_alloc();
#elif defined(_POSIX_VERSION)
  void* ptr = nullptr;
  if (posix_memalign(&ptr, effective_alignment, size) != 0) {
    throw std::bad_alloc();
  }
  return ptr;
#elif defined(__cpp_aligned_new) && __cpp_aligned_new >= 201606L
  const std::size_t padded_size =
      ((size + effective_alignment - 1) / effective_alignment) *
      effective_alignment;
  if (void* ptr = std::aligned_alloc(effective_alignment, padded_size)) {
    return ptr;
  }
  throw std::bad_alloc();
#else
  if (void* ptr = std::malloc(size)) {
    return ptr;
  }
  throw std::bad_alloc();
#endif
}

inline void admc_aligned_free(void* ptr) noexcept {
  if (!ptr) {
    return;
  }

#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

