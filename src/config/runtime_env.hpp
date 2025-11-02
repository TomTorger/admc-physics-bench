#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <thread>

namespace admc::config {

inline std::size_t default_chunk_size() {
  return 4096;
}

inline std::size_t chunk_size(std::size_t fallback = default_chunk_size()) {
  if (fallback == 0) {
    fallback = default_chunk_size();
  }
  if (const char* value = std::getenv("ADMC_CHUNK")) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed > 0) {
      return static_cast<std::size_t>(std::max<long>(256, parsed));
    }
  }
  return fallback;
}

inline unsigned thread_count(unsigned fallback = std::thread::hardware_concurrency()) {
  if (fallback == 0) {
    fallback = 1;
  }
  if (const char* value = std::getenv("ADMC_THREADS")) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed > 0) {
      return static_cast<unsigned>(std::max<long>(1, parsed));
    }
  }
  return fallback;
}

}  // namespace admc::config

