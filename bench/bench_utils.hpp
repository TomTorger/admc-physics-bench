#pragma once
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <thread>

namespace bench {

// -------- High-resolution timer --------
class Timer {
 public:
  Timer() { reset(); }
  void reset() { start_ = std::chrono::steady_clock::now(); }
  double elapsed_ms() const {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

// -------- CPU & thread helpers --------
inline void set_thread_name(const std::string& name) {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#elif defined(_WIN32)
  // Windows 10+ only
  (void)name;
#else
  (void)name;
#endif
}

inline unsigned detect_hardware_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : n;
}

// Optional deterministic seed for reproducible benching
inline uint64_t make_seed(bool deterministic) {
  if (deterministic) return 0x12345678ULL;
  std::random_device rd;
  return ((uint64_t)rd() << 32) ^ rd();
}

// -------- Pretty header --------
inline void print_header(const std::string& title, int threads) {
  std::cout << "=== " << title << " ===\n";
  std::cout << "Detected threads: " << detect_hardware_threads()
            << " | Using " << threads << " thread(s)\n";
  std::cout.flush();
}

} // namespace bench
