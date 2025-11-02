#pragma once

#include "concurrency/task_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace admc {

inline std::size_t default_grain(std::size_t elements) {
  constexpr std::size_t kFallback = 4096;
  if (elements <= kFallback) {
    return std::max<std::size_t>(1, elements);
  }
  return kFallback;
}

template <typename Fn>
void parallel_for(std::size_t N, std::size_t grain, TaskPool& pool, Fn&& fn) {
  if (N == 0) {
    return;
  }
  const std::size_t effective_grain = grain > 0 ? grain : default_grain(N);
  const std::size_t total_workers = pool.worker_count();
  if (total_workers <= 1 || N <= effective_grain) {
    std::forward<Fn>(fn)(0, N);
    return;
  }

  const std::size_t chunk = std::max<std::size_t>(1, effective_grain);
  const std::size_t task_count = (N + chunk - 1) / chunk;
  auto callable = std::make_shared<std::decay_t<Fn>>(std::forward<Fn>(fn));
  std::size_t begin = 0;
  for (std::size_t task = 0; task < task_count; ++task) {
    const std::size_t end = std::min(begin + chunk, N);
    const std::size_t start = begin;
    pool.enqueue([callable, start, end]() { (*callable)(start, end); });
    begin = end;
  }
  pool.wait_idle();
}

template <typename Fn>
void parallel_for(std::size_t N, TaskPool& pool, Fn&& fn) {
  parallel_for(N, default_grain(N), pool, std::forward<Fn>(fn));
}

}  // namespace admc

