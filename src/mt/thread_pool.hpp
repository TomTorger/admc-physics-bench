#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace admc::mt {

class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool tp;
    return tp;
  }

  void set_parallelism(unsigned n) {
    if (n == 0) {
      n = 1;
    }
    const unsigned current = workers_.load(std::memory_order_acquire);
    if (current == n) {
      return;
    }
    stop_workers();
    start_workers(n);
  }

  unsigned size() const {
    return workers_.load(std::memory_order_acquire);
  }

  template <class F>
  std::future<void> enqueue(F&& fn) {
    using Task = std::packaged_task<void()>;
    if (size() <= 1) {
      Task task(std::forward<F>(fn));
      auto future = task.get_future();
      task();
      return future;
    }

    Task task(std::forward<F>(fn));
    auto future = task.get_future();
    {
      std::lock_guard<std::mutex> lk(m_);
      if (stop_.load(std::memory_order_acquire)) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      q_.push(std::move(task));
    }
    cv_.notify_one();
    return future;
  }

  template <class F>
  void parallel_for(std::size_t n, F&& f, std::size_t chunk = 1) {
    if (n == 0) {
      return;
    }
    const std::size_t work_chunk = std::max<std::size_t>(1, chunk);
    const unsigned workers = size();
    if (workers <= 1 || n <= work_chunk) {
      for (std::size_t i = 0; i < n; ++i) {
        f(i);
      }
      return;
    }

    const std::size_t total_chunks = (n + work_chunk - 1) / work_chunk;
    const std::size_t tasks = std::min<std::size_t>(workers, total_chunks);
    std::atomic<std::size_t> next{0};
    auto fn_holder =
        std::make_shared<std::decay_t<F>>(std::forward<F>(f));

    std::vector<std::future<void>> futures;
    futures.reserve(tasks);
    for (std::size_t t = 0; t < tasks; ++t) {
      futures.emplace_back(
          enqueue([fn_holder, &next, n, work_chunk, total_chunks]() {
            for (;;) {
              const std::size_t chunk_index =
                  next.fetch_add(1, std::memory_order_relaxed);
              if (chunk_index >= total_chunks) {
                break;
              }
              const std::size_t begin = chunk_index * work_chunk;
              const std::size_t end = std::min(begin + work_chunk, n);
              for (std::size_t i = begin; i < end; ++i) {
                (*fn_holder)(i);
              }
            }
          }));
    }
    for (auto& future : futures) {
      future.get();
    }
  }

  ~ThreadPool() {
    stop_workers();
  }

 private:
  ThreadPool() = default;

  void start_workers(unsigned n) {
    stop_.store(false, std::memory_order_release);
    workers_.store(n, std::memory_order_release);
    pool_.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
      pool_.emplace_back([this]() {
        worker_loop();
      });
    }
  }

  void stop_workers() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& th : pool_) {
      if (th.joinable()) {
        th.join();
      }
    }
    pool_.clear();
    workers_.store(0, std::memory_order_release);
    std::queue<std::packaged_task<void()>> empty;
    {
      std::lock_guard<std::mutex> lk(m_);
      std::swap(q_, empty);
      stop_.store(false, std::memory_order_release);
    }
  }

  void worker_loop() {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this]() {
          return stop_.load(std::memory_order_acquire) || !q_.empty();
        });
        if (stop_.load(std::memory_order_acquire) && q_.empty()) {
          return;
        }
        task = std::move(q_.front());
        q_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> pool_;
  std::queue<std::packaged_task<void()>> q_;
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  std::atomic<unsigned> workers_{0};
};

}  // namespace admc::mt
