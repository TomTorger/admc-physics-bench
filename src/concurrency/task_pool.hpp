#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace admc {

class TaskPool {
 public:
  explicit TaskPool(unsigned workers = std::thread::hardware_concurrency())
      : worker_count_(workers == 0 ? 1u : workers) {
    if (worker_count_ <= 1u) {
      worker_count_ = 1u;
      return;
    }
    threads_.reserve(worker_count_);
    for (unsigned i = 0; i < worker_count_; ++i) {
      threads_.emplace_back([this](std::stop_token stop) { worker_loop(stop); });
    }
  }

  TaskPool(const TaskPool&) = delete;
  TaskPool& operator=(const TaskPool&) = delete;

  ~TaskPool() {
    if (threads_.empty()) {
      wait_inline();
      return;
    }
    try {
      wait_idle();
    } catch (...) {
      // Destructors must not emit exceptions.
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutting_down_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.request_stop();
    }
  }

  template <class F>
  void enqueue(F&& f) {
    if (worker_count_ <= 1u) {
      ++active_tasks_;
      try {
        std::forward<F>(f)();
      } catch (...) {
        --active_tasks_;
        throw;
      }
      --active_tasks_;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace_back(std::forward<F>(f));
    }
    cv_.notify_one();
  }

  void wait_idle() {
    if (worker_count_ <= 1u) {
      wait_inline();
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this]() {
      return (tasks_.empty() &&
              active_tasks_.load(std::memory_order_acquire) == 0);
    });
    rethrow_if_error();
  }

  unsigned worker_count() const { return worker_count_; }

 private:
  void worker_loop(std::stop_token stop) {
    while (!stop.stop_requested()) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, stop, [this]() { return shutting_down_ || !tasks_.empty(); });
        if ((shutting_down_ || stop.stop_requested()) && tasks_.empty()) {
          break;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
        ++active_tasks_;
      }

      try {
        task();
      } catch (...) {
        // Propagate exceptions by storing the first occurrence.
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!first_exception_) {
          first_exception_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        --active_tasks_;
        if (tasks_.empty() && active_tasks_.load(std::memory_order_acquire) == 0) {
          idle_cv_.notify_all();
        }
      }
    }
  }

  void wait_inline() {
    while (active_tasks_.load(std::memory_order_acquire) > 0) {
      std::this_thread::yield();
    }
    rethrow_if_error();
  }

  void rethrow_if_error();

  unsigned worker_count_ = 1;
  std::vector<std::jthread> threads_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable_any cv_;
  std::condition_variable idle_cv_;
  std::atomic<int> active_tasks_{0};
  bool shutting_down_ = false;
  std::exception_ptr first_exception_;
  std::mutex error_mutex_;
};

inline void TaskPool::rethrow_if_error() {
  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    error = first_exception_;
    if (error) {
      first_exception_ = nullptr;
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace admc

