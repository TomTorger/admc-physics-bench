#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace bench {

// A minimal general-purpose thread pool to parallelize solver tasks.
class ThreadPool {
 public:
  explicit ThreadPool(unsigned nthreads = std::thread::hardware_concurrency())
      : stop_(false) {
    const unsigned count = nthreads ? nthreads : 1;
    for (unsigned i = 0; i < count; ++i) {
      workers_.emplace_back([this, i] {
        while (true) {
          std::function<void()> job;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
          }
          try {
            job();
          } catch (...) {
            // swallow exceptions to prevent thread death
          }
        }
      });
    }
  }

  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using Ret = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<Ret()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<Ret> result = task->get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stop_) throw std::runtime_error("ThreadPool already stopped");
      jobs_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return result;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> jobs_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

// Helper for parallel for-loops — divides N iterations among T threads.
template <typename Func>
void parallel_for(unsigned threads, size_t count, Func&& fn) {
  if (threads <= 1 || count < threads * 2) {
    for (size_t i = 0; i < count; ++i) fn(i);
    return;
  }

  ThreadPool pool(threads);
  const size_t block = (count + threads - 1) / threads;
  std::vector<std::future<void>> futs;
  for (unsigned t = 0; t < threads; ++t) {
    size_t begin = t * block;
    size_t end   = std::min(count, begin + block);
    if (begin >= end) break;
    futs.push_back(pool.enqueue([begin, end, &fn] {
      for (size_t i = begin; i < end; ++i) fn(i);
    }));
  }
  for (auto& f : futs) f.get();
}

} // namespace bench
