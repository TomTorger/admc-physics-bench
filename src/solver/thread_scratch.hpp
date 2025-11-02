#pragma once

#include "types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace admc {

template <typename T>
class AlignedBuffer {
 public:
  using Span = std::span<T>;
  using ConstSpan = std::span<const T>;

  Span ensure(std::size_t count) {
    if (storage_.size() < count) {
      storage_.resize(count);
    }
    return {storage_.data(), count};
  }

  Span span() { return {storage_.data(), storage_.size()}; }
  ConstSpan span() const { return {storage_.data(), storage_.size()}; }

  T* data() { return storage_.data(); }
  const T* data() const { return storage_.data(); }

 private:
  std::vector<T, AlignedAllocator<T, 64>> storage_;
};

struct ThreadScratch {
  struct BodyState {
    std::span<double> vx;
    std::span<double> vy;
    std::span<double> vz;
    std::span<double> wx;
    std::span<double> wy;
    std::span<double> wz;
  };

  BodyState acquire_body_state(std::size_t count) {
    return BodyState{
        body_vx_.ensure(count), body_vy_.ensure(count), body_vz_.ensure(count),
        body_wx_.ensure(count), body_wy_.ensure(count), body_wz_.ensure(count)};
  }

 private:
  AlignedBuffer<double> body_vx_;
  AlignedBuffer<double> body_vy_;
  AlignedBuffer<double> body_vz_;
  AlignedBuffer<double> body_wx_;
  AlignedBuffer<double> body_wy_;
  AlignedBuffer<double> body_wz_;
};

inline ThreadScratch& tls_scratch() {
  thread_local ThreadScratch scratch;
  return scratch;
}

}  // namespace admc

