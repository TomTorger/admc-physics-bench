#include "metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
std::array<math::Vec3, 10> make_directions() {
  const std::array<math::Vec3, 10> raw = {
      math::Vec3(1.0, 0.0, 0.0), math::Vec3(-1.0, 0.0, 0.0),
      math::Vec3(0.0, 1.0, 0.0), math::Vec3(0.0, -1.0, 0.0),
      math::Vec3(0.0, 0.0, 1.0), math::Vec3(0.0, 0.0, -1.0),
      math::Vec3(0.531176, -0.847845, 0.013421),
      math::Vec3(-0.271321, 0.349121, 0.897932),
      math::Vec3(0.713512, 0.204913, -0.669324),
      math::Vec3(-0.402156, -0.551249, -0.730441)};

  std::array<math::Vec3, 10> dirs = raw;
  for (math::Vec3& d : dirs) {
    d = math::normalize_safe(d);
  }
  return dirs;
}

const std::array<math::Vec3, 10> kDirections = make_directions();

inline double directional_sum(const std::vector<RigidBody>& bodies,
                              const math::Vec3& dir) {
  double sum = 0.0;
  for (const RigidBody& b : bodies) {
    if (b.invMass <= math::kEps) {
      continue;
    }
    const double mass = 1.0 / b.invMass;
    const math::Vec3 momentum = b.v * mass;
    sum += math::dot(momentum, dir);
  }
  return sum;
}
}  // namespace

Drift directional_momentum_drift(const std::vector<RigidBody>& pre,
                                 const std::vector<RigidBody>& post) {
  Drift drift;
  for (const math::Vec3& dir : kDirections) {
    const double before = directional_sum(pre, dir);
    const double after = directional_sum(post, dir);
    drift.max_abs = std::max(drift.max_abs, std::fabs(after - before));
  }
  return drift;
}

double constraint_penetration_Linf(const std::vector<Contact>& contacts) {
  double max_pen = 0.0;
  for (const Contact& c : contacts) {
    max_pen = std::max(max_pen, std::fabs(c.C));
  }
  return max_pen;
}

namespace {
double translational_energy(const RigidBody& b) {
  if (b.invMass <= math::kEps) {
    return 0.0;
  }
  const double mass = 1.0 / b.invMass;
  return 0.5 * mass * math::dot(b.v, b.v);
}

math::Mat3 invert3x3(const math::Mat3& M) {
  const double* m = M.m.data();
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                     m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (std::fabs(det) <= math::kEps) {
    return math::Mat3();
  }
  const double inv_det = 1.0 / det;
  math::Mat3 inv;
  inv.m = {{(m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det}};
  return inv;
}

double rotational_energy(const RigidBody& b) {
  if (b.invMass <= math::kEps) {
    return 0.0;
  }
  const math::Mat3 inertia = invert3x3(b.invInertiaWorld);
  const math::Vec3 L = inertia * b.w;
  return 0.5 * math::dot(b.w, L);
}

template <typename Fn>
double sum_energy(const std::vector<RigidBody>& bodies, Fn fn) {
  double sum = 0.0;
  for (const RigidBody& b : bodies) {
    sum += fn(b);
  }
  return sum;
}
}  // namespace

Energy kinetic_energy_delta(const std::vector<RigidBody>& before,
                            const std::vector<RigidBody>& after) {
  Energy e;
  e.kinetic_before = sum_energy(before, [](const RigidBody& b) {
    return translational_energy(b) + rotational_energy(b);
  });
  e.kinetic_after = sum_energy(after, [](const RigidBody& b) {
    return translational_energy(b) + rotational_energy(b);
  });
  e.delta = e.kinetic_after - e.kinetic_before;
  return e;
}

std::uint64_t state_hash64(const std::vector<RigidBody>& bodies) {
  constexpr std::uint64_t kOffset = 1469598103934665603ull;
  constexpr std::uint64_t kPrime = 1099511628211ull;
  std::uint64_t hash = kOffset;

  auto mix_double = [&hash](double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(double));
    hash ^= bits;
    hash *= kPrime;
  };

  for (const RigidBody& b : bodies) {
    mix_double(b.x.x);
    mix_double(b.x.y);
    mix_double(b.x.z);
    mix_double(b.q.w);
    mix_double(b.q.x);
    mix_double(b.q.y);
    mix_double(b.q.z);
    mix_double(b.v.x);
    mix_double(b.v.y);
    mix_double(b.v.z);
    mix_double(b.w.x);
    mix_double(b.w.y);
    mix_double(b.w.z);
  }

  return hash;
}

