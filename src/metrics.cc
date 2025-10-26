#include "metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>

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

double determinant(const math::Mat3& m) {
  const auto& a = m.m;
  return a[0] * (a[4] * a[8] - a[5] * a[7]) -
         a[1] * (a[3] * a[8] - a[5] * a[6]) +
         a[2] * (a[3] * a[7] - a[4] * a[6]);
}

bool invert(const math::Mat3& m, math::Mat3& out) {
  const auto& a = m.m;
  const double det = determinant(m);
  if (std::fabs(det) <= math::kEps) {
    return false;
  }
  const double inv_det = 1.0 / det;
  out = math::Mat3({{(a[4] * a[8] - a[5] * a[7]) * inv_det,
                     (a[2] * a[7] - a[1] * a[8]) * inv_det,
                     (a[1] * a[5] - a[2] * a[4]) * inv_det,
                     (a[5] * a[6] - a[3] * a[8]) * inv_det,
                     (a[0] * a[8] - a[2] * a[6]) * inv_det,
                     (a[2] * a[3] - a[0] * a[5]) * inv_det,
                     (a[3] * a[7] - a[4] * a[6]) * inv_det,
                     (a[1] * a[6] - a[0] * a[7]) * inv_det,
                     (a[0] * a[4] - a[1] * a[3]) * inv_det}});
  return true;
}

math::Vec3 solve(const math::Mat3& A, const math::Vec3& b) {
  math::Mat3 inv;
  if (!invert(A, inv)) {
    return math::Vec3();
  }
  return inv * b;
}

double total_energy(const std::vector<RigidBody>& bodies) {
  double energy = 0.0;
  for (const RigidBody& b : bodies) {
    if (b.invMass > math::kEps) {
      const double mass = 1.0 / b.invMass;
      energy += 0.5 * mass * math::dot(b.v, b.v);
    }
    if (math::length2(b.w) > math::kEps * math::kEps) {
      math::Vec3 L = solve(b.invInertiaWorld, b.w);
      energy += 0.5 * math::dot(b.w, L);
    }
  }
  return energy;
}

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
    max_pen = std::max(max_pen, std::fabs(c.penetration));
  }
  return max_pen;
}

double energy_drift(const std::vector<RigidBody>& pre,
                    const std::vector<RigidBody>& post) {
  const double before = total_energy(pre);
  const double after = total_energy(post);
  return after - before;
}

double cone_consistency(const std::vector<Contact>& contacts) {
  std::size_t considered = 0;
  std::size_t satisfied = 0;
  for (const Contact& c : contacts) {
    if (c.a < 0 || c.b < 0) {
      continue;
    }
    ++considered;
    const double friction_limit = c.mu * std::max(c.jn, 0.0) + 1e-12;
    const double jt_mag = std::sqrt(c.jt1 * c.jt1 + c.jt2 * c.jt2);
    if (jt_mag <= friction_limit) {
      ++satisfied;
    }
  }
  if (considered == 0) {
    return 1.0;
  }
  return static_cast<double>(satisfied) / static_cast<double>(considered);
}

double joint_error_Linf(const std::vector<DistanceJoint>& joints) {
  double max_error = 0.0;
  for (const DistanceJoint& j : joints) {
    max_error = std::max(max_error, std::fabs(j.C));
  }
  return max_error;
}

