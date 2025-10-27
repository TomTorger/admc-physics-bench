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

bool invert3x3(const math::Mat3& A, math::Mat3& Ainv, double eps = 1e-12) {
  const auto& m = A.m;
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                     m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (std::fabs(det) <= eps) {
    return false;
  }
  const double inv_det = 1.0 / det;
  Ainv = math::Mat3({{(m[4] * m[8] - m[5] * m[7]) * inv_det,
                      (m[2] * m[7] - m[1] * m[8]) * inv_det,
                      (m[1] * m[5] - m[2] * m[4]) * inv_det,
                      (m[5] * m[6] - m[3] * m[8]) * inv_det,
                      (m[0] * m[8] - m[2] * m[6]) * inv_det,
                      (m[2] * m[3] - m[0] * m[5]) * inv_det,
                      (m[3] * m[7] - m[4] * m[6]) * inv_det,
                      (m[1] * m[6] - m[0] * m[7]) * inv_det,
                      (m[0] * m[4] - m[1] * m[3]) * inv_det}});
  return true;
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
    max_pen = std::max(max_pen, std::max(0.0, -c.C));
  }
  return max_pen;
}

double kinetic_energy(const std::vector<RigidBody>& bodies) {
  double energy = 0.0;
  for (const RigidBody& b : bodies) {
    if (b.invMass > math::kEps) {
      const double mass = 1.0 / b.invMass;
      energy += 0.5 * mass * math::dot(b.v, b.v);
    }
    if (math::length2(b.w) > math::kEps * math::kEps) {
      math::Mat3 inertia;
      if (invert3x3(b.invInertiaWorld, inertia)) {
        const math::Vec3 ang_momentum = inertia * b.w;
        energy += 0.5 * math::dot(b.w, ang_momentum);
      }
    }
  }
  return energy;
}

double energy_drift(const std::vector<RigidBody>& pre,
                    const std::vector<RigidBody>& post) {
  const double before = kinetic_energy(pre);
  const double after = kinetic_energy(post);
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

