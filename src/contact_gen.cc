#include "contact_gen.hpp"

#include <algorithm>
#include <cmath>

using math::Vec3;

void build_tangent_frame(const Vec3& n, Vec3& t1, Vec3& t2) {
  Vec3 axis = (std::fabs(n.x) > std::fabs(n.y)) ? Vec3(0.0, 1.0, 0.0) : Vec3(1.0, 0.0, 0.0);
  t1 = math::cross(n, axis);
  if (math::length2(t1) <= math::kEps) {
    axis = Vec3(0.0, 0.0, 1.0);
    t1 = math::cross(n, axis);
  }
  t1 = math::normalize_safe(t1);
  t2 = math::cross(n, t1);
}

namespace {
double clamp_nonnegative(double v) {
  return (v < 0.0) ? 0.0 : v;
}

double compute_effective_mass(const RigidBody& bodyA,
                              const RigidBody& bodyB,
                              const Vec3& raXd,
                              const Vec3& rbXd,
                              double invMassA,
                              double invMassB) {
  double k = invMassA + invMassB;
  if (invMassA > math::kEps) {
    k += math::dot(raXd, bodyA.invInertiaWorld * raXd);
  }
  if (invMassB > math::kEps) {
    k += math::dot(rbXd, bodyB.invInertiaWorld * rbXd);
  }
  return k;
}
}  // namespace

void preprocess_contacts(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         double beta,
                         double slop,
                         double dt) {
  const double beta_dt = (dt > math::kEps) ? (beta / dt) : 0.0;

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  for (Contact& c : contacts) {
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.n = math::normalize_safe(c.n);
    if (math::length2(c.n) <= math::kEps) {
      c.n = Vec3(1.0, 0.0, 0.0);
    }

    build_tangent_frame(c.n, c.t1, c.t2);

    c.ra = c.p - A.x;
    c.rb = c.p - B.x;

    c.raXn = math::cross(c.ra, c.n);
    c.rbXn = math::cross(c.rb, c.n);
    c.raXt1 = math::cross(c.ra, c.t1);
    c.rbXt1 = math::cross(c.rb, c.t1);
    c.raXt2 = math::cross(c.ra, c.t2);
    c.rbXt2 = math::cross(c.rb, c.t2);

    const double depth = clamp_nonnegative(-c.C - slop);
    c.bias = -beta_dt * depth;

    c.e = std::clamp(c.e, 0.0, 1.0);
    c.mu = std::max(0.0, c.mu);

    const double restitution = std::clamp(c.e, 0.0, 1.0);
    c.e = restitution;

    const double invMassA = A.invMass;
    const double invMassB = B.invMass;

    c.k_n = compute_effective_mass(A, B, c.raXn, c.rbXn, invMassA, invMassB);
    c.k_t1 = compute_effective_mass(A, B, c.raXt1, c.rbXt1, invMassA, invMassB);
    c.k_t2 = compute_effective_mass(A, B, c.raXt2, c.rbXt2, invMassA, invMassB);

    if (c.k_n < math::kEps) {
      c.k_n = 0.0;
    }
    if (c.k_t1 < math::kEps) {
      c.k_t1 = 0.0;
    }
    if (c.k_t2 < math::kEps) {
      c.k_t2 = 0.0;
    }
  }
}

