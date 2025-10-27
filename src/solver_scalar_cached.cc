#include "solver_scalar_cached.hpp"

#include <algorithm>
#include <cmath>

namespace {
using math::Vec3;

Vec3 ensure_tangent(const Vec3& n, const Vec3& candidate) {
  const Vec3 t = math::normalize_safe(candidate);
  if (math::length2(t) > math::kEps * math::kEps) {
    return t;
  }
  Vec3 axis = std::fabs(n.x) > 0.707 ? Vec3(0.0, 1.0, 0.0) : Vec3(1.0, 0.0, 0.0);
  Vec3 tangent = math::cross(axis, n);
  return math::normalize_safe(tangent);
}

Vec3 generate_t1(const Vec3& n) {
  if (std::fabs(n.x) < 0.57735026919) {
    return math::normalize_safe(math::cross(Vec3(1.0, 0.0, 0.0), n));
  }
  return math::normalize_safe(math::cross(Vec3(0.0, 1.0, 0.0), n));
}

}  // namespace

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         std::vector<DistanceJoint>& joints,
                         const SolverParams& params) {
  const int iterations = std::max(1, params.iterations);

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  const double beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;

  for (Contact& c : contacts) {
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.n = math::normalize_safe(c.n);
    if (math::length2(c.t1) <= math::kEps * math::kEps) {
      c.t1 = generate_t1(c.n);
    } else {
      c.t1 = ensure_tangent(c.n, c.t1);
    }
    c.t2 = math::normalize_safe(math::cross(c.n, c.t1));

    c.ra = c.p - A.x;
    c.rb = c.p - B.x;

    const math::Vec3 va = A.v + math::cross(A.w, c.ra);
    const math::Vec3 vb = B.v + math::cross(B.w, c.rb);
    const double v_rel_n_initial = math::dot(c.n, vb - va);

    if (std::fabs(c.C) <= math::kEps) {
      c.C = 0.0;
    }
    const double restitution = std::max(c.e, params.restitution);
    c.e = restitution;
    c.bias = -beta_dt * std::max(0.0, -c.C - params.slop);
    c.bounce = (v_rel_n_initial < 0.0) ? (-restitution * v_rel_n_initial) : 0.0;
    c.mu = std::max(c.mu, params.mu);

    c.ra_cross_n = math::cross(c.ra, c.n);
    c.rb_cross_n = math::cross(c.rb, c.n);
    c.ra_cross_t1 = math::cross(c.ra, c.t1);
    c.rb_cross_t1 = math::cross(c.rb, c.t1);
    c.ra_cross_t2 = math::cross(c.ra, c.t2);
    c.rb_cross_t2 = math::cross(c.rb, c.t2);

    const Vec3 Iwa_n = A.invInertiaWorld * c.ra_cross_n;
    const Vec3 Iwb_n = B.invInertiaWorld * c.rb_cross_n;
    double k_n = A.invMass + B.invMass;
    k_n += math::dot(c.ra_cross_n, Iwa_n) + math::dot(c.rb_cross_n, Iwb_n);
    c.k_n = (k_n > math::kEps) ? k_n : 1.0;

    const Vec3 Iwa_t1 = A.invInertiaWorld * c.ra_cross_t1;
    const Vec3 Iwb_t1 = B.invInertiaWorld * c.rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(c.ra_cross_t1, Iwa_t1) + math::dot(c.rb_cross_t1, Iwb_t1);
    c.k_t1 = (k_t1 > math::kEps) ? k_t1 : 1.0;

    const Vec3 Iwa_t2 = A.invInertiaWorld * c.ra_cross_t2;
    const Vec3 Iwb_t2 = B.invInertiaWorld * c.rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(c.ra_cross_t2, Iwa_t2) + math::dot(c.rb_cross_t2, Iwb_t2);
    c.k_t2 = (k_t2 > math::kEps) ? k_t2 : 1.0;

    if (c.C >= -params.slop || !params.warm_start) {
      c.jn = 0.0;
      c.jt1 = 0.0;
      c.jt2 = 0.0;
    }
  }

  build_distance_joint_rows(bodies, joints, params.dt);

  std::vector<double> joint_k(joints.size(), 1.0);
  std::vector<double> joint_gamma(joints.size(), 0.0);
  std::vector<double> joint_bias(joints.size(), 0.0);
  std::vector<uint8_t> joint_valid(joints.size(), 0);

  const double dt_sq = (params.dt > math::kEps) ? (params.dt * params.dt) : 0.0;
  const double inv_dt = (params.dt > math::kEps) ? (1.0 / params.dt) : 0.0;

  for (std::size_t i = 0; i < joints.size(); ++i) {
    DistanceJoint& j = joints[i];
    if (j.a < 0 || j.b < 0 || j.a >= static_cast<int>(bodies.size()) ||
        j.b >= static_cast<int>(bodies.size())) {
      j.jd = 0.0;
      continue;
    }

    Vec3 dir = math::normalize_safe(j.d_hat);
    if (math::length2(dir) <= math::kEps * math::kEps) {
      dir = Vec3(1.0, 0.0, 0.0);
    }
    j.d_hat = dir;

    const RigidBody& A = bodies[static_cast<std::size_t>(j.a)];
    const RigidBody& B = bodies[static_cast<std::size_t>(j.b)];

    const Vec3 ra_cross_d = math::cross(j.ra, dir);
    const Vec3 rb_cross_d = math::cross(j.rb, dir);

    double k = A.invMass + B.invMass;
    k += math::dot(ra_cross_d, A.invInertiaWorld * ra_cross_d);
    k += math::dot(rb_cross_d, B.invInertiaWorld * rb_cross_d);
    if (k <= math::kEps) {
      k = 1.0;
    }

    double gamma = 0.0;
    if (dt_sq > 0.0) {
      gamma = j.compliance / dt_sq;
    }

    double bias = 0.0;
    if (params.dt > math::kEps) {
      bias = -(j.beta * inv_dt) * j.C;
    }

    joint_valid[i] = 1;
    joint_k[i] = k;
    joint_gamma[i] = gamma;
    joint_bias[i] = bias;

    if (!params.warm_start) {
      j.jd = 0.0;
    }
  }

  if (params.warm_start) {
    for (Contact& c : contacts) {
      if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
          c.b >= static_cast<int>(bodies.size())) {
        continue;
      }
      RigidBody& A = bodies[c.a];
      RigidBody& B = bodies[c.b];
      const double accum_mag =
          std::max({std::fabs(c.jn), std::fabs(c.jt1), std::fabs(c.jt2)});
      if (accum_mag <= math::kEps) {
        continue;
      }
      const Vec3 linear = c.n * c.jn + c.t1 * c.jt1 + c.t2 * c.jt2;
      const Vec3 angular_a = c.ra_cross_n * c.jn + c.ra_cross_t1 * c.jt1 +
                             c.ra_cross_t2 * c.jt2;
      const Vec3 angular_b = c.rb_cross_n * c.jn + c.rb_cross_t1 * c.jt1 +
                             c.rb_cross_t2 * c.jt2;

      A.v -= linear * A.invMass;
      A.w -= A.invInertiaWorld * angular_a;
      B.v += linear * B.invMass;
      B.w += B.invInertiaWorld * angular_b;
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      if (!joint_valid[i]) {
        continue;
      }
      DistanceJoint& j = joints[i];
      if (std::fabs(j.jd) <= math::kEps) {
        continue;
      }
      RigidBody& A = bodies[j.a];
      RigidBody& B = bodies[j.b];
      const Vec3 impulse = j.d_hat * j.jd;
      A.applyImpulse(-impulse, j.ra);
      B.applyImpulse(impulse, j.rb);
    }
  }

  for (int it = 0; it < iterations; ++it) {
    for (Contact& c : contacts) {
      if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
          c.b >= static_cast<int>(bodies.size())) {
        continue;
      }

      RigidBody& A = bodies[c.a];
      RigidBody& B = bodies[c.b];

      const Vec3 va = A.v + math::cross(A.w, c.ra);
      const Vec3 vb = B.v + math::cross(B.w, c.rb);
      const Vec3 v_rel = vb - va;
      const double v_rel_n = math::dot(c.n, v_rel);

      const double rhs = -(v_rel_n + c.bias - c.bounce);

      const double delta_jn = rhs / c.k_n;
      const double jn_old = c.jn;
      c.jn = std::max(0.0, c.jn + delta_jn);
      const double applied_n = c.jn - jn_old;
      if (std::fabs(applied_n) > math::kEps) {
        const Vec3 linear_n = applied_n * c.n;
        const Vec3 angular_a = c.ra_cross_n * applied_n;
        const Vec3 angular_b = c.rb_cross_n * applied_n;
        A.v -= linear_n * A.invMass;
        A.w -= A.invInertiaWorld * angular_a;
        B.v += linear_n * B.invMass;
        B.w += B.invInertiaWorld * angular_b;
      }

      const Vec3 va_f = A.v + math::cross(A.w, c.ra);
      const Vec3 vb_f = B.v + math::cross(B.w, c.rb);
      const Vec3 v_rel_f = vb_f - va_f;
      const double v_rel_t1 = math::dot(c.t1, v_rel_f);
      const double v_rel_t2 = math::dot(c.t2, v_rel_f);

      double jt1_candidate = c.jt1;
      if (c.k_t1 > math::kEps) {
        jt1_candidate += (-v_rel_t1) / c.k_t1;
      }

      double jt2_candidate = c.jt2;
      if (c.k_t2 > math::kEps) {
        jt2_candidate += (-v_rel_t2) / c.k_t2;
      }

      const double friction_max = c.mu * std::max(c.jn, 0.0);
      const double jt_mag = std::sqrt(jt1_candidate * jt1_candidate +
                                      jt2_candidate * jt2_candidate);
      double scale = 1.0;
      if (jt_mag > friction_max && jt_mag > math::kEps) {
        scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
      }

      jt1_candidate *= scale;
      jt2_candidate *= scale;

      const double delta_jt1 = jt1_candidate - c.jt1;
      const double delta_jt2 = jt2_candidate - c.jt2;

      c.jt1 = jt1_candidate;
      c.jt2 = jt2_candidate;

      if (std::fabs(delta_jt1) > math::kEps || std::fabs(delta_jt2) > math::kEps) {
        const Vec3 linear_t = delta_jt1 * c.t1 + delta_jt2 * c.t2;
        const Vec3 angular_a = c.ra_cross_t1 * delta_jt1 + c.ra_cross_t2 * delta_jt2;
        const Vec3 angular_b = c.rb_cross_t1 * delta_jt1 + c.rb_cross_t2 * delta_jt2;
        A.v -= linear_t * A.invMass;
        A.w -= A.invInertiaWorld * angular_a;
        B.v += linear_t * B.invMass;
        B.w += B.invInertiaWorld * angular_b;
      }
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      if (!joint_valid[i]) {
        continue;
      }

      DistanceJoint& j = joints[i];
      RigidBody& A = bodies[j.a];
      RigidBody& B = bodies[j.b];

      const Vec3 va = A.v + math::cross(A.w, j.ra);
      const Vec3 vb = B.v + math::cross(B.w, j.rb);
      const double v_rel_d = math::dot(j.d_hat, vb - va);

      const double denom = joint_k[i] + joint_gamma[i];
      if (denom <= math::kEps) {
        continue;
      }

      double j_new = j.jd - (v_rel_d + joint_bias[i]) / denom;
      if (j.rope && j_new < 0.0) {
        j_new = 0.0;
      }

      const double applied = j_new - j.jd;
      j.jd = j_new;

      if (std::fabs(applied) > math::kEps) {
        const Vec3 impulse = applied * j.d_hat;
        A.applyImpulse(-impulse, j.ra);
        B.applyImpulse(impulse, j.rb);
      }
    }
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         const SolverParams& params) {
  static std::vector<DistanceJoint> empty_joints;
  empty_joints.clear();
  solve_scalar_cached(bodies, contacts, empty_joints, params);
}

