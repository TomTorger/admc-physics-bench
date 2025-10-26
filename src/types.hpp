#pragma once

#include "math.hpp"

#include <vector>

using math::Mat3;
using math::Quat;
using math::Vec3;

//! Rigid body state used by scalar solvers.
struct RigidBody {
  Vec3 x;                 //!< World-space position.
  Quat q;                 //!< Unit orientation (Hamilton).
  Vec3 v;                 //!< Linear velocity.
  Vec3 w;                 //!< Angular velocity.
  double invMass = 0.0;   //!< Inverse mass; zero indicates static.
  Mat3 invInertiaLocal;   //!< Inverse inertia tensor in local frame.
  Mat3 invInertiaWorld;   //!< Inverse inertia tensor in world frame.

  //! Constructs a static body at the origin.
  RigidBody();

  //! Synchronizes derived quantities using the current orientation.
  void syncDerived();

  //! Applies an impulse at offset r from the center of mass.
  void applyImpulse(const Vec3& P, const Vec3& r);

  //! Integrates velocities and orientation forward by dt.
  void integrate(double dt);
};

//! Contact point description between two bodies.
struct Contact {
  int a = -1;
  int b = -1;
  Vec3 p;     //!< World contact point.
  Vec3 n;     //!< Contact normal from body a to body b (unit).
  Vec3 t1;    //!< Tangent axis 1.
  Vec3 t2;    //!< Tangent axis 2.
  Vec3 ra;    //!< Offset from body a COM to contact.
  Vec3 rb;    //!< Offset from body b COM to contact.
  Vec3 raXn;  //!< Cached ra × n.
  Vec3 rbXn;  //!< Cached rb × n.
  Vec3 raXt1; //!< Cached ra × t1.
  Vec3 rbXt1; //!< Cached rb × t1.
  Vec3 raXt2; //!< Cached ra × t2.
  Vec3 rbXt2; //!< Cached rb × t2.
  double e = 0.0;   //!< Restitution.
  double mu = 0.0;  //!< Friction coefficient.
  double C = 0.0;   //!< Constraint value (<=0 when penetrating).
  double bias = 0.0;
  double k_n = 0.0;  //!< Effective mass for normal row.
  double k_t1 = 0.0; //!< Effective mass for tangent 1.
  double k_t2 = 0.0; //!< Effective mass for tangent 2.
  double jn = 0.0;   //!< Normal impulse accumulator.
  double jt1 = 0.0;  //!< Tangent impulse accumulator (t1).
  double jt2 = 0.0;  //!< Tangent impulse accumulator (t2).
};

//! Placeholder for future structure-of-arrays rows.
struct RowSOA {
};

// ----------------------- Inline implementation -----------------------

inline RigidBody::RigidBody()
    : x(0.0, 0.0, 0.0),
      q(),
      v(0.0, 0.0, 0.0),
      w(0.0, 0.0, 0.0),
      invMass(0.0),
      invInertiaLocal(),
      invInertiaWorld() {}

inline void RigidBody::syncDerived() {
  const Mat3 R = math::from_quat(q);
  invInertiaWorld = R * invInertiaLocal * R.transposed();
}

inline void RigidBody::applyImpulse(const Vec3& P, const Vec3& r) {
  v += P * invMass;
  const Vec3 torque = math::cross(r, P);
  w += invInertiaWorld * torque;
}

inline void RigidBody::integrate(double dt) {
  x += v * dt;

  const math::Vec3 omega = w;
  const math::Quat omega_q(0.0, omega.x, omega.y, omega.z);
  const math::Quat dq = omega_q * q;
  q.w += 0.5 * dq.w * dt;
  q.x += 0.5 * dq.x * dt;
  q.y += 0.5 * dq.y * dt;
  q.z += 0.5 * dq.z * dt;
  q.normalize();

  syncDerived();
}

