#pragma once

#include "math.hpp"

#include <vector>
#include <cstdint>

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
  enum class Type : std::uint8_t {
    kUnknown = 0,
    kSphereSphere,
    kSpherePlane,
  };

  int a = -1;
  int b = -1;
  Vec3 p;      //!< World contact point.
  Vec3 n;      //!< Contact normal from body a to body b.
  Vec3 ra;     //!< Offset from body a COM to contact.
  Vec3 rb;     //!< Offset from body b COM to contact.
  Vec3 t1;     //!< First orthonormal tangent.
  Vec3 t2;     //!< Second orthonormal tangent.
  Vec3 ra_cross_n;
  Vec3 rb_cross_n;
  Vec3 ra_cross_t1;
  Vec3 rb_cross_t1;
  Vec3 ra_cross_t2;
  Vec3 rb_cross_t2;
  Vec3 TWn_a;
  Vec3 TWn_b;
  Vec3 TWt1_a;
  Vec3 TWt1_b;
  Vec3 TWt2_a;
  Vec3 TWt2_b;
  double e = 0.0;
  double mu = 0.0;
  double bias = 0.0;
  double bounce = 0.0; //!< Stored restitution target for scalar solvers.
  double C = 0.0; //!< Constraint violation (negative when penetrating).
  double jn = 0.0; //!< Warm-start accumulator for normal impulse.
  double jt1 = 0.0; //!< Warm-start accumulator for first friction tangent.
  double jt2 = 0.0; //!< Warm-start accumulator for second friction tangent.
  double k_n = 0.0;
  double k_t1 = 0.0;
  double k_t2 = 0.0;
  double radius_a = 0.0; //!< Optional cached radius for body a (spheres).
  double radius_b = 0.0; //!< Optional cached radius for body b (spheres).
  double plane_offset = 0.0; //!< Optional plane offset when involving a plane.
  Type type = Type::kUnknown;
};

//! Structure-of-arrays representation of contacts for batched solves.
struct RowSOA {
  int N = 0;

  std::vector<int> a;
  std::vector<int> b;

  // Contact frames (unit vectors)
  std::vector<double> nx, ny, nz;
  std::vector<double> t1x, t1y, t1z;
  std::vector<double> t2x, t2y, t2z;

  // Offsets from center of mass to contact point
  std::vector<double> rax, ray, raz;
  std::vector<double> rbx, rby, rbz;

  // Cross terms r x d
  std::vector<double> raxn_x, raxn_y, raxn_z;
  std::vector<double> rbxn_x, rbxn_y, rbxn_z;
  std::vector<double> raxt1_x, raxt1_y, raxt1_z;
  std::vector<double> rbxt1_x, rbxt1_y, rbxt1_z;
  std::vector<double> raxt2_x, raxt2_y, raxt2_z;
  std::vector<double> rbxt2_x, rbxt2_y, rbxt2_z;

  // Inertia-multiplied cross terms TW = I^-1 * (r x d)
  std::vector<double> TWn_a_x, TWn_a_y, TWn_a_z;
  std::vector<double> TWn_b_x, TWn_b_y, TWn_b_z;
  std::vector<double> TWt1_a_x, TWt1_a_y, TWt1_a_z;
  std::vector<double> TWt1_b_x, TWt1_b_y, TWt1_b_z;
  std::vector<double> TWt2_a_x, TWt2_a_y, TWt2_a_z;
  std::vector<double> TWt2_b_x, TWt2_b_y, TWt2_b_z;

  // Effective masses per direction
  std::vector<double> k_n, k_t1, k_t2;
  // Cached reciprocals avoid divisions inside the solver hot loop.
  std::vector<double> inv_k_n, inv_k_t1, inv_k_t2;

  // Material / bias terms per contact
  std::vector<double> mu, e, bias, bounce, C;

  // Accumulated impulses (warm-start)
  std::vector<double> jn, jt1, jt2;

  std::vector<int> indices; //!< Mapping back to original contact indices.

  std::size_t size() const { return static_cast<std::size_t>(N); }
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

