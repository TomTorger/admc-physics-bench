#include "types.hpp"

#include "joints.hpp"

void SolverBodySoA::initialize(const std::vector<RigidBody>& bodies,
                               const RowSOA& rows,
                               const JointSOA& joints) {
  slot_of_body.assign(bodies.size(), -1);
  body_of_slot.clear();
  body_of_slot.reserve(bodies.size());

  auto enlist_body = [&](int body_index) {
    if (body_index < 0 ||
        static_cast<std::size_t>(body_index) >= bodies.size()) {
      return;
    }
    int& slot = slot_of_body[static_cast<std::size_t>(body_index)];
    if (slot != -1) {
      return;
    }
    slot = static_cast<int>(body_of_slot.size());
    body_of_slot.push_back(body_index);
  };

  for (int i = 0; i < rows.N; ++i) {
    enlist_body(rows.a[i]);
    enlist_body(rows.b[i]);
  }

  for (std::size_t i = 0; i < joints.size(); ++i) {
    enlist_body(joints.a[i]);
    enlist_body(joints.b[i]);
  }

  const std::size_t count = body_of_slot.size();
  vx.assign(count, 0.0);
  vy.assign(count, 0.0);
  vz.assign(count, 0.0);
  wx.assign(count, 0.0);
  wy.assign(count, 0.0);
  wz.assign(count, 0.0);
}

void SolverBodySoA::load_from(const std::vector<RigidBody>& bodies) {
  for (std::size_t slot = 0; slot < body_of_slot.size(); ++slot) {
    const int index = body_of_slot[slot];
    const RigidBody& body = bodies[static_cast<std::size_t>(index)];
    vx[slot] = body.v.x;
    vy[slot] = body.v.y;
    vz[slot] = body.v.z;
    wx[slot] = body.w.x;
    wy[slot] = body.w.y;
    wz[slot] = body.w.z;
  }
}

void SolverBodySoA::store_to(std::vector<RigidBody>& bodies) const {
  for (std::size_t slot = 0; slot < body_of_slot.size(); ++slot) {
    const int index = body_of_slot[slot];
    if (index < 0 || static_cast<std::size_t>(index) >= bodies.size()) {
      continue;
    }
    RigidBody& body = bodies[static_cast<std::size_t>(index)];
    body.v.x = vx[slot];
    body.v.y = vy[slot];
    body.v.z = vz[slot];
    body.w.x = wx[slot];
    body.w.y = wy[slot];
    body.w.z = wz[slot];
  }
}

