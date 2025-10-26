#include "scenes.hpp"

#include <cmath>
#include <vector>

namespace {
RigidBody make_dynamic_body(const math::Vec3& pos) {
  RigidBody body;
  body.x = pos;
  body.invMass = 1.0;
  body.invInertiaLocal = math::Mat3::identity();
  body.syncDerived();
  return body;
}

Contact make_contact(int a, int b, const math::Vec3& p, const math::Vec3& n,
                     double restitution = 0.0, double penetration = 0.0) {
  Contact c;
  c.a = a;
  c.b = b;
  c.p = p;
  c.n = n;
  c.e = restitution;
  c.penetration = penetration;
  return c;
}
}  // namespace

Scene make_two_spheres_head_on() {
  Scene scene;
  const double radius = 0.5;
  const double speed = 2.0;

  RigidBody a = make_dynamic_body(math::Vec3(-radius, 0.0, 0.0));
  RigidBody b = make_dynamic_body(math::Vec3(radius, 0.0, 0.0));

  a.v = math::Vec3(speed, 0.0, 0.0);
  b.v = math::Vec3(-speed, 0.0, 0.0);

  scene.bodies.push_back(a);
  scene.bodies.push_back(b);

  Contact c = make_contact(0, 1, math::Vec3(0.0, 0.0, 0.0), math::Vec3(1.0, 0.0, 0.0),
                           1.0, 0.0);
  scene.contacts.push_back(c);

  return scene;
}

Scene make_pendulum(int links) {
  Scene scene;
  if (links < 1) {
    links = 1;
  }

  RigidBody pivot;
  pivot.invMass = 0.0;
  pivot.invInertiaLocal = math::Mat3();
  pivot.syncDerived();
  scene.bodies.push_back(pivot);

  const double spacing = 1.0;
  for (int i = 0; i < links; ++i) {
    const double y = -(i + 1) * spacing;
    RigidBody link = make_dynamic_body(math::Vec3(0.25 * i, y, 0.0));
    link.v = math::Vec3(0.0, 0.0, 0.5);
    const int body_index = static_cast<int>(scene.bodies.size());
    scene.bodies.push_back(link);

    DistanceJoint joint;
    joint.a = body_index - 1;
    joint.b = body_index;
    joint.la = math::Vec3();
    joint.lb = math::Vec3();
    joint.rest = spacing;
    joint.compliance = 0.0;
    joint.beta = 0.2;
    scene.joints.push_back(joint);
  }

  return scene;
}

Scene make_chain_64() {
  Scene scene = make_pendulum(64);
  for (DistanceJoint& joint : scene.joints) {
    joint.compliance = 1e-8;
    joint.beta = 0.2;
  }
  return scene;
}

Scene make_rope_256() {
  Scene scene;

  RigidBody anchor;
  anchor.invMass = 0.0;
  anchor.invInertiaLocal = math::Mat3();
  anchor.syncDerived();
  scene.bodies.push_back(anchor);

  const double spacing = 0.5;
  const int count = 256;
  for (int i = 0; i < count; ++i) {
    const double x = (i + 1) * spacing;
    RigidBody node = make_dynamic_body(math::Vec3(x, 0.0, 0.0));
    node.v = math::Vec3(0.0, 0.0, 0.0);
    const int body_index = static_cast<int>(scene.bodies.size());
    scene.bodies.push_back(node);

    DistanceJoint joint;
    joint.a = body_index - 1;
    joint.b = body_index;
    joint.la = math::Vec3();
    joint.lb = math::Vec3();
    joint.rest = spacing;
    joint.compliance = 0.0;
    joint.beta = 0.0;
    joint.rope = true;
    scene.joints.push_back(joint);
  }

  return scene;
}

Scene make_spheres_box_cloud(int N) {
  Scene scene;
  scene.bodies.reserve(static_cast<std::size_t>(N) + 1);

  RigidBody ground;
  ground.invMass = 0.0;
  ground.invInertiaLocal = math::Mat3();
  ground.syncDerived();
  scene.bodies.push_back(ground);

  const int per_axis = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(N))));
  const double radius = 0.5;
  const double spacing = 2.0 * radius;
  const double half_grid = 0.5 * (per_axis - 1);
  const int total_cells = per_axis * per_axis * per_axis;
  std::vector<int> occupancy(static_cast<std::size_t>(total_cells), -1);

  int count = 0;
  for (int iz = 0; iz < per_axis && count < N; ++iz) {
    for (int iy = 0; iy < per_axis && count < N; ++iy) {
      for (int ix = 0; ix < per_axis && count < N; ++ix) {
        const int cell_index = (iz * per_axis + iy) * per_axis + ix;
        const double x = (ix - half_grid) * spacing;
        const double y = radius + iy * spacing;
        const double z = (iz - half_grid) * spacing;

        RigidBody body = make_dynamic_body(math::Vec3(x, y, z));
        const int body_index = static_cast<int>(scene.bodies.size());
        occupancy[static_cast<std::size_t>(cell_index)] = body_index;
        scene.bodies.push_back(body);

        Contact floor_contact =
            make_contact(0, body_index, math::Vec3(x, 0.0, z), math::Vec3(0.0, 1.0, 0.0));
        scene.contacts.push_back(floor_contact);

        if (ix > 0) {
          const int neighbor_cell = cell_index - 1;
          const int neighbor_index = occupancy[static_cast<std::size_t>(neighbor_cell)];
          if (neighbor_index >= 0) {
            const math::Vec3 pos_a = scene.bodies[neighbor_index].x;
            const math::Vec3 pos_b = body.x;
            const math::Vec3 midpoint = (pos_a + pos_b) * 0.5;
            const math::Vec3 normal = math::normalize_safe(pos_b - pos_a);
            scene.contacts.push_back(
                make_contact(neighbor_index, body_index, midpoint, normal));
          }
        }

        if (iy > 0) {
          const int neighbor_cell = cell_index - per_axis;
          const int neighbor_index = occupancy[static_cast<std::size_t>(neighbor_cell)];
          if (neighbor_index >= 0) {
            const math::Vec3 pos_a = scene.bodies[neighbor_index].x;
            const math::Vec3 pos_b = body.x;
            const math::Vec3 midpoint = (pos_a + pos_b) * 0.5;
            const math::Vec3 normal = math::normalize_safe(pos_b - pos_a);
            scene.contacts.push_back(
                make_contact(neighbor_index, body_index, midpoint, normal));
          }
        }

        if (iz > 0) {
          const int neighbor_cell = cell_index - per_axis * per_axis;
          const int neighbor_index = occupancy[static_cast<std::size_t>(neighbor_cell)];
          if (neighbor_index >= 0) {
            const math::Vec3 pos_a = scene.bodies[neighbor_index].x;
            const math::Vec3 pos_b = body.x;
            const math::Vec3 midpoint = (pos_a + pos_b) * 0.5;
            const math::Vec3 normal = math::normalize_safe(pos_b - pos_a);
            scene.contacts.push_back(
                make_contact(neighbor_index, body_index, midpoint, normal));
          }
        }

        ++count;
      }
    }
  }

  return scene;
}

Scene make_box_stack(int layers) {
  Scene scene;
  scene.bodies.reserve(static_cast<std::size_t>(layers) + 1);

  RigidBody ground;
  ground.invMass = 0.0;
  ground.invInertiaLocal = math::Mat3();
  ground.syncDerived();
  scene.bodies.push_back(ground);

  const double height = 1.0;
  const double half_height = 0.5 * height;

  for (int layer = 0; layer < layers; ++layer) {
    const double y = half_height + layer * height;
    RigidBody box = make_dynamic_body(math::Vec3(0.0, y, 0.0));
    const int body_index = static_cast<int>(scene.bodies.size());
    scene.bodies.push_back(box);

    if (layer == 0) {
      scene.contacts.push_back(
          make_contact(0, body_index, math::Vec3(0.0, 0.0, 0.0), math::Vec3(0.0, 1.0, 0.0)));
    } else {
      const int lower_index = body_index - 1;
      const math::Vec3 pos_lower = scene.bodies[lower_index].x;
      const math::Vec3 pos_upper = box.x;
      const math::Vec3 midpoint = (pos_lower + pos_upper) * 0.5;
      scene.contacts.push_back(make_contact(lower_index, body_index, midpoint,
                                            math::Vec3(0.0, 1.0, 0.0)));
    }
  }

  return scene;
}

