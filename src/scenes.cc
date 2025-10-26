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

