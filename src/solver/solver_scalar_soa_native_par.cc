#include "solver/solver_scalar_soa_native_par.hpp"

#include "concurrency/task_pool.hpp"
#include "solver/islands.hpp"
#include "solver/thread_scratch.hpp"

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace admc {
namespace {

template <typename Container>
void copy_scalar_component(Container& dst, const Container& src, std::size_t index) {
  dst.push_back(src[index]);
}

void copy_row_entry(RowSOA& dst, const RowSOA& src, std::size_t index) {
  dst.a.push_back(src.a[index]);
  dst.b.push_back(src.b[index]);

  copy_scalar_component(dst.nx, src.nx, index);
  copy_scalar_component(dst.ny, src.ny, index);
  copy_scalar_component(dst.nz, src.nz, index);
  copy_scalar_component(dst.t1x, src.t1x, index);
  copy_scalar_component(dst.t1y, src.t1y, index);
  copy_scalar_component(dst.t1z, src.t1z, index);
  copy_scalar_component(dst.t2x, src.t2x, index);
  copy_scalar_component(dst.t2y, src.t2y, index);
  copy_scalar_component(dst.t2z, src.t2z, index);

  copy_scalar_component(dst.rax, src.rax, index);
  copy_scalar_component(dst.ray, src.ray, index);
  copy_scalar_component(dst.raz, src.raz, index);
  copy_scalar_component(dst.rbx, src.rbx, index);
  copy_scalar_component(dst.rby, src.rby, index);
  copy_scalar_component(dst.rbz, src.rbz, index);

  copy_scalar_component(dst.raxn_x, src.raxn_x, index);
  copy_scalar_component(dst.raxn_y, src.raxn_y, index);
  copy_scalar_component(dst.raxn_z, src.raxn_z, index);
  copy_scalar_component(dst.rbxn_x, src.rbxn_x, index);
  copy_scalar_component(dst.rbxn_y, src.rbxn_y, index);
  copy_scalar_component(dst.rbxn_z, src.rbxn_z, index);
  copy_scalar_component(dst.raxt1_x, src.raxt1_x, index);
  copy_scalar_component(dst.raxt1_y, src.raxt1_y, index);
  copy_scalar_component(dst.raxt1_z, src.raxt1_z, index);
  copy_scalar_component(dst.rbxt1_x, src.rbxt1_x, index);
  copy_scalar_component(dst.rbxt1_y, src.rbxt1_y, index);
  copy_scalar_component(dst.rbxt1_z, src.rbxt1_z, index);
  copy_scalar_component(dst.raxt2_x, src.raxt2_x, index);
  copy_scalar_component(dst.raxt2_y, src.raxt2_y, index);
  copy_scalar_component(dst.raxt2_z, src.raxt2_z, index);
  copy_scalar_component(dst.rbxt2_x, src.rbxt2_x, index);
  copy_scalar_component(dst.rbxt2_y, src.rbxt2_y, index);
  copy_scalar_component(dst.rbxt2_z, src.rbxt2_z, index);

  copy_scalar_component(dst.TWn_a_x, src.TWn_a_x, index);
  copy_scalar_component(dst.TWn_a_y, src.TWn_a_y, index);
  copy_scalar_component(dst.TWn_a_z, src.TWn_a_z, index);
  copy_scalar_component(dst.TWn_b_x, src.TWn_b_x, index);
  copy_scalar_component(dst.TWn_b_y, src.TWn_b_y, index);
  copy_scalar_component(dst.TWn_b_z, src.TWn_b_z, index);
  copy_scalar_component(dst.TWt1_a_x, src.TWt1_a_x, index);
  copy_scalar_component(dst.TWt1_a_y, src.TWt1_a_y, index);
  copy_scalar_component(dst.TWt1_a_z, src.TWt1_a_z, index);
  copy_scalar_component(dst.TWt1_b_x, src.TWt1_b_x, index);
  copy_scalar_component(dst.TWt1_b_y, src.TWt1_b_y, index);
  copy_scalar_component(dst.TWt1_b_z, src.TWt1_b_z, index);
  copy_scalar_component(dst.TWt2_a_x, src.TWt2_a_x, index);
  copy_scalar_component(dst.TWt2_a_y, src.TWt2_a_y, index);
  copy_scalar_component(dst.TWt2_a_z, src.TWt2_a_z, index);
  copy_scalar_component(dst.TWt2_b_x, src.TWt2_b_x, index);
  copy_scalar_component(dst.TWt2_b_y, src.TWt2_b_y, index);
  copy_scalar_component(dst.TWt2_b_z, src.TWt2_b_z, index);

  copy_scalar_component(dst.k_n, src.k_n, index);
  copy_scalar_component(dst.k_t1, src.k_t1, index);
  copy_scalar_component(dst.k_t2, src.k_t2, index);
  copy_scalar_component(dst.inv_k_n, src.inv_k_n, index);
  copy_scalar_component(dst.inv_k_t1, src.inv_k_t1, index);
  copy_scalar_component(dst.inv_k_t2, src.inv_k_t2, index);

  copy_scalar_component(dst.mu, src.mu, index);
  copy_scalar_component(dst.e, src.e, index);
  copy_scalar_component(dst.bias, src.bias, index);
  copy_scalar_component(dst.bounce, src.bounce, index);
  copy_scalar_component(dst.C, src.C, index);

  copy_scalar_component(dst.jn, src.jn, index);
  copy_scalar_component(dst.jt1, src.jt1, index);
  copy_scalar_component(dst.jt2, src.jt2, index);

  if (index < src.flags.size()) {
    dst.flags.push_back(src.flags[index]);
  }
  if (index < src.types.size()) {
    dst.types.push_back(src.types[index]);
  }
  if (index < src.indices.size()) {
    dst.indices.push_back(src.indices[index]);
  }
}

void copy_joint_entry(JointSOA& dst, const JointSOA& src, std::size_t index) {
  dst.a.push_back(src.a[index]);
  dst.b.push_back(src.b[index]);
  dst.d.push_back(src.d[index]);
  dst.ra.push_back(src.ra[index]);
  dst.rb.push_back(src.rb[index]);
  dst.k.push_back(src.k[index]);
  dst.gamma.push_back(src.gamma[index]);
  dst.bias.push_back(src.bias[index]);
  dst.j.push_back(src.j[index]);
  dst.rope.push_back(src.rope[index]);
  dst.C.push_back(src.C[index]);
  dst.rest.push_back(src.rest[index]);
  dst.beta.push_back(src.beta[index]);
  dst.indices.push_back(src.indices[index]);
}

bool should_use_parallel(const SoaParams& params) {
#if !defined(ADMC_ENABLE_PARALLEL)
  (void)params;
  return false;
#else
#if defined(ADMC_DETERMINISTIC)
  (void)params;
  return false;
#else
  if (!params.use_threads) {
    return false;
  }
  return params.thread_count > 1;
#endif
#endif
}

}  // namespace

bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                      std::vector<Contact>& contacts,
                                      RowSOA& rows,
                                      JointSOA& joints,
                                      const SoaParams& params,
                                      SolverDebugInfo* debug_info) {
  if (!should_use_parallel(params)) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  SceneView view;
  view.bodies = &bodies;
  view.contacts = &contacts;
  view.joint_rows = &joints;
  view.rows = &rows;
  const auto islands = build_islands(view);
  if (islands.size() <= 1) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  TaskPool pool(static_cast<unsigned>(params.thread_count));
  if (pool.worker_count() <= 1) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  std::atomic<bool> used_parallel{false};
  SolverDebugInfo aggregated_debug;
  if (debug_info) {
    aggregated_debug.reset();
  }
  std::mutex debug_mutex;

  for (const Island& island : islands) {
    if (island.bodies.empty() && island.rows.empty() && island.joints.empty()) {
      continue;
    }

    pool.enqueue([&, island]() {
      used_parallel.store(true, std::memory_order_relaxed);

      std::vector<RigidBody> local_bodies;
      local_bodies.reserve(island.bodies.size());
      for (int body_index : island.bodies) {
        local_bodies.push_back(bodies[static_cast<std::size_t>(body_index)]);
      }

      std::vector<Contact> local_contacts;
      local_contacts.reserve(island.contacts.size());
      for (int contact_index : island.contacts) {
        local_contacts.push_back(contacts[static_cast<std::size_t>(contact_index)]);
      }

      RowSOA local_rows;
      if (!island.rows.empty()) {
        local_rows.reserve(island.rows.size());
        for (int row_index : island.rows) {
          copy_row_entry(local_rows, rows, static_cast<std::size_t>(row_index));
        }
      }
      local_rows.N = static_cast<int>(local_rows.a.size());

      JointSOA local_joints;
      if (!island.joints.empty()) {
        local_joints.reserve(island.joints.size());
        for (int joint_index : island.joints) {
          copy_joint_entry(local_joints, joints,
                           static_cast<std::size_t>(joint_index));
        }
      }

      SoaParams local_params = params;
      local_params.use_threads = false;
      local_params.thread_count = 1;

      SolverDebugInfo island_debug;
      SolverDebugInfo* island_debug_ptr = debug_info ? &island_debug : nullptr;
      solve_scalar_soa_native(local_bodies, local_contacts, local_rows, local_joints,
                              local_params, island_debug_ptr);

      for (std::size_t i = 0; i < island.bodies.size(); ++i) {
        const int global_index = island.bodies[i];
        bodies[static_cast<std::size_t>(global_index)] = local_bodies[i];
      }

      for (std::size_t i = 0; i < island.rows.size(); ++i) {
        const int global_index = island.rows[i];
        const std::size_t local_index = i;
        if (local_index < local_rows.jn.size()) {
          rows.jn[static_cast<std::size_t>(global_index)] = local_rows.jn[local_index];
        }
        if (local_index < local_rows.jt1.size()) {
          rows.jt1[static_cast<std::size_t>(global_index)] =
              local_rows.jt1[local_index];
        }
        if (local_index < local_rows.jt2.size()) {
          rows.jt2[static_cast<std::size_t>(global_index)] =
              local_rows.jt2[local_index];
        }
      }

      for (std::size_t i = 0; i < island.joints.size(); ++i) {
        const int global_index = island.joints[i];
        const std::size_t local_index = i;
        if (local_index < local_joints.j.size()) {
          joints.j[static_cast<std::size_t>(global_index)] =
              local_joints.j[local_index];
        }
      }

      if (debug_info && island_debug_ptr) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        aggregated_debug.accumulate(*island_debug_ptr);
      }
    });
  }

  pool.wait_idle();

  if (debug_info) {
    *debug_info = aggregated_debug;
  }

  return used_parallel.load(std::memory_order_relaxed);
}

bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                      std::vector<Contact>& contacts,
                                      RowSOA& rows,
                                      const SoaParams& params,
                                      SolverDebugInfo* debug_info) {
  JointSOA empty_joints;
  return solve_scalar_soa_native_parallel(bodies, contacts, rows, empty_joints,
                                          params, debug_info);
}

}  // namespace admc

