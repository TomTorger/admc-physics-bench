#include "solver/solver_scalar_soa_parallel.hpp"

#include "concurrency/task_pool.hpp"
#include "solver/islands.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace admc {
namespace {

template <typename Container>
inline void copy_component(Container& dst, const Container& src, std::size_t index) {
  dst.push_back(src[index]);
}

struct ThreadScratch {
  std::vector<RigidBody> bodies;
  std::vector<Contact> contacts;
  RowSOA rows;
  JointSOA joints;

  std::vector<int> local_to_global;
  std::vector<int> global_to_local;
  std::vector<int> touched_globals;

  std::vector<int> contact_lookup;
  std::vector<int> touched_contacts;
  std::vector<int> contact_global_indices;

  std::vector<int> row_global_indices;
  std::vector<int> joint_global_indices;

  SolverDebugInfo debug;

  void ensure_body_capacity(std::size_t body_count) {
    if (global_to_local.size() < body_count) {
      global_to_local.resize(body_count, -1);
    }
  }

  void ensure_contact_capacity(std::size_t contact_count) {
    if (contact_lookup.size() < contact_count) {
      contact_lookup.resize(contact_count, -1);
    }
  }

  void reset_body_map() {
    for (int idx : touched_globals) {
      if (idx >= 0 && static_cast<std::size_t>(idx) < global_to_local.size()) {
        global_to_local[static_cast<std::size_t>(idx)] = -1;
      }
    }
    touched_globals.clear();
  }

  void reset_contact_map() {
    for (int idx : touched_contacts) {
      if (idx >= 0 && static_cast<std::size_t>(idx) < contact_lookup.size()) {
        contact_lookup[static_cast<std::size_t>(idx)] = -1;
      }
    }
    touched_contacts.clear();
  }
};

ThreadScratch& thread_scratch() {
  thread_local ThreadScratch scratch;
  return scratch;
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

void copy_row_components(RowSOA& dst, const RowSOA& src, std::size_t index) {
  copy_component(dst.nx, src.nx, index);
  copy_component(dst.ny, src.ny, index);
  copy_component(dst.nz, src.nz, index);
  copy_component(dst.t1x, src.t1x, index);
  copy_component(dst.t1y, src.t1y, index);
  copy_component(dst.t1z, src.t1z, index);
  copy_component(dst.t2x, src.t2x, index);
  copy_component(dst.t2y, src.t2y, index);
  copy_component(dst.t2z, src.t2z, index);

  copy_component(dst.rax, src.rax, index);
  copy_component(dst.ray, src.ray, index);
  copy_component(dst.raz, src.raz, index);
  copy_component(dst.rbx, src.rbx, index);
  copy_component(dst.rby, src.rby, index);
  copy_component(dst.rbz, src.rbz, index);

  copy_component(dst.raxn_x, src.raxn_x, index);
  copy_component(dst.raxn_y, src.raxn_y, index);
  copy_component(dst.raxn_z, src.raxn_z, index);
  copy_component(dst.rbxn_x, src.rbxn_x, index);
  copy_component(dst.rbxn_y, src.rbxn_y, index);
  copy_component(dst.rbxn_z, src.rbxn_z, index);
  copy_component(dst.raxt1_x, src.raxt1_x, index);
  copy_component(dst.raxt1_y, src.raxt1_y, index);
  copy_component(dst.raxt1_z, src.raxt1_z, index);
  copy_component(dst.rbxt1_x, src.rbxt1_x, index);
  copy_component(dst.rbxt1_y, src.rbxt1_y, index);
  copy_component(dst.rbxt1_z, src.rbxt1_z, index);
  copy_component(dst.raxt2_x, src.raxt2_x, index);
  copy_component(dst.raxt2_y, src.raxt2_y, index);
  copy_component(dst.raxt2_z, src.raxt2_z, index);
  copy_component(dst.rbxt2_x, src.rbxt2_x, index);
  copy_component(dst.rbxt2_y, src.rbxt2_y, index);
  copy_component(dst.rbxt2_z, src.rbxt2_z, index);

  copy_component(dst.TWn_a_x, src.TWn_a_x, index);
  copy_component(dst.TWn_a_y, src.TWn_a_y, index);
  copy_component(dst.TWn_a_z, src.TWn_a_z, index);
  copy_component(dst.TWn_b_x, src.TWn_b_x, index);
  copy_component(dst.TWn_b_y, src.TWn_b_y, index);
  copy_component(dst.TWn_b_z, src.TWn_b_z, index);
  copy_component(dst.TWt1_a_x, src.TWt1_a_x, index);
  copy_component(dst.TWt1_a_y, src.TWt1_a_y, index);
  copy_component(dst.TWt1_a_z, src.TWt1_a_z, index);
  copy_component(dst.TWt1_b_x, src.TWt1_b_x, index);
  copy_component(dst.TWt1_b_y, src.TWt1_b_y, index);
  copy_component(dst.TWt1_b_z, src.TWt1_b_z, index);
  copy_component(dst.TWt2_a_x, src.TWt2_a_x, index);
  copy_component(dst.TWt2_a_y, src.TWt2_a_y, index);
  copy_component(dst.TWt2_a_z, src.TWt2_a_z, index);
  copy_component(dst.TWt2_b_x, src.TWt2_b_x, index);
  copy_component(dst.TWt2_b_y, src.TWt2_b_y, index);
  copy_component(dst.TWt2_b_z, src.TWt2_b_z, index);

  copy_component(dst.k_n, src.k_n, index);
  copy_component(dst.k_t1, src.k_t1, index);
  copy_component(dst.k_t2, src.k_t2, index);
  copy_component(dst.inv_k_n, src.inv_k_n, index);
  copy_component(dst.inv_k_t1, src.inv_k_t1, index);
  copy_component(dst.inv_k_t2, src.inv_k_t2, index);

  copy_component(dst.mu, src.mu, index);
  copy_component(dst.e, src.e, index);
  copy_component(dst.bias, src.bias, index);
  copy_component(dst.bounce, src.bounce, index);
  copy_component(dst.C, src.C, index);

  copy_component(dst.jn, src.jn, index);
  copy_component(dst.jt1, src.jt1, index);
  copy_component(dst.jt2, src.jt2, index);

  if (index < src.flags.size()) {
    dst.flags.push_back(src.flags[index]);
  } else {
    dst.flags.push_back(0);
  }
  if (index < src.types.size()) {
    dst.types.push_back(src.types[index]);
  } else {
    dst.types.push_back(0);
  }
  if (index < src.indices.size()) {
    dst.indices.push_back(src.indices[index]);
  } else {
    dst.indices.push_back(-1);
  }
}

void copy_joint_components(JointSOA& dst, const JointSOA& src, std::size_t index) {
  copy_component(dst.d, src.d, index);
  copy_component(dst.ra, src.ra, index);
  copy_component(dst.rb, src.rb, index);
  copy_component(dst.k, src.k, index);
  copy_component(dst.gamma, src.gamma, index);
  copy_component(dst.bias, src.bias, index);
  copy_component(dst.j, src.j, index);
  copy_component(dst.rope, src.rope, index);
  copy_component(dst.C, src.C, index);
  copy_component(dst.rest, src.rest, index);
  copy_component(dst.beta, src.beta, index);
  if (index < src.indices.size()) {
    dst.indices.push_back(src.indices[index]);
  } else {
    dst.indices.push_back(-1);
  }
}

bool process_island(const Island& island,
                    std::vector<RigidBody>& bodies,
                    std::vector<Contact>& contacts,
                    RowSOA& rows,
                    JointSOA& joints,
                    const SoaParams& params,
                    ThreadScratch& scratch,
                    SolverDebugInfo* debug_out) {
  const std::size_t global_body_count = bodies.size();
  const std::size_t global_contact_count = contacts.size();

  scratch.ensure_body_capacity(global_body_count);
  scratch.ensure_contact_capacity(global_contact_count);

  scratch.bodies.clear();
  scratch.local_to_global.clear();

  for (int global_body : island.bodies) {
    if (global_body < 0 ||
        static_cast<std::size_t>(global_body) >= global_body_count) {
      continue;
    }
    const int local_index = static_cast<int>(scratch.bodies.size());
    scratch.bodies.push_back(bodies[static_cast<std::size_t>(global_body)]);
    scratch.local_to_global.push_back(global_body);
    scratch.global_to_local[static_cast<std::size_t>(global_body)] = local_index;
    scratch.touched_globals.push_back(global_body);
  }

  if (scratch.bodies.empty()) {
    scratch.reset_body_map();
    scratch.reset_contact_map();
    return false;
  }

  scratch.contacts.clear();
  scratch.contact_global_indices.clear();
  for (int global_contact : island.contacts) {
    if (global_contact < 0 ||
        static_cast<std::size_t>(global_contact) >= global_contact_count) {
      continue;
    }
    const Contact& src = contacts[static_cast<std::size_t>(global_contact)];
    int local_a = -1;
    int local_b = -1;
    if (src.a >= 0 &&
        static_cast<std::size_t>(src.a) < scratch.global_to_local.size()) {
      local_a = scratch.global_to_local[static_cast<std::size_t>(src.a)];
    }
    if (src.b >= 0 &&
        static_cast<std::size_t>(src.b) < scratch.global_to_local.size()) {
      local_b = scratch.global_to_local[static_cast<std::size_t>(src.b)];
    }
    if ((src.a >= 0 && local_a < 0) || (src.b >= 0 && local_b < 0)) {
      continue;
    }

    Contact local = src;
    local.a = local_a;
    local.b = local_b;
    const int local_index = static_cast<int>(scratch.contacts.size());
    scratch.contacts.push_back(local);
    scratch.contact_global_indices.push_back(global_contact);
    scratch.contact_lookup[static_cast<std::size_t>(global_contact)] = local_index;
    scratch.touched_contacts.push_back(global_contact);
  }

  scratch.rows.clear();
  scratch.rows.reserve(island.rows.size());
  scratch.row_global_indices.clear();
  for (int global_row : island.rows) {
    if (global_row < 0 || global_row >= rows.N) {
      continue;
    }
    const int global_a = rows.a[static_cast<std::size_t>(global_row)];
    const int global_b = rows.b[static_cast<std::size_t>(global_row)];
    int local_a = -1;
    int local_b = -1;
    if (global_a >= 0 &&
        static_cast<std::size_t>(global_a) < scratch.global_to_local.size()) {
      local_a = scratch.global_to_local[static_cast<std::size_t>(global_a)];
    }
    if (global_b >= 0 &&
        static_cast<std::size_t>(global_b) < scratch.global_to_local.size()) {
      local_b = scratch.global_to_local[static_cast<std::size_t>(global_b)];
    }
    if ((global_a >= 0 && local_a < 0) || (global_b >= 0 && local_b < 0)) {
      continue;
    }

    int local_contact = -1;
    if (static_cast<std::size_t>(global_row) < rows.indices.size()) {
      const int global_contact = rows.indices[static_cast<std::size_t>(global_row)];
      if (global_contact >= 0 &&
          static_cast<std::size_t>(global_contact) < scratch.contact_lookup.size()) {
        local_contact = scratch.contact_lookup[static_cast<std::size_t>(global_contact)];
      }
      if (global_contact >= 0 && local_contact < 0) {
        // Row references a contact that is not part of this island pack.
        continue;
      }
    }

    scratch.rows.a.push_back(local_a);
    scratch.rows.b.push_back(local_b);
    copy_row_components(scratch.rows, rows, static_cast<std::size_t>(global_row));
    if (!scratch.rows.indices.empty()) {
      scratch.rows.indices.back() = local_contact;
    }
    scratch.row_global_indices.push_back(global_row);
  }
  scratch.rows.N = static_cast<int>(scratch.row_global_indices.size());

  scratch.joints.clear();
  scratch.joint_global_indices.clear();
  for (int global_joint : island.joints) {
    if (global_joint < 0 ||
        static_cast<std::size_t>(global_joint) >= joints.size()) {
      continue;
    }
    const int global_a = joints.a[static_cast<std::size_t>(global_joint)];
    const int global_b = joints.b[static_cast<std::size_t>(global_joint)];
    int local_a = -1;
    int local_b = -1;
    if (global_a >= 0 &&
        static_cast<std::size_t>(global_a) < scratch.global_to_local.size()) {
      local_a = scratch.global_to_local[static_cast<std::size_t>(global_a)];
    }
    if (global_b >= 0 &&
        static_cast<std::size_t>(global_b) < scratch.global_to_local.size()) {
      local_b = scratch.global_to_local[static_cast<std::size_t>(global_b)];
    }
    if ((global_a >= 0 && local_a < 0) || (global_b >= 0 && local_b < 0)) {
      continue;
    }

    scratch.joints.a.push_back(local_a);
    scratch.joints.b.push_back(local_b);
    copy_joint_components(scratch.joints, joints, static_cast<std::size_t>(global_joint));
    scratch.joint_global_indices.push_back(global_joint);
  }

  if (scratch.row_global_indices.empty() && scratch.joint_global_indices.empty()) {
    scratch.reset_body_map();
    scratch.reset_contact_map();
    return false;
  }

  SoaParams local_params = params;
  local_params.use_threads = false;
  local_params.thread_count = 1;

  SolverDebugInfo* island_debug = nullptr;
  if (debug_out) {
    scratch.debug.reset();
    island_debug = &scratch.debug;
  }

  solve_scalar_soa_native(scratch.bodies,
                          scratch.contacts,
                          scratch.rows,
                          scratch.joints,
                          local_params,
                          island_debug);

  // Scatter bodies back.
  for (std::size_t local = 0; local < scratch.local_to_global.size(); ++local) {
    const int global_index = scratch.local_to_global[local];
    bodies[static_cast<std::size_t>(global_index)] = scratch.bodies[local];
  }

  // Scatter contact warm-start data.
  for (std::size_t i = 0; i < scratch.contact_global_indices.size(); ++i) {
    const int global_contact = scratch.contact_global_indices[i];
    Contact& dst = contacts[static_cast<std::size_t>(global_contact)];
    const Contact& src = scratch.contacts[i];
    dst.jn = src.jn;
    dst.jt1 = src.jt1;
    dst.jt2 = src.jt2;
    dst.bias = src.bias;
    dst.C = src.C;
    dst.bounce = src.bounce;
  }

  // Scatter row impulses.
  for (std::size_t i = 0; i < scratch.row_global_indices.size(); ++i) {
    const int global_row = scratch.row_global_indices[i];
    if (static_cast<std::size_t>(global_row) >= rows.jn.size()) {
      continue;
    }
    rows.jn[static_cast<std::size_t>(global_row)] = scratch.rows.jn[i];
    rows.jt1[static_cast<std::size_t>(global_row)] = scratch.rows.jt1[i];
    rows.jt2[static_cast<std::size_t>(global_row)] = scratch.rows.jt2[i];
    if (i < scratch.rows.C.size()) {
      rows.C[static_cast<std::size_t>(global_row)] = scratch.rows.C[i];
    }
    if (i < scratch.rows.bias.size()) {
      rows.bias[static_cast<std::size_t>(global_row)] = scratch.rows.bias[i];
    }
    if (i < scratch.rows.bounce.size()) {
      rows.bounce[static_cast<std::size_t>(global_row)] = scratch.rows.bounce[i];
    }
  }

  // Scatter joint impulses.
  for (std::size_t i = 0; i < scratch.joint_global_indices.size(); ++i) {
    const int global_joint = scratch.joint_global_indices[i];
    if (static_cast<std::size_t>(global_joint) >= joints.j.size()) {
      continue;
    }
    joints.j[static_cast<std::size_t>(global_joint)] = scratch.joints.j[i];
  }

  if (debug_out && island_debug) {
    debug_out->accumulate(*island_debug);
  }

  scratch.reset_body_map();
  scratch.reset_contact_map();
  scratch.local_to_global.clear();
  scratch.contact_global_indices.clear();
  scratch.row_global_indices.clear();
  scratch.joint_global_indices.clear();
  scratch.contacts.clear();
  scratch.rows.clear();
  scratch.rows.N = 0;
  scratch.joints.clear();
  scratch.bodies.clear();
  return true;
}

}  // namespace

bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
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
  view.joints = nullptr;
  view.joint_rows = &joints;
  view.rows = &rows;
  const auto islands = build_islands(view);
  if (islands.size() <= 1) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  std::vector<std::pair<Island, std::size_t>> tasks;
  tasks.reserve(islands.size());
  for (const Island& island : islands) {
    const std::size_t cost =
        static_cast<std::size_t>(island.rows.size()) +
        static_cast<std::size_t>(island.contacts.size()) +
        static_cast<std::size_t>(island.joints.size());
    if (cost == 0 && island.bodies.empty()) {
      continue;
    }
    tasks.emplace_back(island, cost);
  }

  if (tasks.empty()) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  std::sort(tasks.begin(), tasks.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  TaskPool pool(static_cast<unsigned>(params.thread_count));
  if (pool.worker_count() <= 1) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }

  std::atomic<std::size_t> cursor{0};
  std::atomic<bool> used_parallel{false};
  SolverDebugInfo aggregated_debug;
  if (debug_info) {
    aggregated_debug.reset();
  }
  std::mutex debug_mutex;

  const unsigned workers = pool.worker_count();
  for (unsigned w = 0; w < workers; ++w) {
    pool.enqueue([&, w]() {
      ThreadScratch& scratch = thread_scratch();
      SolverDebugInfo worker_debug;
      if (debug_info) {
        worker_debug.reset();
      }

      while (true) {
        const std::size_t index =
            cursor.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size()) {
          break;
        }

        SolverDebugInfo* dbg_ptr = debug_info ? &worker_debug : nullptr;
        const bool island_used =
            process_island(tasks[index].first,
                           bodies,
                           contacts,
                           rows,
                           joints,
                           params,
                           scratch,
                           dbg_ptr);
        if (island_used) {
          used_parallel.store(true, std::memory_order_relaxed);
        }
      }

      if (debug_info) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        aggregated_debug.accumulate(worker_debug);
      }
    });
  }

  pool.wait_idle();

  if (debug_info) {
    *debug_info = aggregated_debug;
  }

  if (!used_parallel.load(std::memory_order_relaxed)) {
    solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
    return false;
  }
  return true;
}

bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
                               std::vector<Contact>& contacts,
                               RowSOA& rows,
                               const SoaParams& params,
                               SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  return solve_scalar_soa_parallel(bodies, contacts, rows, empty_joints,
                                   params, debug_info);
}

}  // namespace admc

