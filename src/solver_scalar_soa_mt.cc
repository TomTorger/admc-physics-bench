#include "solver_scalar_soa_mt.hpp"

#include "concurrency/task_pool.hpp"
#include "solver/islands.hpp"
#include "solver_scalar_soa.hpp"

#include <atomic>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

using admc::TaskPool;

template <typename Container>
void copy_scalar_component(Container& dst, const Container& src, std::size_t index) {
  dst.push_back(src[index]);
}

struct ThreadLocalBuffers {
  std::vector<RigidBody> bodies;
  std::vector<Contact> contacts;
  RowSOA rows;
  JointSOA joints;
  SolverDebugInfo debug;
  std::vector<int> contact_lookup;
  std::vector<int> touched_contacts;
  std::vector<int> row_global_indices;
  std::vector<int> joint_global_indices;
};

ThreadLocalBuffers& thread_buffers() {
  thread_local ThreadLocalBuffers buffers;
  return buffers;
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

TaskPool& persistent_pool(unsigned requested_threads) {
  struct Holder {
    std::mutex mutex;
    std::unique_ptr<TaskPool> pool;
    unsigned threads = 0;
  };
  static Holder holder;

  const unsigned desired = std::max(1u, requested_threads);

  std::lock_guard<std::mutex> lock(holder.mutex);
  if (!holder.pool || holder.threads != desired) {
    holder.pool.reset();
    holder.pool = std::make_unique<TaskPool>(desired);
    holder.threads = desired;
  }
  return *holder.pool;
}

bool copy_row_entry(RowSOA& dst,
                    const RowSOA& src,
                    std::size_t index,
                    const std::vector<int>& contact_lookup) {
  int local_contact = -1;
  if (index < src.indices.size()) {
    const int global_contact = src.indices[index];
    if (global_contact >= 0 &&
        static_cast<std::size_t>(global_contact) < contact_lookup.size()) {
      local_contact = contact_lookup[static_cast<std::size_t>(global_contact)];
    }
  }
  if (local_contact < 0) {
    return false;
  }

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

  dst.indices.push_back(local_contact);
  return true;
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

}  // namespace

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  if (!should_use_parallel(params)) {
    solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
    return;
  }

  admc::SceneView view;
  view.bodies = &bodies;
  view.contacts = &contacts;
  view.joints = nullptr;
  view.joint_rows = &joints;
  view.rows = &rows;

  const auto islands = admc::build_islands(view);
  if (islands.size() <= 1) {
    solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
    return;
  }

  const unsigned thread_count = params.thread_count > 0
                                    ? static_cast<unsigned>(params.thread_count)
                                    : std::thread::hardware_concurrency();
  TaskPool& pool = persistent_pool(thread_count);
  if (pool.worker_count() <= 1) {
    solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
    return;
  }

  if (debug_info) {
    debug_info->reset();
  }

  std::atomic<bool> used_parallel{false};
  SolverDebugInfo aggregated_debug;
  if (debug_info) {
    aggregated_debug.reset();
  }
  std::mutex debug_mutex;

  for (const auto& island : islands) {
    if (island.bodies.empty() && island.contacts.empty() && island.joints.empty() &&
        island.rows.empty()) {
      continue;
    }

    pool.enqueue([&, island]() {
      used_parallel.store(true, std::memory_order_relaxed);

      ThreadLocalBuffers& buffers = thread_buffers();

      buffers.bodies.clear();
      buffers.bodies.reserve(island.bodies.size());
      for (int body_index : island.bodies) {
        if (body_index < 0 ||
            body_index >= static_cast<int>(bodies.size())) {
          continue;
        }
        buffers.bodies.push_back(bodies[static_cast<std::size_t>(body_index)]);
      }

      buffers.contacts.clear();
      buffers.contacts.reserve(island.contacts.size());
      const std::size_t global_contact_count = contacts.size();
      if (buffers.contact_lookup.size() < global_contact_count) {
        buffers.contact_lookup.resize(global_contact_count, -1);
      }
      buffers.touched_contacts.clear();
      buffers.touched_contacts.reserve(island.contacts.size());
      for (std::size_t i = 0; i < island.contacts.size(); ++i) {
        const int global_contact = island.contacts[i];
        if (global_contact < 0 ||
            static_cast<std::size_t>(global_contact) >= global_contact_count) {
          continue;
        }
        buffers.contact_lookup[static_cast<std::size_t>(global_contact)] =
            static_cast<int>(buffers.contacts.size());
        buffers.touched_contacts.push_back(global_contact);
        buffers.contacts.push_back(
            contacts[static_cast<std::size_t>(global_contact)]);
      }

      buffers.rows.clear_but_keep_capacity();
      buffers.rows.ensure_capacity(island.rows.size());
      buffers.row_global_indices.clear();
      buffers.row_global_indices.reserve(island.rows.size());
      for (int global_row : island.rows) {
        if (global_row < 0 || global_row >= rows.N) {
          continue;
        }
        if (copy_row_entry(buffers.rows, rows,
                           static_cast<std::size_t>(global_row),
                           buffers.contact_lookup)) {
          buffers.row_global_indices.push_back(global_row);
        }
      }
      buffers.rows.N = static_cast<int>(buffers.row_global_indices.size());

      buffers.joints.clear_but_keep_capacity();
      buffers.joints.ensure_capacity(island.joints.size());
      buffers.joint_global_indices.clear();
      buffers.joint_global_indices.reserve(island.joints.size());
      for (int joint_index : island.joints) {
        if (joint_index < 0 ||
            static_cast<std::size_t>(joint_index) >= joints.size()) {
          continue;
        }
        copy_joint_entry(buffers.joints, joints,
                         static_cast<std::size_t>(joint_index));
        buffers.joint_global_indices.push_back(joint_index);
      }

      SoaParams local_params = params;
      local_params.use_threads = false;
      local_params.thread_count = 1;

      SolverDebugInfo* local_debug_ptr = nullptr;
      if (debug_info) {
        buffers.debug.reset();
        local_debug_ptr = &buffers.debug;
      }

      solve_scalar_soa_scalar(buffers.bodies, buffers.contacts, buffers.rows,
                              buffers.joints, local_params, local_debug_ptr);

      for (std::size_t i = 0; i < island.bodies.size(); ++i) {
        const int global_index = island.bodies[i];
        if (global_index < 0 ||
            static_cast<std::size_t>(global_index) >= bodies.size()) {
          continue;
        }
        if (i >= buffers.bodies.size()) {
          continue;
        }
        bodies[static_cast<std::size_t>(global_index)] =
            buffers.bodies[i];
      }

      for (std::size_t i = 0; i < buffers.row_global_indices.size(); ++i) {
        const int global_row = buffers.row_global_indices[i];
        if (global_row < 0 || global_row >= rows.N) {
          continue;
        }
        const std::size_t row_index = static_cast<std::size_t>(global_row);
        if (i < buffers.rows.jn.size()) {
          rows.jn[row_index] = buffers.rows.jn[i];
        }
        if (i < buffers.rows.jt1.size()) {
          rows.jt1[row_index] = buffers.rows.jt1[i];
        }
        if (i < buffers.rows.jt2.size()) {
          rows.jt2[row_index] = buffers.rows.jt2[i];
        }
        if (i < buffers.rows.C.size()) {
          rows.C[row_index] = buffers.rows.C[i];
        }
        if (i < buffers.rows.bias.size()) {
          rows.bias[row_index] = buffers.rows.bias[i];
        }
        if (i < buffers.rows.bounce.size()) {
          rows.bounce[row_index] = buffers.rows.bounce[i];
        }
      }

      for (std::size_t i = 0; i < buffers.contacts.size(); ++i) {
        const int global_contact = island.contacts[i];
        if (global_contact < 0 ||
            static_cast<std::size_t>(global_contact) >= contacts.size()) {
          continue;
        }
        contacts[static_cast<std::size_t>(global_contact)] = buffers.contacts[i];
      }

      for (std::size_t i = 0; i < buffers.joint_global_indices.size(); ++i) {
        const int global_joint = buffers.joint_global_indices[i];
        if (global_joint < 0 ||
            static_cast<std::size_t>(global_joint) >= joints.size()) {
          continue;
        }
        if (i < buffers.joints.j.size()) {
          joints.j[static_cast<std::size_t>(global_joint)] =
              buffers.joints.j[i];
        }
      }

      for (int touched : buffers.touched_contacts) {
        if (touched < 0 || static_cast<std::size_t>(touched) >=
                               buffers.contact_lookup.size()) {
          continue;
        }
        buffers.contact_lookup[static_cast<std::size_t>(touched)] = -1;
      }
      buffers.touched_contacts.clear();

      if (local_debug_ptr) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        aggregated_debug.accumulate(*local_debug_ptr);
      }
    });
  }

  pool.wait_idle();

  if (!used_parallel.load(std::memory_order_relaxed)) {
    solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
    return;
  }

  if (debug_info) {
    *debug_info = aggregated_debug;
  }
}

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  JointSOA empty_joints;
  empty_joints.clear();
  solve_scalar_soa_mt(bodies, contacts, rows, empty_joints, params, debug_info);
}
