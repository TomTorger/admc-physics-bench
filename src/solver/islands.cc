#include "solver/islands.hpp"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace admc {
namespace {

struct DisjointSet {
  explicit DisjointSet(std::size_t count) : parent(count) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    int root = x;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[x] != root) {
      const int next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  }

  void unite(int a, int b) {
    const int ra = find(a);
    const int rb = find(b);
    if (ra == rb) {
      return;
    }
    parent[rb] = ra;
  }

  std::vector<int> parent;
};

struct IslandScratch {
  std::vector<Island> islands;
  std::vector<int> body_indices;
  std::vector<int> contact_indices;
  std::vector<int> joint_indices;
  std::vector<int> row_indices;
  std::vector<int> root_ids;
  std::vector<int> body_offsets;
  std::vector<int> contact_offsets;
  std::vector<int> joint_offsets;
  std::vector<int> row_offsets;
  std::vector<int> temp_counts;
};

thread_local IslandScratch g_scratch;

void accumulate_offsets(const std::vector<int>& counts, std::vector<int>& offsets) {
  offsets.resize(counts.size() + 1);
  offsets[0] = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }
}

}  // namespace

std::vector<Island> build_islands(const SceneView& view) {
  auto* bodies = view.bodies;
  if (!bodies || bodies->empty()) {
    g_scratch.islands.clear();
    return g_scratch.islands;
  }

  const int body_count = static_cast<int>(bodies->size());
  DisjointSet dsu(static_cast<std::size_t>(body_count));

  if (view.contacts) {
    for (std::size_t i = 0; i < view.contacts->size(); ++i) {
      const Contact& c = (*view.contacts)[i];
      if (c.a >= 0 && c.b >= 0 && c.a < body_count && c.b < body_count) {
        dsu.unite(c.a, c.b);
      }
    }
  }

  if (view.joints) {
    for (const DistanceJoint& joint : *view.joints) {
      if (joint.a >= 0 && joint.b >= 0 && joint.a < body_count &&
          joint.b < body_count) {
        dsu.unite(joint.a, joint.b);
      }
    }
  }

  if (view.joint_rows) {
    const std::size_t joint_row_count = view.joint_rows->size();
    for (std::size_t i = 0; i < joint_row_count; ++i) {
      const int a = (*view.joint_rows).a[i];
      const int b = (*view.joint_rows).b[i];
      if (a >= 0 && b >= 0 && a < body_count && b < body_count) {
        dsu.unite(a, b);
      }
    }
  }

  g_scratch.root_ids.assign(body_count, -1);
  std::vector<int> unique_roots;
  unique_roots.reserve(body_count);
  for (int i = 0; i < body_count; ++i) {
    const int root = dsu.find(i);
    int& slot = g_scratch.root_ids[root];
    if (slot < 0) {
      slot = static_cast<int>(unique_roots.size());
      unique_roots.push_back(root);
    }
  }

  const std::size_t island_count = unique_roots.size();
  if (island_count <= 1) {
    g_scratch.islands.resize(1);
    g_scratch.body_indices.resize(body_count);
    std::iota(g_scratch.body_indices.begin(), g_scratch.body_indices.end(), 0);
    g_scratch.contact_indices.resize(view.contacts ? view.contacts->size() : 0);
    if (view.contacts) {
      std::iota(g_scratch.contact_indices.begin(), g_scratch.contact_indices.end(),
                0);
    }
    const std::size_t joint_single_count =
        view.joints ? view.joints->size()
                    : (view.joint_rows ? view.joint_rows->size() : 0);
    g_scratch.joint_indices.resize(joint_single_count);
    if (joint_single_count > 0) {
      std::iota(g_scratch.joint_indices.begin(), g_scratch.joint_indices.end(), 0);
    }
    const std::size_t row_count = view.rows ? view.rows->size() : 0;
    g_scratch.row_indices.resize(row_count);
    std::iota(g_scratch.row_indices.begin(), g_scratch.row_indices.end(), 0);

    Island& single = g_scratch.islands[0];
    single.bodies = std::span<int>(g_scratch.body_indices.data(), body_count);
    single.contacts =
        std::span<int>(g_scratch.contact_indices.data(),
                       view.contacts ? view.contacts->size() : 0);
    single.joints = std::span<int>(g_scratch.joint_indices.data(),
                                   joint_single_count);
    single.rows = std::span<int>(g_scratch.row_indices.data(), row_count);
    return g_scratch.islands;
  }

  std::vector<int> body_counts(island_count, 0);
  for (int i = 0; i < body_count; ++i) {
    const int root = dsu.find(i);
    const int idx = g_scratch.root_ids[root];
    ++body_counts[static_cast<std::size_t>(idx)];
  }

  accumulate_offsets(body_counts, g_scratch.body_offsets);
  g_scratch.body_indices.resize(body_count);
  std::vector<int> cursor = g_scratch.body_offsets;
  for (int i = 0; i < body_count; ++i) {
    const int root = dsu.find(i);
    const int idx = g_scratch.root_ids[root];
    g_scratch.body_indices[cursor[static_cast<std::size_t>(idx)]++] = i;
  }

  auto& counts = g_scratch.temp_counts;
  counts.assign(island_count, 0);
  const std::size_t contact_count = view.contacts ? view.contacts->size() : 0;
  for (std::size_t i = 0; i < contact_count; ++i) {
    const Contact& c = (*view.contacts)[i];
    if (c.a < 0 || c.a >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(c.a)];
    if (root >= 0) {
      ++counts[static_cast<std::size_t>(root)];
    }
  }
  accumulate_offsets(counts, g_scratch.contact_offsets);
  g_scratch.contact_indices.resize(contact_count);
  std::vector<int> contact_head = g_scratch.contact_offsets;
  for (std::size_t i = 0; i < contact_count; ++i) {
    const Contact& c = (*view.contacts)[i];
    if (c.a < 0 || c.a >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(c.a)];
    if (root < 0) {
      continue;
    }
    g_scratch.contact_indices
        [contact_head[static_cast<std::size_t>(root)]++] = static_cast<int>(i);
  }

  counts.assign(island_count, 0);
  const std::size_t joint_count =
      view.joints ? view.joints->size()
                  : (view.joint_rows ? view.joint_rows->size() : 0);
  auto joint_body = [&](std::size_t index) {
    int a = -1;
    int b = -1;
    if (view.joints) {
      const DistanceJoint& joint = (*view.joints)[index];
      a = joint.a;
      b = joint.b;
    } else if (view.joint_rows) {
      a = (*view.joint_rows).a[index];
      b = (*view.joint_rows).b[index];
    }
    return std::pair<int, int>{a, b};
  };
  for (std::size_t i = 0; i < joint_count; ++i) {
    const auto [a, b] = joint_body(i);
    if (a < 0 || a >= body_count || b < 0 || b >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(a)];
    if (root >= 0) {
      ++counts[static_cast<std::size_t>(root)];
    }
  }
  accumulate_offsets(counts, g_scratch.joint_offsets);
  g_scratch.joint_indices.resize(joint_count);
  std::vector<int> joint_head = g_scratch.joint_offsets;
  for (std::size_t i = 0; i < joint_count; ++i) {
    const auto [a, b] = joint_body(i);
    if (a < 0 || a >= body_count || b < 0 || b >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(a)];
    if (root < 0) {
      continue;
    }
    g_scratch.joint_indices
        [joint_head[static_cast<std::size_t>(root)]++] = static_cast<int>(i);
  }

  counts.assign(island_count, 0);
  const int row_count = view.rows ? view.rows->N : 0;
  for (int i = 0; i < row_count; ++i) {
    const int body = view.rows->a[static_cast<std::size_t>(i)];
    if (body < 0 || body >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(body)];
    if (root >= 0) {
      ++counts[static_cast<std::size_t>(root)];
    }
  }
  accumulate_offsets(counts, g_scratch.row_offsets);
  g_scratch.row_indices.resize(static_cast<std::size_t>(row_count));
  std::vector<int> row_head = g_scratch.row_offsets;
  for (int i = 0; i < row_count; ++i) {
    const int body = view.rows->a[static_cast<std::size_t>(i)];
    if (body < 0 || body >= body_count) {
      continue;
    }
    const int root = g_scratch.root_ids[dsu.find(body)];
    if (root < 0) {
      continue;
    }
    g_scratch.row_indices[row_head[static_cast<std::size_t>(root)]++] = i;
  }

  g_scratch.islands.resize(island_count);
  for (std::size_t i = 0; i < island_count; ++i) {
    const int body_begin = g_scratch.body_offsets[i];
    const int body_end = g_scratch.body_offsets[i + 1];
    const int contact_begin = g_scratch.contact_offsets[i];
    const int contact_end = g_scratch.contact_offsets[i + 1];
    const int joint_begin = g_scratch.joint_offsets[i];
    const int joint_end = g_scratch.joint_offsets[i + 1];
    const int row_begin = g_scratch.row_offsets[i];
    const int row_end = g_scratch.row_offsets[i + 1];
    g_scratch.islands[i].bodies = std::span<int>(
        g_scratch.body_indices.data() + body_begin,
        static_cast<std::size_t>(body_end - body_begin));
    g_scratch.islands[i].contacts = std::span<int>(
        g_scratch.contact_indices.data() + contact_begin,
        static_cast<std::size_t>(contact_end - contact_begin));
    g_scratch.islands[i].joints = std::span<int>(
        g_scratch.joint_indices.data() + joint_begin,
        static_cast<std::size_t>(joint_end - joint_begin));
    g_scratch.islands[i].rows = std::span<int>(
        g_scratch.row_indices.data() + row_begin,
        static_cast<std::size_t>(row_end - row_begin));
  }

  return g_scratch.islands;
}

}  // namespace admc

