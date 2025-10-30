#pragma once

#include "types.hpp"

#include <cstddef>
#include <vector>

namespace soa {

struct World {
  World() = default;
  explicit World(std::vector<RigidBody>& bodies_in) : bodies_(&bodies_in) {}

  bool valid() const { return bodies_ != nullptr; }
  int bodyCount() const {
    return bodies_ ? static_cast<int>(bodies_->size()) : 0;
  }
  std::vector<RigidBody>& bodies() { return *bodies_; }
  const std::vector<RigidBody>& bodies() const { return *bodies_; }
  RigidBody& body(int index) { return (*bodies_)[static_cast<std::size_t>(index)]; }
  const RigidBody& body(int index) const {
    return (*bodies_)[static_cast<std::size_t>(index)];
  }

private:
  std::vector<RigidBody>* bodies_ = nullptr;
};

struct ContactManifold {
  ContactManifold() = default;
  explicit ContactManifold(std::vector<Contact>& contacts_in)
      : contacts_(&contacts_in) {}

  bool valid() const { return contacts_ != nullptr; }
  int size() const {
    return contacts_ ? static_cast<int>(contacts_->size()) : 0;
  }
  std::vector<Contact>& contacts() { return *contacts_; }
  const std::vector<Contact>& contacts() const { return *contacts_; }
  Contact& contact(int index) {
    return (*contacts_)[static_cast<std::size_t>(index)];
  }
  const Contact& contact(int index) const {
    return (*contacts_)[static_cast<std::size_t>(index)];
  }

private:
  std::vector<Contact>* contacts_ = nullptr;
};

}  // namespace soa

