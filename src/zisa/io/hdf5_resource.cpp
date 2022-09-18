// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Luc Grosheintz-Laval

#if ZISA_HAS_HDF5

#include <zisa/io/hdf5_resource.hpp>

namespace zisa {

HDF5Resource::HDF5Resource(hid_t id, std::function<void(hid_t)> free_callback)
    : id(id), free_callback(std::move(free_callback)) {}

HDF5Resource::HDF5Resource(HDF5Resource &&other) noexcept {
  move_internals(std::move(other));
}

HDF5Resource &HDF5Resource::operator=(HDF5Resource &&other) noexcept {
  free_internals();
  move_internals(std::move(other));
  return *this;
}

HDF5Resource::~HDF5Resource() { free_internals(); }

hid_t HDF5Resource::operator*() const { return id; }

void HDF5Resource::drop_ownership() {
  free_callback = [](hid_t) {};
}
void HDF5Resource::free_internals() {
  auto lock = std::lock_guard(hdf5_mutex);
  if (H5Iis_valid(id)) {
    free_callback(id);
  }
}

void HDF5Resource::move_internals(HDF5Resource &&other) {
  id = other.id;
  free_callback = std::move(other.free_callback);
  other.drop_ownership();
}

}
#endif