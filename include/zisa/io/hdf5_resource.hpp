// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Luc Grosheintz-Laval

#ifndef HDF5_RESOURCE_HPP_IWNQUIR
#define HDF5_RESOURCE_HPP_IWNQUIR

#include <functional>
#include <zisa/io/hdf5.hpp>

namespace zisa {
class HDF5Resource {
private:
  using free_callback_t = std::function<void(hid_t)>;

public:
  /// Tie life time of `id` to this object.
  /** Takes ownership of `id` and manages it life time by calling
   *  `free_callback` to free the resource.
   *
   *  By passing an no-op `free_callback` one can avoid the RAII semantics. This
   *  is useful when the ID is a builtin identifier whose lifetime is managed by
   *  the library itself, e.g. H5_NATIVE_DOUBLE.
   */
  HDF5Resource(hid_t id, free_callback_t free_callback);
  ;

  HDF5Resource(HDF5Resource &&other) noexcept;
  HDF5Resource(const HDF5Resource &) = delete;

  HDF5Resource &operator=(HDF5Resource &&other) noexcept;
  HDF5Resource &operator=(const HDF5Resource &other) = delete;

  virtual ~HDF5Resource();

  /// The raw resource handle.
  hid_t operator*() const;

  /// This resource will remain valid, but with RAII semantics.
  /** The purpose of this method is to allow transferring ownership
   *  to something else. For example to a library which manages the
   *  resource C-style; or uses a different resource abstraction.
   */
  void drop_ownership();

protected:
  /// Moves the internals of `other` to `*this`.
  /** This ensures that the members of `HDF5Resource` are moved
   *  correctly; and the moved from it set to an appropriate state.
   */
  void move_internals(HDF5Resource &&other);

  /// Safely frees any resources held by `this`.
  void free_internals();

private:
  hid_t id = -1;
  free_callback_t free_callback = [](hid_t) {
    throw std::runtime_error("Internal error: invalid free call-back.");
  };
};

template <class Derived>
class HDF5ResourceStaticFree : public HDF5Resource {
public:
  using HDF5Resource::HDF5Resource;
  explicit HDF5ResourceStaticFree(hid_t id)
      : HDF5Resource(id, [](hid_t dataset) { Derived::close(dataset); }) {}
};

#define ZISA_CREATE_SIMPLE_HDF5_RESOURCE(Resource, free_callback)              \
  class Resource : public HDF5ResourceStaticFree<Resource> {                   \
  private:                                                                     \
    using super = HDF5ResourceStaticFree<Resource>;                            \
                                                                               \
  public:                                                                      \
    using super::super;                                                        \
                                                                               \
  protected:                                                                   \
    static inline void close(hid_t id) { free_callback(id); }                  \
  };

ZISA_CREATE_SIMPLE_HDF5_RESOURCE(HDF5Dataset, H5D::close);
ZISA_CREATE_SIMPLE_HDF5_RESOURCE(HDF5Dataspace, H5S::close);
ZISA_CREATE_SIMPLE_HDF5_RESOURCE(HDF5Property, H5P::close);

#undef ZISA_CREATE_SIMPLE_HDF5_RESOURCE

}
#endif // HDF5_RESOURCE_HPP
