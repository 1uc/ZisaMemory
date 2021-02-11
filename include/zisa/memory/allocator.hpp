#ifndef ALLOCATOR_H_EZF6H
#define ALLOCATOR_H_EZF6H

#include "zisa/memory/device_type.hpp"
#include "zisa/memory/memory_resource.hpp"
#include "zisa/memory/memory_resource_factory.hpp"

namespace zisa {

template <class T>
class allocator {
public:
  using value_type = T;
  using size_type = size_t;
  using pointer = T *;

public:
  allocator() : resource(make_memory_resource<T>(device_type::cpu)) {}
  allocator(device_type device) : resource(make_memory_resource<T>(device)) {}
  allocator(std::shared_ptr<memory_resource<T>> resource)
      : resource(std::move(resource)) {}
  allocator(const allocator &alloc) = default;
  allocator(allocator &&alloc) = default;

  inline pointer allocate(size_type n) { return resource->allocate(n); }
  inline void deallocate(pointer ptr, size_type n) {
    resource->deallocate(ptr, n);
  }

  template <class... Args>
  inline void construct(T *const xptr, Args &&...args) {
    *xptr = T(std::forward<Args>(args)...);
  }

  inline void destroy(T *const xptr) { xptr->~T(); }

  inline device_type device() const { return resource->device(); }

protected:
  std::shared_ptr<memory_resource<T>> resource;
};

template <class T>
device_type memory_location(const allocator<T> &alloc) {
  return alloc.device();
}

template <class T>
struct AllocatorEquivalence {
  static bool are_equivalent(const allocator<T> &a, const allocator<T> &b) {
    return a.device() == b.device();
  }
};

} // namespace zisa
#endif /* end of include guard */
