#ifndef MEMORY_RESOURCE_H_DH9EB
#define MEMORY_RESOURCE_H_DH9EB

#include "zisa/memory/device_type.hpp"

namespace zisa {

template <class T>
class memory_resource {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T *;

public:
  inline pointer allocate(size_type n) { return do_allocate(n); }
  inline void deallocate(pointer ptr, size_type n) { do_deallocate(ptr, n); }

  inline device_type device() const { return do_device(); }

protected:
  virtual pointer do_allocate(size_type n) = 0;
  virtual void do_deallocate(pointer ptr, size_type n) = 0;

  virtual device_type do_device() const = 0;
};

} // namespace zisa
#endif /* end of include guard */
