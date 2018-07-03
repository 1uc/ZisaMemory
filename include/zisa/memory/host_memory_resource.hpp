#ifndef HOST_MEMORY_RESOURCE_H_05367
#define HOST_MEMORY_RESOURCE_H_05367

#include "zisa/config.hpp"
#include "zisa/memory/device_type.hpp"
#include "zisa/memory/memory_resource.hpp"

namespace zisa {

template <class T>
class host_memory_resource : public memory_resource<T> {
private:
  using super = memory_resource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

protected:
  virtual inline pointer do_allocate(size_type n) override {
    return (pointer)malloc(n * sizeof(T));
  }

  virtual inline void do_deallocate(pointer ptr, size_type) override {
    free(ptr);
  }

  virtual inline device_type do_device() const override {
    return device_type::cpu;
  }
};

template <class T>
device_type memory_location(const host_memory_resource<T> &resource) {
  return resource.device();
}

} // namespace zisa

#endif /* end of include guard */
