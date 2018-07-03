#ifndef HOST_CONTIGUOUS_MEMORY_H_TF105
#define HOST_CONTIGUOUS_MEMORY_H_TF105

#include <memory>

#include "zisa/memory/contiguous_memory.hpp"
#include "zisa/memory/device_type.hpp"

namespace zisa {

template <class T>
class host_contiguous_memory : public contiguous_memory<T> {
private:
  using super = contiguous_memory<T>;

public:
  host_contiguous_memory(int n_elements)
      : super(n_elements, allocator<T>(device_type::cuda)) {}
  host_contiguous_memory(const host_contiguous_memory &other) : super(other) {}
  host_contiguous_memory(host_contiguous_memory &&other)
      : super(std::move(other)) {}

  template <class A>
  host_contiguous_memory(const contiguous_memory_base<T, A> &other)
      : super(other.size(), allocator<T>(device_type::cpu)) {

    (*this) = other;
  }

  using super::operator=;
};

} // namespace zisa

#endif /* end of include guard */
