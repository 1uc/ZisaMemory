#ifndef CONTIGUOUS_MEMORY_BASE_INL_H_23KMR
#define CONTIGUOUS_MEMORY_BASE_INL_H_23KMR

#include <cassert>

#include "zisa/config.hpp"
#include "zisa/memory/device_type.hpp"
#include "zisa/memory/memory_location.hpp"

namespace zisa {

template <class T, class A1, class A2>
void copy(contiguous_memory_base<T, A1> &dst,
          const contiguous_memory_base<T, A2> &src) {
  assert(dst.size() == src.size());

  device_type dst_device = memory_location(*dst.allocator());
  device_type src_device = memory_location(*src.allocator());

  bool is_dst_on_cpu = dst_device == device_type::cpu;
  bool is_src_on_cpu = src_device == device_type::cpu;

  if (is_dst_on_cpu && is_src_on_cpu) {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    size_type n_elements, const Allocator &allocator)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {
  allocate(n_elements, allocator);
}

template <class T, class Allocator>
template <class A>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    const contiguous_memory_base<T, A> &other)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {
  allocate(other.size(), Allocator());
  copy(*this, other);
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    const contiguous_memory_base<T, Allocator> &other)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {
  allocate(other.n_elements, *other.allocator());
  copy(*this, other);
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    contiguous_memory_base<T, Allocator> &&other)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {

  (*this) = std::move(other);
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::~contiguous_memory_base() {
  free();
}

template <class T, class Allocator>
template <class A>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(const contiguous_memory_base<T, A> &other) {
  copy(*this, other);
  return *this;
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(const contiguous_memory_base &other) {
  copy(*this, other);
  return *this;
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(contiguous_memory_base &&other) {
  free();

  _raw_data = other._raw_data;
  n_elements = other.n_elements;
  _allocator = std::move(other._allocator);

  other._raw_data = nullptr;
  other.n_elements = 0;
  other._allocator = nullptr;

  return *this;
}

template <class T, class Allocator>
ANY_DEVICE_INLINE T &contiguous_memory_base<T, Allocator>::
operator[](size_type i) {
  return _raw_data[i];
}

template <class T, class Allocator>
ANY_DEVICE_INLINE const T &contiguous_memory_base<T, Allocator>::
operator[](size_type i) const {
  return _raw_data[i];
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::allocate(size_type n_elements,
                                                    Allocator alloc) {
  _allocator = new Allocator(std::move(alloc));
  _raw_data = this->allocator()->allocate(n_elements);
  this->n_elements = n_elements;
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::free() {
  if (_raw_data != nullptr) {
    allocator()->deallocate(_raw_data, n_elements);
  }

  if (_allocator != nullptr) {
    delete _allocator;
  }
}

template <class T, class Allocator>
Allocator *contiguous_memory_base<T, Allocator>::allocator() {
  return _allocator;
}

template <class T, class Allocator>
Allocator const *contiguous_memory_base<T, Allocator>::allocator() const {
  return _allocator;
}
} // namespace zisa

#endif /* end of include guard */
