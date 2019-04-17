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

  bool is_dst_on_cpu = memory_location(*dst.allocator()) == device_type::cpu;
  bool is_src_on_cpu = memory_location(*src.allocator()) == device_type::cpu;

  assert(is_dst_on_cpu && is_src_on_cpu);
  if (is_dst_on_cpu && is_src_on_cpu) {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

template <class T, class A1, class A2>
void copy_construct(contiguous_memory_base<T, A1> &dst,
                    const contiguous_memory_base<T, A2> &src) {
  assert(dst.size() == src.size());

  bool is_dst_on_cpu = memory_location(*dst.allocator()) == device_type::cpu;
  bool is_src_on_cpu = memory_location(*src.allocator()) == device_type::cpu;

  assert(is_dst_on_cpu && is_src_on_cpu);
  if (is_dst_on_cpu && is_src_on_cpu) {
    auto *raw = dst.raw();

    for (int_t i = 0; i < dst.size(); ++i) {
      new (raw + i) T(src[i]);
    }
  }
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    size_type n_elements, const Allocator &allocator)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {
  allocate(n_elements, allocator);
  default_construct();
}

template <class T, class Allocator>
template <class A>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    const contiguous_memory_base<T, A> &other)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {

  if (other._raw_data != nullptr) {
    allocate(other.size(), Allocator());
    copy_construct(*this, other);
  }
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    const contiguous_memory_base<T, Allocator> &other)
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {

  if (other._raw_data != nullptr) {
    allocate(other.n_elements, *other.allocator());
    copy_construct(*this, other);
  }
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::contiguous_memory_base(
    contiguous_memory_base<T, Allocator> &&other) noexcept
    : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {

  if (other._raw_data != nullptr) {
    (*this) = std::move(other);
  }
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator>::~contiguous_memory_base() {
  this->free();
}

template <class T, class Allocator>
template <class A>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(const contiguous_memory_base<T, A> &other) {

  if (other._raw_data == nullptr) {
    this->free();
    return *this;
  }

  resize(other);
  copy(*this, other);
  return *this;
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(const contiguous_memory_base &other) {
  if (other._raw_data == nullptr) {
    this->free();
    return *this;
  }

  resize(other);
  copy(*this, other);
  return *this;
}

template <class T, class Allocator>
contiguous_memory_base<T, Allocator> &contiguous_memory_base<T, Allocator>::
operator=(contiguous_memory_base &&other) noexcept {
  this->free();

  _raw_data = other._raw_data;
  n_elements = other.n_elements;
  _allocator = other._allocator;

  other._raw_data = nullptr;
  other.n_elements = 0;
  other._allocator = nullptr;

  return *this;
}

template <class T, class Allocator>
ANY_DEVICE_INLINE T &contiguous_memory_base<T, Allocator>::
operator[](size_type i) {
  assert(i < size());
  return _raw_data[i];
}

template <class T, class Allocator>
ANY_DEVICE_INLINE const T &contiguous_memory_base<T, Allocator>::
operator[](size_type i) const {
  assert(i < size());
  return _raw_data[i];
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::allocate(size_type n_elements,
                                                    Allocator alloc) {
  _allocator = new Allocator(std::move(alloc));
  allocate(n_elements);
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::allocate(size_type n_elements) {
  _raw_data = this->allocator()->allocate(n_elements);
  this->n_elements = n_elements;
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::free_data() {
  if (_raw_data != nullptr) {
    assert(allocator() != nullptr);

    allocator()->deallocate(_raw_data, n_elements);
    _raw_data = nullptr;
    n_elements = 0;
  }
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::free_allocator() {
  if (_allocator != nullptr) {
    delete _allocator;
    _allocator = nullptr;
  }
}

template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::free() {
  free_data();
  free_allocator();
}

template <class T, class Allocator>
Allocator *contiguous_memory_base<T, Allocator>::allocator() {
  return _allocator;
}

template <class T, class Allocator>
Allocator const *contiguous_memory_base<T, Allocator>::allocator() const {
  return _allocator;
}

template <class T, class Allocator>
template <class A>
bool contiguous_memory_base<T, Allocator>::resize(
    const contiguous_memory_base<T, A> &other) {

  if (_allocator == nullptr) {
    assert(other.allocator() != nullptr);
    _allocator = new Allocator(*other.allocator());
  }

  return resize(other.size());
}
template <class T, class Allocator>
void contiguous_memory_base<T, Allocator>::default_construct() {
  auto alloc = allocator();

  for (size_type i = 0; i < size(); ++i) {
    alloc->construct(&(*this)[i]);
  }
}

template <class T, class Allocator>
bool contiguous_memory_base<T, Allocator>::resize(const size_type &n_elements) {
  if (n_elements == size()) {
    return false;
  }

  free_data();
  allocate(n_elements);

  return true;
}
} // namespace zisa

#endif /* end of include guard */
