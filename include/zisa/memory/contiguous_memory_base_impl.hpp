// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef CONTIGUOUS_MEMORY_BASE_INL_H_23KMR
#define CONTIGUOUS_MEMORY_BASE_INL_H_23KMR

#include <cassert>

#include "copy_bytes.hpp"
#include "zisa/config.hpp"
#include "zisa/memory/copy.hpp"
#include "zisa/memory/device_type.hpp"
#include "zisa/memory/fill.hpp"
#include "zisa/memory/memory_location.hpp"

namespace zisa {

template <class T, class A1, class A2, class E1, class E2, class C1, class C2>
void copy(contiguous_memory_base<T, A1, E1, C1> &dst,
          const contiguous_memory_base<T, A2, E2, C2> &src) {
  LOG_ERR_IF(dst.size() != src.size(), "Mismatching sizes.");
  zisa::internal::copy(
      dst.raw(), dst.device(), src.raw(), src.device(), dst.size());
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    contiguous_memory_base(size_type n_elements, const Allocator &allocator)
    : _raw_data(nullptr), _n_elements(0), _allocator(nullptr) {

  allocate(n_elements, allocator);
  default_construct();
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    contiguous_memory_base(
        const contiguous_memory_base<T, Allocator, Equivalence, Construction>
            &other)
    : _raw_data(nullptr), _n_elements(0), _allocator(nullptr) {

  if (other._raw_data != nullptr) {
    allocate(other._n_elements, *other.allocator());
    copy_construct(other);
  }
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    contiguous_memory_base(
        contiguous_memory_base<T, Allocator, Equivalence, Construction>
            &&other) noexcept
    : _raw_data(nullptr), _n_elements(0), _allocator(nullptr) {

  if (other._raw_data != nullptr) {
    (*this) = std::move(other);
  }
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    ~contiguous_memory_base() {
  this->free();
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction> &
contiguous_memory_base<T, Allocator, Equivalence, Construction>::operator=(
    const contiguous_memory_base &other) {

  // We're the same.
  if (this == &other) {
    return (*this);
  }

  // The other is empty/null.
  if (other._raw_data == nullptr) {
    this->free();
    return *this;
  }

  // This one is empty/null.
  if (this->allocator() == nullptr) {
    allocate(other.size(), *other.allocator());
    copy(*this, other);
    return *this;
  }

  // Neither is empty/null.
  if (Equivalence::are_equivalent(*this->allocator(), *other.allocator())) {
    resize(other.size());
  } else {
    this->free();
    allocate(other.size(), *other.allocator());
  }

  copy(*this, other);
  return *this;
}

template <class T, class Allocator, class Equivalence, class Construction>
contiguous_memory_base<T, Allocator, Equivalence, Construction> &
contiguous_memory_base<T, Allocator, Equivalence, Construction>::operator=(
    contiguous_memory_base &&other) noexcept {
  this->free();

  _raw_data = other._raw_data;
  _n_elements = other._n_elements;
  _allocator = other._allocator;

  other._raw_data = nullptr;
  other._n_elements = 0;
  other._allocator = nullptr;

  return *this;
}

template <class T, class Allocator, class Equivalence, class Construction>
ANY_DEVICE_INLINE T &
contiguous_memory_base<T, Allocator, Equivalence, Construction>::operator[](
    size_type i) {
  assert(i < size());
  return _raw_data[i];
}

template <class T, class Allocator, class Equivalence, class Construction>
ANY_DEVICE_INLINE const T &
contiguous_memory_base<T, Allocator, Equivalence, Construction>::operator[](
    size_type i) const {
  assert(i < size());
  return _raw_data[i];
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::allocate(
    size_type n_elements, Allocator alloc) {

  free_allocator();
  _allocator = new Allocator(std::move(alloc));
  allocate(n_elements);
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::allocate(
    size_type n_elements) {
  _raw_data = this->allocator()->allocate(n_elements);
  this->_n_elements = n_elements;
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    free_data() {
  if (_raw_data != nullptr) {
    assert(allocator() != nullptr);

    allocator()->deallocate(_raw_data, _n_elements);
    _raw_data = nullptr;
    _n_elements = 0;
  }
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    free_allocator() {
  if (_allocator != nullptr) {
    delete _allocator;
    _allocator = nullptr;
  }
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::free() {
  free_data();
  free_allocator();
}

template <class T, class Allocator, class Equivalence, class Construction>
Allocator *
contiguous_memory_base<T, Allocator, Equivalence, Construction>::allocator() {
  return _allocator;
}

template <class T, class Allocator, class Equivalence, class Construction>
Allocator const *
contiguous_memory_base<T, Allocator, Equivalence, Construction>::allocator()
    const {
  return _allocator;
}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    default_construct() {}

template <class T, class Allocator, class Equivalence, class Construction>
void contiguous_memory_base<T, Allocator, Equivalence, Construction>::
    copy_construct(const contiguous_memory_base &other) {

  Construction::copy_construct(
      this->raw(), other.raw(), this->size(), *this->allocator());
}

template <class T, class Allocator, class Equivalence, class Construction>
bool contiguous_memory_base<T, Allocator, Equivalence, Construction>::resize(
    const size_type &n_elements) {
  if (n_elements == size()) {
    return false;
  }

  free_data();
  allocate(n_elements);

  return true;
}

template <class T, class Allocator, class Equivalence, class Construction>
device_type
contiguous_memory_base<T, Allocator, Equivalence, Construction>::device()
    const {
  return memory_location(*allocator());
}

} // namespace zisa

#endif /* end of include guard */