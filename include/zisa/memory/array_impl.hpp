// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef ARRAY_IMPL_H_1WC9M
#define ARRAY_IMPL_H_1WC9M

#include "zisa/memory/array_base.hpp"
#include "zisa/memory/array_decl.hpp"

namespace zisa {
template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(T *raw_ptr, const shape_type &shape)
    : super(shape, contiguous_memory<T>(product(shape))) {

  auto n = product(shape);
  for (int_t i = 0; i < n; ++i) {
    (*this)[i] = raw_ptr[i];
  }
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(const shape_type &shape, device_type device)
    : super(shape, contiguous_memory<T>(product(shape), device)) {}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(const shape_type &shape,
                                  const allocator<T> &alloc)
    : super(shape, contiguous_memory<T>(product(shape), alloc)) {}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(
    const array_const_view<T, n_dims, Indexing> &other)
    : super(shape, contiguous_memory<T>(product(other.shape()))) {
  std::copy(other.begin(), other.end(), this->begin());
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(const array_view<T, n_dims, Indexing> &other)
    : super(shape, contiguous_memory<T>(product(other.shape()))) {
  std::copy(other.begin(), other.end(), this->begin());
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing> &array<T, n_dims, Indexing>::operator=(
    const array_const_view<T, n_dims, Indexing> &other) {

  if (this->shape() != other.shape()) {
    (*this) = array<T, n_dims, Indexing>(other.shape());
  }

  // It's pointing to *all* of this array.
  if (raw_ptr(*this) == raw_ptr(other)) {
    return *this;
  }

  std::copy(other.begin(), other.end(), this->begin());

  return *this;
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing> &array<T, n_dims, Indexing>::operator=(
    const array_view<T, n_dims, Indexing> &other) {

  (*this) = array_const_view<T, n_dims, Indexing>(other);
  return *this;
}

template <class T, int n_dims, template <int> class Indexing>
void save(HierarchicalWriter &writer,
          const array<T, n_dims, Indexing> &arr,
          const std::string &tag) {

  save(writer, arr.const_view(), tag);
}

template <class T, int n_dims>
void load_impl(HierarchicalReader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               split_array_dispatch_tag) {

  using scalar_type = typename array_save_traits<T>::scalar_type;
  auto datatype = erase_data_type<scalar_type>();
  reader.read_array(arr.raw(), datatype, tag);
}

template <class T, int n_dims>
void load_impl(HierarchicalReader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               default_dispatch_tag) {

  auto datatype = erase_data_type<T>();
  reader.read_array(arr.raw(), datatype, tag);
}

template <class T, int n_dims>
void load_impl(HierarchicalReader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               bool_dispatch_tag) {

  using scalar_type = typename array_save_traits<T>::scalar_type;
  auto datatype = erase_data_type<scalar_type>();

  auto int_arr = array<scalar_type, n_dims, row_major>(arr.shape());
  reader.read_array(int_arr.raw(), datatype, tag);

  std::copy(int_arr.cbegin(), int_arr.cend(), arr.begin());
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, row_major>
array<T, n_dims, Indexing>::load(HierarchicalReader &reader,
                                 const std::string &tag) {

  static_assert(std::is_same<row_major<n_dims>, Indexing<n_dims>>::value,
                "This has only been implemented for row-major index order.");

  auto dims = reader.dims(tag);
  auto shape = shape_t<n_dims>{};

  // Some types T are stored as an array of fixed size. Such types may require
  // an additional dimension (at the end of `dims`).
  LOG_ERR_IF(dims.size() < n_dims, "Dimension mismatch.");
  for (int_t i = 0; i < n_dims; ++i) {
    shape[i] = dims[i];
  }

  auto arr = array<T, n_dims>(shape, device_type::cpu);
  load_impl(reader, arr, tag, typename array_save_traits<T>::dispatch_tag{});

  return arr;
}
} // namespace zisa

#endif /* end of include guard */
