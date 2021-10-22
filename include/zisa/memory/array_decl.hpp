// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef ARRAY_H_TH7WE
#define ARRAY_H_TH7WE

#include <zisa/config.hpp>

#include <zisa/memory/array_base_decl.hpp>
#include <zisa/memory/column_major.hpp>
#include <zisa/memory/contiguous_memory.hpp>
#include <zisa/memory/row_major.hpp>
#include <zisa/memory/shape.hpp>

#include <zisa/memory/array_view.hpp>

namespace zisa {

namespace detail {

template <class T, int n_dims, template <int N> class Indexing = row_major>
using array_super_
    = array_base<T,
                 Indexing<n_dims>,
                 contiguous_memory<T>,
                 shape_t<n_dims, typename contiguous_memory<T>::size_type>>;

} // namespace detail

template <class T, int n_dims, template <int N> class Indexing = row_major>
class array : public detail::array_super_<T, n_dims, Indexing> {
private:
  using super = detail::array_super_<T, n_dims, Indexing>;

protected:
  using shape_type = typename super::shape_type;

public:
  using super::super;

  array(const array &) = default;
  array(array &&) noexcept = default;

  array(const array_const_view<T, n_dims, Indexing> &other,
        device_type mem_loc);

  explicit array(const array_const_view<T, n_dims, Indexing> &other);
  explicit array(const array_view<T, n_dims, Indexing> &other);

  array(T *raw_ptr, const shape_type &shape);
  explicit array(const shape_type &shape,
                 device_type device = device_type::cpu);
  array(const shape_type &shape, const allocator<T> &alloc);

  array &operator=(const array &) = default;
  array &operator=(array &&) noexcept = default;

  array &operator=(const array_const_view<T, n_dims, Indexing> &);
  array &operator=(const array_view<T, n_dims, Indexing> &);

  array_view<T, n_dims, Indexing> view() {
    return array_view<T, n_dims, Indexing>(*this);
  }

  array_const_view<T, n_dims, Indexing> const_view() const {
    return array_const_view<T, n_dims, Indexing>(*this);
  }

  [[nodiscard]] static array<T, n_dims, row_major>
  load(HierarchicalReader &reader, const std::string &tag);
};

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing> empty_like(const array<T, n_dims, Indexing> &other) {
  return array<T, n_dims, Indexing>(other.shape());
}

template <class T, int n_dims, template <int N> class Indexing>
void save(HierarchicalWriter &writer,
          const array<T, n_dims, Indexing> &arr,
          const std::string &tag);

template <class T, int n_dims, template <int> class Indexing>
void copy(array<T, n_dims, Indexing> &dst,
          const array<T, n_dims, Indexing> &src) {

  return zisa::copy(array_view<T, n_dims, Indexing>(dst),
                    array_const_view<T, n_dims, Indexing>(src));
}

template <class T, int n_dims, template <int> class Indexing>
void copy(array<T, n_dims, Indexing> &dst,
          const array_view<T, n_dims, Indexing> &src) {
  return zisa::copy(array_view<T, n_dims, Indexing>(dst),
                    array_const_view<T, n_dims, Indexing>(src));
}

template <class T, int n_dims, template <int> class Indexing>
void copy(array<T, n_dims, Indexing> &dst,
          const array_const_view<T, n_dims, Indexing> &src) {
  return zisa::copy(array_view<T, n_dims, Indexing>(dst), src);
}

template <class T, int n_dims, template <int> class Indexing>
void copy(const array_view<T, n_dims, Indexing> &dst,
          const array<T, n_dims, Indexing> &src) {
  return zisa::copy(dst, array_const_view<T, n_dims, Indexing>(src));
}

} // namespace zisa
#endif /* end of include guard */
