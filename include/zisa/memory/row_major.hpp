// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef ROW_MAJOR_H_5W1YS
#define ROW_MAJOR_H_5W1YS

#include "zisa/config.hpp"

namespace zisa {

template <int n_dims>
struct row_major;

template <>
struct row_major<1> {
  template <class Shape>
  ANY_DEVICE_INLINE static int_t linear_index(const Shape &, int_t i) {
    return i;
  }
};

template <>
struct row_major<2> {
  template <class Shape>
  ANY_DEVICE_INLINE static int_t
  linear_index(const Shape &shape, int_t i, int_t j) {
    return j + i * shape[1];
  }
};

template <>
struct row_major<3> {
  template <class Shape>
  ANY_DEVICE_INLINE static int_t
  linear_index(const Shape &shape, int_t i0, int_t i1, int_t i2) {
    return i2 + shape[2] * (i1 + shape[1] * i0);
  }
};

template <>
struct row_major<4> {
  template <class Shape>
  ANY_DEVICE_INLINE static int_t
  linear_index(const Shape &shape, int_t i0, int_t i1, int_t i2, int_t i3) {
    return i3 + shape[3] * (i2 + shape[2] * (i1 + shape[1] * i0));
  }
};

template <>
struct row_major<5> {
  template <class Shape>
  ANY_DEVICE_INLINE static int_t
  linear_index(const Shape &shape, int_t i0, int_t i1, int_t i2, int_t i3, int_t i4) {
    return i4 + shape[4] * (i3 + shape[3] * (i2 + shape[2] * (i1 + shape[1] * i0)));
  }
};

template <class Indexing>
struct indexing_traits;

template <>
struct indexing_traits<row_major<1>> {
  static constexpr int_t n_dims = 1;
};

template <>
struct indexing_traits<row_major<2>> {
  static constexpr int_t n_dims = 2;
};

template <>
struct indexing_traits<row_major<3>> {
  static constexpr int_t n_dims = 3;
};

template <>
struct indexing_traits<row_major<4>> {
  static constexpr int_t n_dims = 4;
};

template <>
struct indexing_traits<row_major<5>> {
  static constexpr int_t n_dims = 5;
};

} // namespace zisa
#endif /* end of include guard */
