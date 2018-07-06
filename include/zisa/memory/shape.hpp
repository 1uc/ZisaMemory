#ifndef SHAPE_H_2YNV5
#define SHAPE_H_2YNV5

#include <assert.h>
#include <initializer_list>
#include <iostream>

#include "zisa/config.hpp"
#include "zisa/meta/all_integral.hpp"

namespace zisa {

template <int n_dims, class Int = int_t>
struct shape_t {
  ANY_DEVICE shape_t() = default;

  ANY_DEVICE shape_t(const shape_t &rhs) = default;

  template <class... Ints,
            class SFINAE
            = typename std::enable_if<all_integral<Ints...>::value, void>::type>
  ANY_DEVICE_INLINE shape_t(Ints... ints) : _raw_data{ints...} {}

  ANY_DEVICE_INLINE
  Int operator[](int dim) const { return _raw_data[dim]; }

  ANY_DEVICE_INLINE
  Int &operator[](int dim) { return _raw_data[dim]; }

  ANY_DEVICE_INLINE
  Int operator()(int dim) const { return _raw_data[dim]; }

  ANY_DEVICE_INLINE
  Int &operator()(int dim) { return _raw_data[dim]; }

  ANY_DEVICE_INLINE
  bool operator==(const shape_t &other) const {
    for (int i = 0; i < n_dims; ++i) {
      if ((*this)(i) != other(i))
        return false;
    }

    return true;
  }

  ANY_DEVICE_INLINE
  bool operator!=(const shape_t &other) const { return !((*this) == other); }

  ANY_DEVICE_INLINE
  static constexpr int size(void) { return n_dims; }

protected:
  Int _raw_data[n_dims];
};

template <int n_dims, class Int>
ANY_DEVICE_INLINE Int product(const shape_t<n_dims, Int> &shape) {
  Int nml = 1;
  for (int k = 0; k < n_dims; ++k) {
    nml *= shape[k];
  }

  return nml;
}

template <int n, class Int>
std::ostream &operator<<(std::ostream &os, const shape_t<n, Int> &shape) {
  os << "[";
  for (int i = 0; i < n; ++i) {
    os << shape(i) << (i != n - 1 ? ", " : "]");
  }

  return os;
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int> operator+(const shape_t<n_dims, Int> &s0,
                                                 int i) {
  auto s = shape_t<n_dims, Int>();

  for (int k = 0; k < s0.size(); ++k) {
    s[k] = s0[k] + i;
  }

  return s;
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int>
operator+(const shape_t<n_dims, Int> &s0, const shape_t<n_dims, Int> &s1) {
  auto s = shape_t<n_dims, Int>();

  for (int k = 0; k < s0.size(); ++k) {
    s[k] = s0[k] + s1[k];
  }

  return s;
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int>
operator+(int i, const shape_t<n_dims, Int> &s0) {
  return s0 + i;
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int>
operator-(const shape_t<n_dims, Int> &s0) {
  auto s = shape_t<n_dims, Int>();

  for (int k = 0; k < s0.size(); ++k) {
    s[k] = -s0[k];
  }

  return s;
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int> operator-(const shape_t<n_dims, Int> &s0,
                                                 int i) {
  return s0 + (-i);
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int>
operator-(int i, const shape_t<n_dims, Int> &s0) {
  return i + (-s0);
}

template <int n_dims, class Int>
ANY_DEVICE_INLINE shape_t<n_dims, Int>
operator-(const shape_t<n_dims, Int> &s0, const shape_t<n_dims, Int> &s1) {
  return s0 + (-s1);
}

} // namespace zisa
#endif /* end of include guard */
