// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

/* Array traits.
 *
 */

#ifndef ARRAY_TRAITS_H_8G39N
#define ARRAY_TRAITS_H_8G39N

#include <type_traits>
#include <zisa/config.hpp>
#include <zisa/memory/device_type.hpp>

namespace zisa {

namespace detail {
template <class A>
device_type memory_location_(const A &) {
  return device_type::unknown;
}

template <class A>
device_type memory_location_(
    const typename std::enable_if<
        std::is_same<decltype(std::declval<A>().device()), device_type>::value,
        A>::type &a) {
  return a.device();
}
}

template <class Array>
struct array_traits {
  using pointer = typename Array::pointer;
  using const_pointer = typename Array::const_pointer;
  using size_type = typename Array::size_type;

  static inline device_type device(const Array &array) {
    return detail::memory_location_<Array>(array);
  }
};

struct default_dispatch_tag {};
struct split_array_dispatch_tag {};
struct bool_dispatch_tag {};

template <class T>
struct array_save_traits {
  using dispatch_tag = default_dispatch_tag;
  using scalar_type = T;
};

template <>
struct array_save_traits<bool> {
  using dispatch_tag = bool_dispatch_tag;
  using scalar_type = char;
};

} // namespace zisa

#endif /* end of include guard */
