#ifndef ARRAY_BASE_H_QTIBA
#define ARRAY_BASE_H_QTIBA

#include <iostream>
#include <utility>

#include <zisa/io/hdf5.hpp>
#include <zisa/io/hdf5_writer.hpp>
#include <zisa/memory/array_base_fwd.hpp>
#include <zisa/memory/array_traits.hpp>

namespace zisa {

template <class Indexing>
struct indexing_traits;

template <class Array>
struct array_traits;

template <class T, class Indexing, class Array, class Shape>
class array_base {
public:
  using shape_type = Shape;
  using size_type = typename array_traits<Array>::size_type;
  using pointer = typename array_traits<Array>::pointer;
  using const_pointer = typename array_traits<Array>::const_pointer;

public:
  static constexpr int n_dims = indexing_traits<Indexing>::n_dims;

public:
  array_base() = default;

  array_base(Shape shape, Array array)
      : _shape(std::move(shape)), _array(std::move(array)) {}

  array_base(const array_base &other) = default;
  array_base(array_base &&other) noexcept = default;

  ~array_base() = default;

  array_base &operator=(const array_base &other) = default;
  array_base &operator=(array_base &&other) noexcept = default;

  ANY_DEVICE_INLINE T &operator[](size_type i) { return _array[i]; }
  ANY_DEVICE_INLINE const T &operator[](size_type i) const { return _array[i]; }

  template <class... Ints>
  ANY_DEVICE_INLINE T &operator()(Ints... ints) {
    auto l = Indexing::linear_index(shape(), ints...);
    return (*this)[l];
  }

  template <class... Ints>
  ANY_DEVICE_INLINE const T &operator()(Ints... ints) const {
    auto l = Indexing::linear_index(shape(), ints...);
    return (*this)[l];
  }

  ANY_DEVICE_INLINE const shape_type &shape() const { return _shape; }
  ANY_DEVICE_INLINE size_type shape(size_type k) const { return _shape[k]; }
  ANY_DEVICE_INLINE size_type size() const { return product(_shape); }

  ANY_DEVICE_INLINE pointer raw() { return raw_ptr(_array); }
  ANY_DEVICE_INLINE const_pointer raw() const { return raw_ptr(_array); }

  ANY_DEVICE_INLINE pointer begin() { return _array.begin(); }
  ANY_DEVICE_INLINE const_pointer begin() const { return _array.begin(); }
  ANY_DEVICE_INLINE const_pointer cbegin() const { return _array.cbegin(); }

  ANY_DEVICE_INLINE pointer end() { return _array.end(); }
  ANY_DEVICE_INLINE const_pointer end() const { return _array.end(); }
  ANY_DEVICE_INLINE const_pointer cend() const { return _array.cend(); }

private:
  Shape _shape;
  Array _array;
};

template <class T, class Indexing, class Array, class Shape>
bool operator==(const array_base<T, Indexing, Array, Shape> &lhs,
                const array_base<T, Indexing, Array, Shape> &rhs) {

  if (shape(lhs) != shape(rhs)) {
    return false;
  }

  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <class T, class Indexing, class Array, class Shape>
bool operator!=(const array_base<T, Indexing, Array, Shape> &lhs,
                const array_base<T, Indexing, Array, Shape> &rhs) {

  return !(lhs == rhs);
}

template <class T, class Indexing, class Array, class Shape>
ANY_DEVICE_INLINE auto raw_ptr(array_base<T, Indexing, Array, Shape> &a)
    -> decltype(a.raw()) {
  return a.raw();
}

template <class T, class Indexing, class Array, class Shape>
ANY_DEVICE_INLINE auto raw_ptr(const array_base<T, Indexing, Array, Shape> &a)
    -> decltype(a.raw()) {
  return a.raw();
}

template <class T, class Indexing, class Array, class Shape>
ANY_DEVICE_INLINE auto shape(array_base<T, Indexing, Array, Shape> &a)
    -> decltype(a.shape()) {
  return a.shape();
}

template <class T, class Indexing, class Array, class Shape>
ANY_DEVICE_INLINE auto shape(const array_base<T, Indexing, Array, Shape> &a)
    -> decltype(a.shape()) {
  return a.shape();
}

} // namespace zisa

#endif /* end of include guard */
