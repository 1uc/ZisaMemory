#ifndef ARRAY_VIEW_H_NPPW3
#define ARRAY_VIEW_H_NPPW3

#include "zisa/config.hpp"

#include "zisa/memory/array.hpp"
#include "zisa/memory/array_traits.hpp"
#include "zisa/memory/column_major.hpp"
#include "zisa/memory/contiguous_memory.hpp"
#include "zisa/memory/shape.hpp"
#include "zisa/meta/add_const_if.hpp"
#include "zisa/meta/if_t.hpp"

namespace zisa {

template <class T>
struct array_traits<T *> {
  using pointer = typename std::remove_const<T>::type *;
  using const_pointer = typename std::add_const<T>::type *;
  using size_type = std::size_t;
};

template <class T>
ANY_DEVICE_INLINE auto raw_ptr(T *a) -> decltype(a) {
  return a;
}

template <class T, class Indexing>
class array_view_base {
protected:
  using size_type = typename array_traits<T *>::size_type;
  using shape_type = shape_t<indexing_traits<Indexing>::n_dims, size_type>;

public:
  ANY_DEVICE array_view_base(shape_type shape, T *ptr)
      : _shape(shape), _ptr(ptr) {}

  ANY_DEVICE_INLINE T *raw() { return _ptr; }
  ANY_DEVICE_INLINE T const *raw() const { return _ptr; }

  ANY_DEVICE_INLINE T &operator[](size_type i) { return _ptr[i]; }
  ANY_DEVICE_INLINE T operator[](size_type i) const { return _ptr[i]; }

  template <class... Ints>
  ANY_DEVICE_INLINE T &operator()(Ints... ints) {
    auto l = Indexing::linear_index(shape(), std::forward<Ints>(ints)...);
    return (*this)[l];
  }

  template <class... Ints>
  ANY_DEVICE_INLINE const T &operator()(Ints... ints) const {
    auto l = Indexing::linear_index(shape(), std::forward<Ints>(ints)...);
    return (*this)[l];
  }

  ANY_DEVICE_INLINE const shape_type &shape() const { return _shape; }

private:
  shape_type _shape;
  T *_ptr;
};

template <class T, int n_dims, template <int> class Indexing = column_major>
class array_view : public array_view_base<T, Indexing<n_dims>> {
private:
  using super = array_view_base<T, Indexing<n_dims>>;

public:
  using super::array_view_base;

  template <class Array, class Shape>
  ANY_DEVICE_INLINE
  array_view(array_base<T, Indexing<n_dims>, Array, Shape> &other)
      : array_view(zisa::shape(other), zisa::raw_ptr(other)) {}
};

template <class T, int n_dims, template <int> class Indexing = column_major>
class array_const_view : public array_view_base<const T, Indexing<n_dims>> {

private:
  using super = array_view_base<const T, Indexing<n_dims>>;

public:
  using super::array_view_base;

  template <class Array, class Shape>
  ANY_DEVICE_INLINE
  array_const_view(const array_base<T, Indexing<n_dims>, Array, Shape> &other)
      : array_const_view(zisa::shape(other), zisa::raw_ptr(other)) {}

  ANY_DEVICE_INLINE
  array_const_view(const array_view<T, n_dims, Indexing> &other)
      : super(zisa::shape(other), zisa::raw_ptr(other)) {}
};

} // namespace zisa
#endif /* end of include guard */
