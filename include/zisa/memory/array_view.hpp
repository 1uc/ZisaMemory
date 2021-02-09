#ifndef ARRAY_VIEW_H_NPPW3
#define ARRAY_VIEW_H_NPPW3

#include <zisa/config.hpp>

#include <zisa/memory/array.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/array_view_fwd.hpp>
#include <zisa/memory/column_major.hpp>
#include <zisa/memory/contiguous_memory.hpp>
#include <zisa/memory/shape.hpp>
#include <zisa/meta/add_const_if.hpp>
#include <zisa/meta/if_t.hpp>

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
  ANY_DEVICE array_view_base(shape_type shape, T *ptr, device_type mem_location)
      : _shape(shape), _ptr(ptr), _mem_location(mem_location) {}

  ANY_DEVICE_INLINE size_type size() const { return product(_shape); }

  ANY_DEVICE_INLINE const shape_type &shape() const { return _shape; }
  ANY_DEVICE_INLINE size_type shape(size_type i) const { return _shape(i); }

  ANY_DEVICE_INLINE device_type memory_location() const {
    return _mem_location;
  }

protected:
  shape_type _shape;
  T *_ptr;
  device_type _mem_location;
};

template <class T, class Indexing>
device_type memory_location(const array_view_base<T, Indexing> &view) {
  return view.memory_location();
}

template <class T, int n_dims, template <int> class Indexing>
class array_view : public array_view_base<T, Indexing<n_dims>> {
private:
  using super = array_view_base<T, Indexing<n_dims>>;
  using size_type = typename super::size_type;

public:
  ANY_DEVICE_INLINE
  array_view(const shape_t<n_dims> &shape,
             T *ptr,
             device_type mem_location = device_type::unknown)
      : super(shape, ptr, mem_location) {}

  template <class Array, class Shape>
  ANY_DEVICE_INLINE
  array_view(array_base<T, Indexing<n_dims>, Array, Shape> &other)
      : array_view(zisa::shape(other),
                   zisa::raw_ptr(other),
                   zisa::memory_location(other)) {}

  array_view(std::vector<T> &v)
      : array_view(shape_t<1>{v.size()}, v.data(), device_type::cpu) {}

  ANY_DEVICE_INLINE T *raw() const { return this->_ptr; }
  ANY_DEVICE_INLINE T &operator[](size_type i) const { return this->_ptr[i]; }

  template <class... Ints>
  ANY_DEVICE_INLINE T &operator()(Ints... ints) const {
    auto l = Indexing<n_dims>::linear_index(this->shape(),
                                            integer_cast<size_type>(ints)...);
    return (*this)[l];
  }

  ANY_DEVICE_INLINE T *begin() const { return this->_ptr; }
  ANY_DEVICE_INLINE T *end() const { return this->_ptr + this->size(); }

  void copy_data(const array_view<T, n_dims, Indexing> &other) const {
    copy_data(array_const_view(other));
  }

  void copy_data(const array_const_view<T, n_dims, Indexing> &other) const {
    assert((*this).shape() == other.shape());
    LOG_ERR_IF(this->memory_location() != device_type::cpu, "Implement this.");

    if (other.raw() != (*this).raw()) {
      std::copy(other.begin(), other.end(), (*this).begin());
    }
  }
};

template <class T>
array_view(std::vector<T> &v) -> array_view<T, 1, row_major>;

template <class T, int n_dims, template <int> class Indexing>
class array_const_view : public array_view_base<const T, Indexing<n_dims>> {

private:
  using super = array_view_base<const T, Indexing<n_dims>>;
  using size_type = typename super::size_type;

public:
  ANY_DEVICE_INLINE
  array_const_view(const shape_t<n_dims> &shape,
                   T const *ptr,
                   device_type mem_location = device_type::unknown)
      : super(shape, ptr, mem_location) {}

  template <class Array, class Shape>
  ANY_DEVICE_INLINE
  array_const_view(const array_base<T, Indexing<n_dims>, Array, Shape> &other)
      : array_const_view(zisa::shape(other),
                         zisa::raw_ptr(other),
                         zisa::memory_location(other)) {}

  ANY_DEVICE_INLINE
  array_const_view(const array_view<T, n_dims, Indexing> &other)
      : super(other.shape(), other.raw(), zisa::memory_location(other)) {}

  array_const_view(const std::vector<T> &v)
      : array_const_view(shape_t<1>{v.size()}, v.data(), device_type::cpu) {}

  ANY_DEVICE_INLINE const T *raw() const { return this->_ptr; }
  ANY_DEVICE_INLINE const T &operator[](size_type i) const {
    return this->_ptr[i];
  }

  template <class... Ints>
  ANY_DEVICE_INLINE const T &operator()(Ints... ints) const {
    auto l = Indexing<n_dims>::linear_index(this->shape(),
                                            integer_cast<size_type>(ints)...);
    return (*this)[l];
  }

  ANY_DEVICE_INLINE const T *begin() const { return this->_ptr; }
  ANY_DEVICE_INLINE const T *end() const { return this->_ptr + this->size(); }
};

template <class T>
array_const_view(const std::vector<T> &) -> array_const_view<T, 1, row_major>;

namespace detail {
template <class T, int n_dims>
array_const_view<T, n_dims, row_major>
slice(const array_const_view<T, n_dims, row_major> &arr, int_t i0, int_t i1) {
  auto sub_shape = arr.shape();
  sub_shape[0] = i1 - i0;

  int_t offset = i0 * (product(sub_shape) / (i1 - i0));
  auto ptr = arr.raw() + offset;
  return {sub_shape, ptr};
}
}

template <class T, int n_dims>
array_view<T, n_dims, row_major>
slice(const array_view<T, n_dims, row_major> &arr, int_t i0, int_t i1) {
  auto const_view = detail::slice(array_const_view(arr), i0, i1);
  return {const_view.shape(), const_cast<T *>(const_view.raw())};
}

template <class T, int n_dims>
array_const_view<T, n_dims, row_major> const_slice(
    const array_const_view<T, n_dims, row_major> &arr, int_t i0, int_t i1) {
  return detail::slice(arr, i0, i1);
}

template <class T, int n_dims, template <int> class Indexing>
ANY_DEVICE_INLINE auto raw_ptr(const array_view<T, n_dims, Indexing> &a)
    -> decltype(a.raw()) {
  return a.raw();
}

template <class T, int n_dims, template <int> class Indexing>
ANY_DEVICE_INLINE auto raw_ptr(const array_const_view<T, n_dims, Indexing> &a)
    -> decltype(a.raw()) {
  return a.raw();
}

} // namespace zisa
#endif /* end of include guard */
