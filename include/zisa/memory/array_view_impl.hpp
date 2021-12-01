#ifndef ARRAY_VIEW_IMPL_HPP
#define ARRAY_VIEW_IMPL_HPP

namespace zisa {

template <class T>
auto raw_ptr(T *a) -> decltype(a) {
  return a;
}

template <class T, class Indexing>
array_view_base<T, Indexing>::array_view_base(array_view_base::shape_type shape,
                                              T *ptr,
                                              device_type mem_location)
    : _shape(shape), _ptr(ptr), _mem_location(mem_location) {}

template <class T, class Indexing>
typename array_view_base<T, Indexing>::size_type
array_view_base<T, Indexing>::size() const {
  return product(_shape);
}

template <class T, class Indexing>
const typename array_view_base<T, Indexing>::shape_type &
array_view_base<T, Indexing>::shape() const {
  return _shape;
}

template <class T, class Indexing>
typename array_view_base<T, Indexing>::size_type
array_view_base<T, Indexing>::shape(array_view_base::size_type i) const {
  return _shape(i);
}

template <class T, class Indexing>
device_type memory_location(const array_view_base<T, Indexing> &view) {
  return view.memory_location();
}

template <class T, int n_dims, template <int> class Indexing>
array_view<T, n_dims, Indexing>::array_view(const shape_t<n_dims> &shape,
                                            T *ptr,
                                            device_type mem_location)
    : super(shape, ptr, mem_location) {}

template <class T, int n_dims, template <int> class Indexing>
template <class Array, class Shape>
array_view<T, n_dims, Indexing>::array_view(
    array_base<T, Indexing<n_dims>, Array, Shape> &other)
    : array_view(zisa::shape(other),
                 zisa::raw_ptr(other),
                 zisa::memory_location(other)) {}

template <class T, int n_dims, template <int> class Indexing>
array_view<T, n_dims, Indexing>::array_view(std::vector<T> &v)
    : array_view(shape_t<1>{v.size()}, v.data(), device_type::cpu) {}

template <class T, int n_dims, template <int> class Indexing>
T *array_view<T, n_dims, Indexing>::raw() const {
  return this->_ptr;
}

template <class T, int n_dims, template <int> class Indexing>
T &array_view<T, n_dims, Indexing>::operator[](array_view::size_type i) const {
  return this->_ptr[i];
}

template <class T, int n_dims, template <int> class Indexing>
template <class... Ints>
T &array_view<T, n_dims, Indexing>::operator()(Ints... ints) const {
  auto l = Indexing<n_dims>::linear_index(this->shape(),
                                          integer_cast<size_type>(ints)...);
  return (*this)[l];
}

template <class T, int n_dims, template <int> class Indexing>
T *array_view<T, n_dims, Indexing>::begin() const {
  return this->_ptr;
}

template <class T, int n_dims, template <int> class Indexing>
T *array_view<T, n_dims, Indexing>::end() const {
  return this->_ptr + this->size();
}

template <class T, int n_dims, template <int> class Indexing>
T const *array_view<T, n_dims, Indexing>::cbegin() const {
  return this->_ptr;
}

template <class T, int n_dims, template <int> class Indexing>
T const *array_view<T, n_dims, Indexing>::cend() const {
  return this->_ptr + this->size();
}

template <class T, int n_dims, template <int> class Indexing>
void array_view<T, n_dims, Indexing>::copy_data(
    const array_view<T, n_dims, Indexing> &other) const {
  copy_data(array_const_view<T, n_dims, Indexing>(other));
}

template <class T, int n_dims, template <int> class Indexing>
void array_view<T, n_dims, Indexing>::copy_data(
    const array_const_view<T, n_dims, Indexing> &other) const {
  assert((*this).shape() == other.shape());
  LOG_ERR_IF(this->memory_location() == device_type::cuda, "Implement this.");

  if (other.raw() != (*this).raw()) {
    std::copy(other.begin(), other.end(), (*this).begin());
  }
}

}

#endif // ARRAY_VIEW_IMPL_HPP
