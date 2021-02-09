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

template <class T, int n_dims>
void save(HDF5Writer &writer,
          const array<T, n_dims, row_major> &arr,
          const std::string &tag,
          default_dispatch_tag) {

  T const *const data = arr.raw();
  const auto &dims = arr.shape();

  HDF5DataType data_type = make_hdf5_data_type<T>();

  hsize_t h5_dims[n_dims];
  for (int_t i = 0; i < n_dims; ++i) {
    h5_dims[i] = hsize_t(dims(i)); // size of (i, j, k) axes
  }

  writer.write_array(data, data_type, tag, n_dims, h5_dims);
}

template <class T, int n_dims>
void save(HDF5Writer &writer,
          const array<T, n_dims, row_major> &arr,
          const std::string &tag,
          split_array_dispatch_tag) {

  using scalar_type = typename array_save_traits<T>::scalar_type;
  HDF5DataType data_type = make_hdf5_data_type<scalar_type>();

  constexpr int_t rank = n_dims + 1;
  hsize_t h5_dims[rank];
  for (int_t i = 0; i < rank - 1; ++i) {
    h5_dims[i] = hsize_t(arr.shape(i));
  }
  h5_dims[rank - 1] = T::size();

  writer.write_array(arr.raw(), data_type, tag, rank, h5_dims);
}

template <int n_dims>
void save(HDF5Writer &writer,
          const array<bool, n_dims, row_major> &arr,
          const std::string &tag,
          bool_dispatch_tag) {

  using scalar_type = typename array_save_traits<bool>::scalar_type;
  auto int_arr = array<scalar_type, n_dims>(arr.shape());
  std::copy(arr.cbegin(), arr.cend(), int_arr.begin());

  save(writer, int_arr, tag);
}

template <class T, int n_dims, template <int> class Indexing>
void save(HDF5Writer &writer,
          const array<T, n_dims, Indexing> &arr,
          const std::string &tag) {

  save(writer, arr, tag, typename array_save_traits<T>::dispatch_tag{});
}

template <class T, int n_dims>
void load_impl(HDF5Reader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               split_array_dispatch_tag) {

  using scalar_type = typename array_save_traits<T>::scalar_type;
  auto h5_datatype = make_hdf5_data_type<scalar_type>();
  reader.read_array(arr.raw(), h5_datatype, tag);
}

template <class T, int n_dims>
void load_impl(HDF5Reader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               default_dispatch_tag) {

  auto h5_datatype = make_hdf5_data_type<T>();
  reader.read_array(arr.raw(), h5_datatype, tag);
}

template <class T, int n_dims>
void load_impl(HDF5Reader &reader,
               array<T, n_dims, row_major> &arr,
               const std::string &tag,
               bool_dispatch_tag) {

  using scalar_type = typename array_save_traits<T>::scalar_type;
  auto h5_datatype = make_hdf5_data_type<scalar_type>();

  auto int_arr = array<scalar_type, n_dims, row_major>(arr.shape());
  reader.read_array(int_arr.raw(), h5_datatype, tag);

  std::copy(int_arr.cbegin(), int_arr.cend(), arr.begin());
}

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, row_major>
array<T, n_dims, Indexing>::load(HDF5Reader &reader, const std::string &tag) {

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
