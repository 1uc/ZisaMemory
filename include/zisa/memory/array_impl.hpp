/*
 *
 */

#ifndef ARRAY_IMPL_H_1WC9M
#define ARRAY_IMPL_H_1WC9M

#include "zisa/memory/array_base.hpp"
#include "zisa/memory/array_decl.hpp"

namespace zisa {

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(const shape_type &shape, device_type device)
    : super(shape, contiguous_memory<T>(product(shape), device)) {}

template <class T, int n_dims>
void save(const HDF5Writer &writer,
          const array<T, n_dims> &arr,
          const std::string &tag) {

  T const *const data = arr.raw();
  const auto &dims = arr.shape();

  HDF5DataType data_type = make_hdf5_data_type<T>();

  hsize_t h5_dims[n_dims];
  for (int i = 0; i < n_dims; ++i) {
    h5_dims[i] = hsize_t(dims(i)); // size of (i, j, k) axes
  }

  writer.write_array((T *)data, data_type, tag, n_dims, h5_dims);
}

template <class T, int n_dims>
void load(const HDF5Reader &reader,
          array<T, n_dims> &arr,
          const std::string &tag) {

  auto dims = reader.dims(tag);

  assert(dims.size() == arr.n_dims);
  for (int i = 0; i < n_dims; ++i) {
    assert(dims[i] == arr.shape(i));
  }

  auto h5_datatype = make_hdf5_data_type<T>();
  reader.read_array(arr.raw(), h5_datatype, tag);
}
} // namespace zisa

#endif /* end of include guard */
