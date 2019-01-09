/* Wrapper for HDF5 library calls.
 *
 */

#ifndef HDF5_H_XC8WA
#define HDF5_H_XC8WA

#include "zisa/config.hpp"
#include <hdf5.h>

namespace zisa {

#define HDF5_SAFE_CALL(h5_function, message)                                   \
  auto ret = h5_function(std::forward<Args>(args)...);                         \
  LOG_ERR_IF(ret < 0, string_format(message, ret));                            \
  return ret;

namespace H5D {

template <class... Args>
hid_t get_space(Args &&... args) {
  HDF5_SAFE_CALL(H5Dget_space, "Failed to get HDF5 dataset. [%d].");
}

template <class... Args>
hid_t read(Args &&... args) {
  HDF5_SAFE_CALL(H5Dread, "Failed to read HDF5 dataset. [%d].");
}

template <class... Args>
hid_t write(Args &&... args) {
  HDF5_SAFE_CALL(H5Dwrite, "Failed to write HDF5 dataset. [%d].");
}

template <class... Args>
hid_t create(hid_t h5_file, char const *const tag, Args &&... args) {
  auto ret = H5Dcreate(h5_file, tag, std::forward<Args>(args)...);

  LOG_ERR_IF(
      ret < 0,
      string_format("Failed to create HDF5 dataset '%s'. [%d]", tag, ret));

  return ret;
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Dclose, "Failed to close HDF5 dataset. [%d].");
}

} // namespace H5D

namespace H5F {
template <class... Args>
hid_t open(char const *const filename, Args &&... args) {
  auto h5_file = H5Fopen(filename, std::forward<Args>(args)...);
  LOG_ERR_IF(
      h5_file < 0,
      string_format("Failed to open file '%s'. [%d]", filename, h5_file));
  return h5_file;
}

template <class... Args>
hid_t create(char const *const filename, Args &&... args) {
  auto h5_file = H5Fcreate(filename, std::forward<Args>(args)...);
  LOG_ERR_IF(
      h5_file < 0,
      string_format("Failed to open file '%s', [%d]", filename, h5_file));
  return h5_file;
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Fclose, "Failed to close HDF5 file. [%d].");
}
} // namespace H5F

namespace H5G {

template <class... Args>
hid_t open(hid_t h5_file, char const *const group_name, Args &&... args) {
  auto ret = H5Gopen(h5_file, group_name, std::forward<Args>(args)...);
  LOG_ERR_IF(ret < 0,
             string_format("Failed to open group '%s'. [%d]", group_name, ret));

  return ret;
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Gclose, "Failed to close HDF5 group. [%d].");
}

template <class... Args>
hid_t create(Args &&... args) {
  HDF5_SAFE_CALL(H5Gcreate, "Failed to create HDF5 group. [%d].");
}

} // namespace H5G

namespace H5S {

template <class... Args>
hid_t create(Args &&... args) {
  HDF5_SAFE_CALL(H5Screate, "Failed to create HDF5 dataspace. [%d].");
}

template <class... Args>
hid_t select_hyperslab(Args &&... args) {
  HDF5_SAFE_CALL(H5Sselect_hyperslab, "Failed to select HDF5 hyperslab. [%d].");
}

template <class... Args>
hid_t create_simple(Args &&... args) {
  HDF5_SAFE_CALL(H5Screate_simple, "Failed to create HDF5 dataspace. [%d].");
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Sclose, "Failed to close HDF5 dataspace. [%d].");
}

template <class... Args>
int get_simple_extent_dims(Args &&... args) {
  HDF5_SAFE_CALL(H5Sget_simple_extent_dims,
                 "Failed to get dataspace dims. [%d].");
}

template <class... Args>
int get_simple_extent_ndims(Args &&... args) {
  HDF5_SAFE_CALL(H5Sget_simple_extent_ndims,
                 "Failed to get rank of dataspace. [%d].");
}

} // namespace H5S

namespace H5T {

template <class... Args>
hid_t copy(Args &&... args) {
  HDF5_SAFE_CALL(H5Tcopy, "Failed to copy HDF5 datatype. [%d].");
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Tclose, "Failed to close HDF5 datatype. [%d].");
}
} // namespace H5T

namespace H5P {

template <class... Args>
hid_t create(Args &&... args) {
  HDF5_SAFE_CALL(H5Pcreate, "Failed to create HDF5 property list. [%d].");
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Pclose, "Failed to close HDF5 property list. [%d].");
}

template <class... Args>
hid_t set_dxpl_mpio(Args &&... args) {
  HDF5_SAFE_CALL(H5Pset_dxpl_mpio, "Failed to close HDF5 property list. [%d].");
}

template <class... Args>
hid_t set_fapl_mpio(Args &&... args) {
  HDF5_SAFE_CALL(H5Pset_fapl_mpio, "Failed to close HDF5 property list. [%d].");
}

template <class... Args>
hid_t set_chunk(Args &&... args) {
  HDF5_SAFE_CALL(H5Pset_chunk, "Failed to set chunk. [%d].");
}

template <class... Args>
hid_t set_deflate(Args &&... args) {
  HDF5_SAFE_CALL(H5Pset_deflate, "Failed to set deflate. [%d].");
}

} // namespace H5P

} // namespace zisa
#endif /* end of include guard */
