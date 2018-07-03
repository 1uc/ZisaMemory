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
hid_t create(Args &&... args) {
  HDF5_SAFE_CALL(H5Dcreate, "Failed to create HDF5 dataset. [%d].");
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Dclose, "Failed to close HDF5 dataset. [%d].");
}

} // namespace H5D

namespace H5F {
template <class... Args>
hid_t open(Args &&... args) {
  HDF5_SAFE_CALL(H5Fopen, "Failed to open HDF5 file. [%d].");
}

template <class... Args>
hid_t create(char const *const filename, Args &&... args) {
  auto ret = H5Fcreate(filename, std::forward<Args>(args)...);
  LOG_ERR_IF(ret < 0, string_format("Failed to open file. [%s]", filename));
  return ret;
}

template <class... Args>
hid_t close(Args &&... args) {
  HDF5_SAFE_CALL(H5Fclose, "Failed to close HDF5 file. [%d].");
}
} // namespace H5F

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

} // namespace H5S

namespace H5T {

template <class... Args>
hid_t copy(Args &&... args) {
  HDF5_SAFE_CALL(H5Tcopy, "Failed to copy HDF5 datatype. [%d].");
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

} // namespace H5P

} // namespace tyr
#endif /* end of include guard */
