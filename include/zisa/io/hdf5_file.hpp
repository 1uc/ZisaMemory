// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef HDF5_FILE_H_QYUAX
#define HDF5_FILE_H_QYUAX

#include <stack>
#include <string>
#include <vector>

#include <zisa/io/hdf5.hpp>
#include <zisa/io/hdf5_resource.hpp>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/utils/integer_cast.hpp>

namespace zisa {

/// Abstraction of the HDF5 data-type macros.
/** More direct approaches using templates exist, however we intend to
 *  pass the data-type into several functions that would in that case need
 *  do be turned into templates as well.
 *
 *  @see make_hdf5_data_type
 */
class HDF5DataType : public HDF5Resource {
private:
  using super = HDF5Resource;

public:
  /// Initialize object.
  /** @param h5_type  The H5 identifier of the data-type.
   *  @param size  Size in bytes of the data type.
   */
  HDF5DataType(const hid_t &h5_type, size_t size);

  [[deprecated("Use `operator*`.")]] hid_t operator()() const;

public:
  size_t size; ///< essentially, `sizeof(T)`
};

namespace {
template <typename T>
inline hid_t get_hdf5_data_type();

template <>
inline hid_t get_hdf5_data_type<double>() {
  return H5T_NATIVE_DOUBLE;
}

template <>
inline hid_t get_hdf5_data_type<float>() {
  return H5T_NATIVE_FLOAT;
}

template <>
inline hid_t get_hdf5_data_type<int>() {
  return H5T_NATIVE_INT;
}

template <>
inline hid_t get_hdf5_data_type<unsigned long>() {
  return H5T_NATIVE_ULONG;
}

template <>
inline hid_t get_hdf5_data_type<char>() {
  return H5T_NATIVE_CHAR;
}
} // namespace

HDF5DataType make_hdf5_data_type(const hid_t &hdf5_data_type, size_t size);

/// Return the HDF5 native data-type identifier for `T`.
template <typename T>
HDF5DataType make_hdf5_data_type() {
  return make_hdf5_data_type(get_hdf5_data_type<T>(), sizeof(T));
}

/// Return the HDF5 native data-type an erased data type identifier.
HDF5DataType make_hdf5_data_type(const ErasedDataType &data_type);

std::vector<hsize_t> make_hdf5_dims(std::size_t const *const dims, int rank);

/// Representation of the current branch of the opened HDF5 file.
class HDF5File : public virtual HierarchicalFile {
public:
  virtual ~HDF5File() override;

protected:
  void do_open_group(const std::string &group_name) override;
  void do_close_group() override;
  void do_switch_group(const std::string &group_name) override;
  bool do_group_exists(const std::string &group_name) const override;
  std::string do_hierarchy() const override;
  void do_unlink(const std::string &tag) override {
    zisa::H5L::unlink(file.top(), tag.c_str(), H5P_DEFAULT);
  }

protected:
  HDF5Dataset open_dataset(const std::string &tag) const;
  HDF5Dataspace get_dataspace(const hid_t &dataset) const;

protected:
  std::stack<hid_t> file;        ///< HDF5 file/group identifiers (branch)
  std::vector<std::string> path; ///< HDF5 path
};

} // namespace zisa
#endif /* end of include guard */
