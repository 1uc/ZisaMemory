#ifndef HDF5_FILE_H_QYUAX
#define HDF5_FILE_H_QYUAX

#include <mutex>
#include <stack>
#include <string>
#include <vector>

#include <zisa/io/hdf5.hpp>

namespace zisa {

/// Locks the HDF5 library.
extern std::recursive_mutex hdf5_mutex;

/// Abstraction of the HDF5 data-type macros.
/** More direct approaches using templates exist, however we intend to
 *  pass the data-type into several functions that would in that case need
 *  do be turned into templates aswell.
 *
 *  @see make_hdf5_data_type
 */
class HDF5DataType {
public:
  /// Initialize object.
  /** @param h5_type  The H5 identifier of the data-type.
   *  @param size  Size in bytes of the data type.
   */
  HDF5DataType(const hid_t &h5_type, size_t size);

  virtual ~HDF5DataType();

  /// Return the HDF5 identifier of the data-type.
  hid_t operator()() const;

public:
  size_t size; ///< essentially, `sizeof(T)`

protected:
  hid_t h5_type;
};

namespace {
template <typename T>
inline hid_t get_hdf5_data_type();

template <>
inline hid_t get_hdf5_data_type<double>() {
  return H5T_NATIVE_DOUBLE;
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

/// Representation of the current branch of the opened HDF5 file.
class HDF5File {
public:
  virtual ~HDF5File();

  /// Open HDF5 group.
  void open_group(const std::string &group_name);

  /// Close HDF5 group.
  void close_group();

  /// Switch HDF5 group.
  void switch_group(const std::string &group_name);

  /// Does this group exist in the file?
  bool group_exists(const std::string &group_name) const;

  /// Human readable description of the current hierarchy.
  std::string hierarchy() const;

protected:
  hid_t open_dataset(const std::string &tag) const;
  hid_t get_dataspace(const hid_t &dataset) const;

protected:
  std::stack<hid_t> file;        ///< HDF5 file/group identifiers (branch)
  std::vector<std::string> path; ///< HDF5 path
};

} // namespace zisa
#endif /* end of include guard */
