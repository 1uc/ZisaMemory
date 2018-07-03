/* File I/O with HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2014-09-03
 */
#ifndef HDF5_WRITER_H_2GZ8LVPI
#define HDF5_WRITER_H_2GZ8LVPI
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <stack>
#include <string>
#include <vector>

#include <zisa/config.hpp>
#include <zisa/io/hdf5.hpp>

namespace zisa {
/// Sequentially numbered file names.
/** Provides consistent names for snapshots, the grid and the steady-state
 *  files. The snapshot names are a incrementally numbered sequence.
 *
 *  Example:
 *
 *      auto fng = FileNameGenerator("data/foo", "-%04d", ".h5");
 *
 *      std::cout << fng.grid_filename << "\n";
 *      std::cout << fng.steady_state_filename << "\n";
 *      for (int i = 0; i < 293; ++i) {
 *        std::cout << fng() << "\n";
 *      }
 *  will produce:
 *
 *      data/foo_grid.h5
 *      data/foo_steady-state.h5
 *      data/foo-0000.h5
 *        ...
 *      data/foo-0292.h5
 *
 *  @note The pattern should be a printf-style format for one integer.
 */
class FileNameGenerator {
public:
  FileNameGenerator(const std::string &stem,
                    const std::string &pattern,
                    const std::string &suffix);

  /// Generate the next numbered file name.
  std::string next_name(void);

  /// Generate numbers starting from `k`.
  void advance_to(int k);

  const std::string filename_stem;         ///< First part of all filenames.
  const std::string steady_state_filename; ///< Name of the steady-state.
  const std::string reference_filename;    ///< Name of the reference solution.
  const std::string grid_filename;         ///< Name of the grid.
  const std::string xdmf_grid_filename;    ///< Name of the grid.

private:
  std::string pattern;
  int count;
};

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
  hid_t operator()(void) const;

public:
  size_t size; ///< essentially, `sizeof(T)`

protected:
  hid_t h5_type;
};

namespace {
template <typename T>
inline hid_t get_hdf5_data_type(void);

template <>
inline hid_t get_hdf5_data_type<double>(void) {
  return H5T_NATIVE_DOUBLE;
}

template <>
inline hid_t get_hdf5_data_type<int>(void) {
  return H5T_NATIVE_INT;
}

template <>
inline hid_t get_hdf5_data_type<char>(void) {
  return H5T_NATIVE_CHAR;
}
} // namespace

HDF5DataType make_hdf5_data_type(const hid_t &hdf5_data_type, size_t size);

/// Return the HDF5 native data-type identifier for `T`.
template <typename T>
HDF5DataType make_hdf5_data_type(void) {
  return make_hdf5_data_type(get_hdf5_data_type<T>(), sizeof(T));
}

/// Representation of the current branch of the opened HDF5 file.
class HDF5File {
public:
  HDF5File() = default;
  virtual ~HDF5File();

  /// Open HDF5 group.
  void open_group(const std::string &group_name);

  /// Close HDF5 group.
  void close_group(void);

  /// Switch HDF5 group.
  void switch_group(const std::string &group_name);

  /// Does this group exist in the file?
  bool group_exists(const std::string &group_name) const;

  /// Human readable description of the current hierarchy.
  std::string hierarchy(void) const;

protected:
  hid_t open_dataset(const std::string &tag) const;
  hid_t get_dataspace(const hid_t &dataset) const;
  int get_dims(const hid_t &dataspace, hsize_t *const dims, int rank) const;

protected:
  std::stack<hid_t> file;        ///< HDF5 file/group identifiers (branch)
  std::vector<std::string> path; ///< HDF5 path
};

/// Interface for writing data to an HDF5 file.
class HDF5Writer : public HDF5File {
public:
  HDF5Writer() = default;
  virtual ~HDF5Writer() = default;

  /// Write a multi-dimensional array to an HDF5 file.
  /** @param data  Raw pointer to the data.
   *  @param data_type  HDF5 data type identifier.
   *  @param tag  Name of the data field in the HDF5 file.
   *  @param rank  Number of dimension of the array.
   *  @param dims  Size (in byte) of each dimension.
   */
  virtual void write_array(void const *const data,
                           const HDF5DataType &data_type,
                           const std::string &tag,
                           const int rank,
                           hsize_t const *const dims) const = 0;

  /// Write a scalar to an HDF5 file.
  /**  @param data  The scalar to be written to file.
   *   @param tag  Name of the scalar in the HDF5 file.
   */
  template <typename T>
  void write_scalar(const T &scalar, const std::string &tag) const {
    auto data_type = make_hdf5_data_type<T>();
    write_scalar((void *)&scalar, data_type, tag);
  }

  /// Write a scalar to an HDF5 file.
  /** @param data  Address of the scalar.
   *  @param data_type  HDF5 data type identifier.
   *  @param tag  Name of the scalar in the HDF5 file.
   */
  virtual void write_scalar(void const *const addr,
                            const HDF5DataType &data_type,
                            const std::string &tag) const = 0;

  /// Write a C++ string to the HDF5 file.
  /** @param data  String to write to the file.
   *  @param tag  Name of the string in the HDF5 file.
   */
  virtual void write_string(const std::string &data,
                            const std::string &tag) const = 0;
};

/// Read data from HDF5 file.
class HDF5Reader : public HDF5File {
private:
  using super = HDF5File;

public:
  virtual ~HDF5Reader() = default;

  /// Low-level function to read an HDF5 array into memory.
  /** This will allocate and initialize an array.
   *
   *  @param[in] data_type  HDF5 data type of the data.
   *  @param[in] tag  Name of the array in the HDF5 file.
   *  @param[in] rank  Number of dimensions of the array.
   *  @param[out] dims  The shape of the array, i.e. the extent of each
   * dimension.
   *
   *  @note Ownership of the memory is passed to the caller.
   */
  virtual void *read_array(const HDF5DataType &data_type,
                           const std::string &tag,
                           int rank,
                           hsize_t *const dims) const = 0;

  /// Read a scalar from the HDF5 file.
  /** @param tag Name of the scalar in the HDF5 file.
   */
  template <class T>
  T read_scalar(const std::string &tag) const {
    T scalar;
    (*this).read_scalar(&scalar, make_hdf5_data_type<T>(), tag);
    return scalar;
  }

  virtual void read_scalar(void *const data,
                           const HDF5DataType &data_type,
                           const std::string &tag) const = 0;

  /// Read a string from the HDF5 file.
  /** @param tag Name of the scalar in the HDF5 file.
   */
  virtual std::string read_string(const std::string &tag) const = 0;
};

} // namespace zisa
#endif /* end of include guard: HDF5_WRITER_H_2GZ8LVPI */
