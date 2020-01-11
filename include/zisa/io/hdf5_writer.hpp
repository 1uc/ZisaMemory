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
#include <zisa/io/hdf5_file.hpp>

namespace zisa {

/// Interface for writing data to an HDF5 file.
class HDF5Writer : public HDF5File {
protected:
  // Aquires a lock on the HDF5 library.
  HDF5Writer() = default;

public:
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

  /// Dimensions of the array named `tag`.
  virtual std::vector<hsize_t> dims(const std::string &tag) const = 0;

  /// Low-level function to read an HDF5 array into memory.
  /** This will allocate and initialize an array.
   *
   *  @param[out] data  allocated memory of sufficient size.
   *  @param[in] data_type  HDF5 data type of the data.
   *  @param[in] tag  Name of the array in the HDF5 file.
   * dimension.
   *
   */
  virtual void read_array(void *data,
                          const HDF5DataType &data_type,
                          const std::string &tag) const = 0;

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
