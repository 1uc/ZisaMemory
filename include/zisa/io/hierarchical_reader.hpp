// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef HIERARCHICAL_READER_HPP_XEIQO
#define HIERARCHICAL_READER_HPP_XEIQO

#include <zisa/io/hierarchical_file.hpp>

namespace zisa {

/// Read data from a hierarchical file.
class HierarchicalReader : public virtual HierarchicalFile {
public:
  /// Dimensions of the array named `tag`.
  std::vector<std::size_t> dims(const std::string &tag) const {
    return do_dims(tag);
  }

  /// Low-level function to read an array from disc into memory.
  /** Allocating (and managing) the memory, is job of the caller. One can
   *  use `dims` to query the size of the array.
   *
   *  @param[out] data       allocated memory of sufficient size.
   *  @param[in]  data_type  Erased data type of the data.
   *  @param[in]  tag        Name of the array in the file.
   */
  void read_array(void *data,
                  const DataType &data_type,
                  const std::string &tag) const;

  /// Read a scalar from the hierarchical file.
  /** @param tag Name of the scalar in the file.
   */
  template <class T>
  T read_scalar(const std::string &tag) const;

  void read_scalar(void *data,
                   const DataType &data_type,
                   const std::string &tag) const;

  /// Read a string from the hierarchical file.
  /** @param tag Name of the scalar in the file.
   */
  std::string read_string(const std::string &tag) const;

protected:
  virtual std::vector<std::size_t> do_dims(const std::string &tag) const = 0;

  virtual void do_read_array(void *data,
                             const DataType &data_type,
                             const std::string &tag) const = 0;

  virtual void do_read_scalar(void *data,
                              const DataType &data_type,
                              const std::string &tag) const = 0;

  virtual std::string do_read_string(const std::string &tag) const = 0;
};

template <class T>
T HierarchicalReader::read_scalar(const std::string &tag) const {
  T scalar;
  (*this).read_scalar(&scalar, erase_data_type<T>(), tag);
  return scalar;
}

}
#endif // HIERARCHICAL_READER_HPP