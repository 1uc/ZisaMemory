#ifndef HIERARCHICAL_WRITER_HPP
#define HIERARCHICAL_WRITER_HPP

#include <zisa/io/hierarchical_file.hpp>

namespace zisa {

class HierarchicalWriter : public virtual HierarchicalFile {
public:
  /// Write a multi-dimensional array to an hierarchical file.
  /** @param data  Raw pointer to the data.
   *  @param data_type  Data type identifier.
   *  @param tag  Name of the data field in the file.
   *  @param rank  Number of dimension of the array.
   *  @param dims  Size (number of elements) of each dimension.
   */
  void write_array(void const *data,
                   const DataType &data_type,
                   const std::string &tag,
                   int rank,
                   std::size_t const *dims);

  /// Write a scalar to an HDF5 file.
  /**  @param data  The scalar to be written to file.
   *   @param tag  Name of the scalar in the HDF5 file.
   */
  template <typename T>
  void write_scalar(const T &scalar, const std::string &tag) {
    auto data_type = erase_data_type<T>();
    write_scalar((void *)&scalar, data_type, tag);
  }

  /// Write a scalar to a hierarchical file.
  /** @param data  Address of the scalar.
   *  @param data_type  Data type identifier.
   *  @param tag  Name of the scalar in the file.
   */
  void write_scalar(void const *addr,
                    const DataType &data_type,
                    const std::string &tag);

  /// Write a C++ string to the hierarchical file.
  /** @param data  String to write to the file.
   *  @param tag  Name of the string in the file.
   */
  void write_string(const std::string &data, const std::string &tag);

protected:
  virtual void do_write_array(void const *data,
                              const DataType &data_type,
                              const std::string &tag,
                              int rank,
                              std::size_t const *dims)
      = 0;

  virtual void do_write_scalar(void const *addr,
                               const DataType &data_type,
                               const std::string &tag)
      = 0;

  virtual void do_write_string(const std::string &data, const std::string &tag)
      = 0;
};

}

#endif // HIERARCHICAL_WRITER_HPP
