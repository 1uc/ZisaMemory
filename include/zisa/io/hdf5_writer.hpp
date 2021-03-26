/* File I/O with HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2014-09-03
 */
#ifndef HDF5_WRITER_H_2GZ8LVPI
#define HDF5_WRITER_H_2GZ8LVPI
#include <string>
#include <vector>

#include <zisa/config.hpp>
#include <zisa/io/hdf5.hpp>
#include <zisa/io/hdf5_file.hpp>
#include <zisa/io/hierarchical_reader.hpp>
#include <zisa/io/hierarchical_writer.hpp>

namespace zisa {

/// Interface for writing data to an HDF5 file.
class HDF5Writer : public virtual HDF5File, public virtual HierarchicalWriter {
protected:
  // Redirects `HierarchicalWriter::do_write_array` to the HDF5 implementation.
  virtual void do_write_array(void const *data,
                              const DataType &data_type,
                              const std::string &tag,
                              int rank,
                              std::size_t const *dims) override;

  // Redirects `HierarchicalWriter::do_write_array` to the HDF5 implementation.
  virtual void do_write_scalar(void const *addr,
                               const DataType &data_type,
                               const std::string &tag) override;

  // Provides the interface on in the HDF5 world which needs to be implemented.
  virtual void do_write_array(void const *data,
                              const HDF5DataType &data_type,
                              const std::string &tag,
                              int rank,
                              hsize_t const *dims)
      = 0;

  // Provides the interface on in the HDF5 world which needs to be implemented.
  virtual void do_write_scalar(void const *addr,
                               const HDF5DataType &data_type,
                               const std::string &tag)
      = 0;
};

/// Read data from HDF5 file.
class HDF5Reader : public virtual HDF5File, public virtual HierarchicalReader {
private:
  using super = HDF5File;

public:
protected:
  std::vector<std::size_t> do_dims(const std::string &tag) const;

  virtual std::vector<hsize_t> do_hdf5_dims(const std::string &tag) const = 0;

  // Translation from `HierarchicalReader::do_read_array` to HDF5Reader.
  virtual void do_read_array(void *data,
                             const DataType &data_type,
                             const std::string &tag) const override;

  // This is where the actual functionality is implemented.
  virtual void do_read_array(void *data,
                             const HDF5DataType &data_type,
                             const std::string &tag) const = 0;

  // Translation from `HierarchicalReader::do_read_array` to HDF5Reader.
  virtual void do_read_scalar(void *data,
                              const DataType &data_type,
                              const std::string &tag) const override;

  // This is where the actual functionality is implemented.
  virtual void do_read_scalar(void *data,
                              const HDF5DataType &data_type,
                              const std::string &tag) const = 0;
};

} // namespace zisa
#endif /* end of include guard: HDF5_WRITER_H_2GZ8LVPI */
