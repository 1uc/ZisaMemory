/* Interface for serial HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2016-06-11
 */
#ifndef HDF5_SERIAL_WRITER_H_RVN8KWYI
#define HDF5_SERIAL_WRITER_H_RVN8KWYI

#include "zisa/config.hpp"
#include "zisa/io/hdf5_writer.hpp"

namespace zisa {
/// Write data to an HDF5 file, serial version.
class HDF5SerialWriter : public HDF5Writer {
public:
  explicit HDF5SerialWriter(const std::string &filename);
  virtual ~HDF5SerialWriter() = default;

  virtual void write_array(void const *const data,
                           const HDF5DataType &data_type,
                           const std::string &tag,
                           const int rank,
                           hsize_t const *const dims) const override;

  using HDF5Writer::write_scalar;

  virtual void write_scalar(void const *const data,
                            const HDF5DataType &data_type,
                            const std::string &tag) const override;

  virtual void write_string(const std::string &data,
                            const std::string &tag) const override;
};

/// Read data from HDF5 file sequentially.
class HDF5SerialReader : public HDF5Reader {
private:
  using super = HDF5Reader;

public:
  explicit HDF5SerialReader(const std::string &filename);
  virtual ~HDF5SerialReader() = default;

  virtual std::vector<hsize_t> dims(const std::string &tag) const override;

  virtual void read_array(void *data,
                          const HDF5DataType &data_type,
                          const std::string &tag) const override;

  using HDF5Reader::read_scalar;

  virtual void read_scalar(void *const data,
                           const HDF5DataType &data_type,
                           const std::string &tag) const override;

  virtual std::string read_string(const std::string &tag) const override;
};

} // namespace zisa
#endif /* end of include guard: HDF5_SERIAL_WRITER_H_RVN8KWYI */
