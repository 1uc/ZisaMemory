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

  virtual void do_write_array(const void *data,
                           const HDF5DataType &data_type,
                           const std::string &tag,
                           int rank,
                              const hsize_t *dims) const override;

  using HDF5Writer::do_write_scalar;

  virtual void do_write_scalar(const void *data,
                            const HDF5DataType &data_type,
                            const std::string &tag) const override;

  virtual void do_write_string(const std::string &data,
                            const std::string &tag) const override;
};

/// Read data from HDF5 file sequentially.
class HDF5SerialReader : public HDF5Reader {
private:
  using super = HDF5Reader;

public:
  explicit HDF5SerialReader(const std::string &filename);
  virtual ~HDF5SerialReader() = default;

  virtual std::vector<hsize_t> do_dims(const std::string &tag) const override;

  virtual void do_read_array(void *data,
                          const HDF5DataType &data_type,
                          const std::string &tag) const override;

  using HDF5Reader::read_scalar;

  virtual void do_read_scalar(void *data,
                           const HDF5DataType &data_type,
                           const std::string &tag) const override;

  virtual std::string do_read_string(const std::string &tag) const override;
};

template<class T, class... Args>
T load_serial(const std::string &filename, Args&& ...args){
  auto reader = HDF5SerialReader(filename);
  return T::load(reader, std::forward<Args>(args)...);
}

} // namespace zisa
#endif /* end of include guard: HDF5_SERIAL_WRITER_H_RVN8KWYI */
