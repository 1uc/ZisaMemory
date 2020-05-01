#include <zisa/io/hdf5_writer.hpp>

namespace zisa {

void HDF5Writer::write_scalar(void const *addr,
                              const HDF5DataType &data_type,
                              const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_write_scalar(addr, data_type, tag);
}

void HDF5Writer::write_string(const std::string &data,
                              const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_write_string(data, tag);
}

void HDF5Writer::write_array(void const *data,
                             const HDF5DataType &data_type,
                             const std::string &tag,
                             int rank,
                             hsize_t const *dims) {

  auto lock = std::lock_guard(hdf5_mutex);
  do_write_array(data, data_type, tag, rank, dims);
}

std::string HDF5Reader::read_string(const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  return do_read_string(tag);
}

void HDF5Reader::read_scalar(void *data,
                             const HDF5DataType &data_type,
                             const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_read_scalar(data, data_type, tag);
}

void HDF5Reader::read_array(void *data,
                            const HDF5DataType &data_type,
                            const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_read_array(data, data_type, tag);
}

std::vector<hsize_t> HDF5Reader::dims(const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  return do_dims(tag);
}

}