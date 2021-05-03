#if ZISA_HAS_HDF5 == 1
#include <zisa/io/hdf5_writer.hpp>

namespace zisa {

void HDF5Writer::do_write_scalar(const void *addr,
                                 const HierarchicalFile::DataType &data_type,
                                 const std::string &tag) {
  auto lock = std::lock_guard(hdf5_mutex);

  auto hdf5_data_type = make_hdf5_data_type(data_type);
  do_write_scalar(addr, hdf5_data_type, tag);
}

void HDF5Writer::do_write_array(const void *data,
                                const HierarchicalFile::DataType &data_type,
                                const std::string &tag,
                                int rank,
                                const std::size_t *dims) {
  auto lock = std::lock_guard(hdf5_mutex);

  auto hdf5_data_type = make_hdf5_data_type(data_type);
  std::vector<hsize_t> hdf5_dims = make_hdf5_dims(dims, rank);
  do_write_array(data, hdf5_data_type, tag, rank, hdf5_dims.data());
}

std::vector<std::size_t> HDF5Reader::do_dims(const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);

  auto hdf5_dims = do_hdf5_dims(tag);
  std::vector<std::size_t> dims(hdf5_dims.size());
  std::copy(hdf5_dims.begin(), hdf5_dims.end(), dims.begin());

  return dims;
}

void HDF5Reader::do_read_array(void *data,
                               const HierarchicalFile::DataType &data_type,
                               const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_read_array(data, make_hdf5_data_type(data_type), tag);
}

void HDF5Reader::do_read_scalar(void *data,
                                const HierarchicalFile::DataType &data_type,
                                const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  do_read_scalar(data, make_hdf5_data_type(data_type), tag);
}
}
#endif
