/* Interface for serial HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2016-06-11
 */
#if ZISA_HAS_HDF5
#include "zisa/io/hdf5_serial_writer.hpp"

namespace zisa {
HDF5SerialWriter::HDF5SerialWriter(const std::string &filename) {
  auto lock = std::lock_guard(hdf5_mutex);

  hid_t h5_file
      = H5F::create(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  file.push(h5_file);
}

void HDF5SerialWriter::do_write_array(const void *data,
                                      const HDF5DataType &data_type,
                                      const std::string &tag,
                                      int rank,
                                      const hsize_t *dims) {
  assert(data_type() > 0);
  assert(rank > 0);

  auto urank = static_cast<std::size_t>(rank);

  // create a simple dataspace for storing an array of fixed size 'dims'.
  hid_t dataspace = H5S::create_simple(rank, dims, nullptr);

  // create properties list for chunking and compression
  hid_t properties = H5P::create(H5P_DATASET_CREATE);

  std::vector<hsize_t> chunks(urank);

  hsize_t chunk_size = 32;
  for (std::size_t i = 0; i < urank; ++i) {
    if (rank == 1) {
      chunks[i] = chunk_size * chunk_size;
    } else {
      chunks[i] = (i < 2 ? chunk_size : 1);
    }

    chunks[i] = std::min(dims[i], chunks[i]);
  }

  bool is_compressed = true;
  bool is_chunked = true || is_compressed; // compression requires chunking
  unsigned int compression_level = 6u;

  if (is_chunked) {
    H5P::set_chunk(properties, rank, &chunks[0]);
  }
  if (is_compressed) {
    H5P::set_deflate(properties, compression_level);
  }

  // set meta data, where the data is say to represent the pressure.
  hid_t dataset = H5D::create(file.top(),
                              tag.c_str(),
                              data_type(),
                              dataspace,
                              H5P_DEFAULT,
                              properties,
                              H5P_DEFAULT);

  // finally, write the data
  H5D::write(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5D::close(dataset);
  H5S::close(dataspace);
  H5P::close(properties);
}

void HDF5SerialWriter::do_write_scalar(const void *data,
                                       const HDF5DataType &data_type,
                                       const std::string &tag) {
  // create a scalar data space.
  hid_t dataspace = H5S::create(H5S_SCALAR);

  // create dataspace
  hid_t dataset = H5D::create(file.top(),
                              tag.c_str(),
                              data_type(),
                              dataspace,
                              H5P_DEFAULT,
                              H5P_DEFAULT,
                              H5P_DEFAULT);

  // write the scalar
  H5D::write(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5D::close(dataset);
  H5S::close(dataspace);
}

void HDF5SerialWriter::do_write_string(const std::string &data,
                                       const std::string &tag) {
  // strings can be stored as 1d-arrays of characters.
  // don't forget the null-character at the end of 'data.c_str()'.
  hsize_t dims[1] = {data.size() + 1};
  hid_t dataspace = H5S::create_simple(1, dims, nullptr);

  // this type of characters
  HDF5DataType data_type = make_hdf5_data_type<char>();

  // create dataset
  hid_t dataset = H5D::create(file.top(),
                              tag.c_str(),
                              data_type(),
                              dataspace,
                              H5P_DEFAULT,
                              H5P_DEFAULT,
                              H5P_DEFAULT);

  // write the string
  H5D::write(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.c_str());

  // close
  H5D::close(dataset);
  H5S::close(dataspace);
}

HDF5SerialReader::HDF5SerialReader(const std::string &filename) {
  auto lock = std::lock_guard(hdf5_mutex);
  hid_t h5_file = H5F::open(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  file.push(h5_file);
  path.push_back(filename);
}

std::vector<hsize_t>
HDF5SerialReader::do_hdf5_dims(const std::string &tag) const {
  hid_t dataset = open_dataset(tag);
  hid_t dataspace = get_dataspace(dataset);

  auto rank = static_cast<int_t>(H5S::get_simple_extent_ndims(dataspace));
  std::vector<hsize_t> dims(rank);

  H5S::get_simple_extent_dims(dataspace, &(dims[0]), nullptr);

  return dims;
}

void HDF5SerialReader::do_read_array(void *data,
                                     const HDF5DataType &data_type,
                                     const std::string &tag) const {
  hid_t dataset = open_dataset(tag);
  H5D::read(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5D::close(dataset);
}

void HDF5SerialReader::do_read_scalar(void *data,
                                      const HDF5DataType &data_type,
                                      const std::string &tag) const {
  hid_t dataset = open_dataset(tag);
  hid_t dataspace = get_dataspace(dataset);

  // read the scalar
  H5D::read(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5D::close(dataset);
  H5S::close(dataspace);
}

std::string HDF5SerialReader::do_read_string(const std::string &tag) const {
  HDF5DataType hdf5_data_type = make_hdf5_data_type<char>();

  auto length = dims(tag)[0];

  std::vector<char> buf(length);
  do_read_array(&buf[0], hdf5_data_type, tag);

  return std::string(&buf[0], buf.size());
}

} // namespace zisa
#endif
