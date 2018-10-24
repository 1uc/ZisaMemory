/* Interface for serial HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2016-06-11
 */
#include "zisa/io/hdf5_serial_writer.hpp"

namespace zisa {
HDF5SerialWriter::HDF5SerialWriter(const std::string &filename) {
  hid_t h5_file
      = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  if (h5_file < 0) {
    LOG_ERR(string_format("Can't open file. [%s]", filename.c_str()));
  }

  file.push(h5_file);
}

void HDF5SerialWriter::write_array(void const *const data,
                                   const HDF5DataType &data_type,
                                   const std::string &tag,
                                   const int rank,
                                   hsize_t const *const dims) const {
  assert(data_type() > 0);
  assert(rank > 0);

  std::size_t urank = static_cast<std::size_t>(rank);

  // create a simple dataspace for storing an array of fixed size 'dims'.
  hid_t dataspace = H5Screate_simple(rank, dims, NULL);

  // create properties list for chunking and compression
  hid_t properties = H5Pcreate(H5P_DATASET_CREATE);

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
    H5Pset_chunk(properties, rank, &chunks[0]);
  }
  if (is_compressed) {
    H5Pset_deflate(properties, compression_level);
  }

  // set meta data, where the data is say to represent the pressure.
  hid_t dataset = H5Dcreate(file.top(),
                            tag.c_str(),
                            data_type(),
                            dataspace,
                            H5P_DEFAULT,
                            properties,
                            H5P_DEFAULT);

  assert(dataset >= 0);

  // finally, write the data
  H5Dwrite(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Pclose(properties);
}

void HDF5SerialWriter::write_scalar(void const *const data,
                                    const HDF5DataType &data_type,
                                    const std::string &tag) const {
  // create a scalar data space.
  hid_t dataspace = H5Screate(H5S_SCALAR);

  // create dataspace
  hid_t dataset = H5Dcreate(file.top(),
                            tag.c_str(),
                            data_type(),
                            dataspace,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

  // write the scalar
  H5Dwrite(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

void HDF5SerialWriter::write_string(const std::string &data,
                                    const std::string &tag) const {
  // strings can be stored as 1d-arrays of characters.
  // don't forget the null-character at the end of 'data.c_str()'.
  hsize_t dims[1] = {data.size() + 1};
  hid_t dataspace = H5Screate_simple(1, dims, NULL);

  // this type of characters
  HDF5DataType data_type = make_hdf5_data_type<char>();

  // create dataset
  hid_t dataset = H5Dcreate(file.top(),
                            tag.c_str(),
                            data_type(),
                            dataspace,
                            H5P_DEFAULT,
                            H5P_DEFAULT,
                            H5P_DEFAULT);

  // write the string
  H5Dwrite(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.c_str());

  // close
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

HDF5SerialReader::HDF5SerialReader(const std::string &filename) {
  hid_t h5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (h5_file < 0) {
    LOG_ERR(string_format("Failed to open file. [%s]", filename.c_str()));
  }

  file.push(h5_file);
  path.push_back(filename);
}

void *HDF5SerialReader::read_array(const HDF5DataType &data_type,
                                   const std::string &tag,
                                   int rank,
                                   hsize_t *const dims) const {
  // open the dataset
  hid_t dataset = open_dataset(tag);

  // open a dataspace, which knows about size of array etc.
  hid_t dataspace = get_dataspace(dataset);

  // read the dimensions
  get_dims(dataspace, dims, rank);

  size_t data_size = 1;
  for (int i = 0; i < rank; ++i) {
    data_size *= dims[i];
  }
  assert(data_size > 0);

  // allocate and read data
  void *data = malloc(data_size * data_type.size);
  auto status
      = H5Dread(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  if (status < 0) {
    LOG_ERR(string_format("Failed to read dataset, tag = %s [%s]",
                          tag.c_str(),
                          hierarchy().c_str()));
  }

  // close
  H5Dclose(dataset);
  H5Sclose(dataspace);

  return data;
}

void HDF5SerialReader::read_scalar(void *const data,
                                   const HDF5DataType &data_type,
                                   const std::string &tag) const {
  hid_t dataset = open_dataset(tag);
  hid_t dataspace = get_dataspace(dataset);

  // read the scalar
  H5Dread(dataset, data_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

  // close
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

std::string HDF5SerialReader::read_string(const std::string &tag) const {
  HDF5DataType data_type = make_hdf5_data_type<char>();

  int ndims = 1;
  hsize_t dims;

  char *raw_data = (char *)read_array(data_type, tag, ndims, &dims);
  auto str = std::string(raw_data, dims);
  free(raw_data);

  return str;
}

} // namespace zisa
