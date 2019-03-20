/* File I/O using HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2014-09-03
 */

#include <zisa/io/hdf5_file.hpp>

#include <zisa/io/concatenate.hpp>
#include <zisa/io/hdf5.hpp>

namespace zisa {
HDF5DataType::HDF5DataType(const hid_t &h5_type, size_t size)
    : size(size), h5_type(h5_type) {}

HDF5DataType::~HDF5DataType() { zisa::H5T::close(h5_type); }

/// Return the HDF5 identifier of the data-type.
hid_t HDF5DataType::operator()() const {
  assert(h5_type > 0);
  return h5_type;
}

HDF5DataType make_hdf5_data_type(const hid_t &hdf5_data_type, size_t size) {
  hid_t h5_type = zisa::H5T::copy(hdf5_data_type);
  return HDF5DataType(h5_type, size);
}

HDF5File::~HDF5File() {
  // all IDs not on the bottom are group IDs
  while (file.size() > 1) {
    zisa::H5G::close(file.top());
    file.pop();
  }

  // the bottom of the stack is a file ID
  H5Fclose(file.top());
  file.pop();
}

void HDF5File::open_group(const std::string &group_name) {
  hid_t h5_group = -1;

  if (group_name.empty()) {
    LOG_ERR("Empty HDF5 group name.");
  }

  // if the group exists open it
  if (group_exists(group_name)) {
    h5_group = zisa::H5G::open(file.top(), group_name.c_str(), H5P_DEFAULT);
  }

  // otherwise create it
  else {
    h5_group = zisa::H5G::create(
        file.top(), group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }

  assert(h5_group > 0); // check for success
  file.push(h5_group);
  path.push_back(group_name);
}

void HDF5File::close_group() {
  if (file.size() == 1) {
    LOG_ERR("Closing group, but its a file.");
  }
  if (file.empty()) {
    LOG_ERR("Closing group, but tree is empty.");
  }

  zisa::H5G::close(file.top());
  file.pop();
  path.pop_back();
}

void HDF5File::switch_group(const std::string &group_name) {
  close_group();
  open_group(group_name);
}

bool HDF5File::group_exists(const std::string &group_name) const {
  return H5Lexists(file.top(), group_name.c_str(), H5P_DEFAULT);
}

hid_t HDF5File::open_dataset(const std::string &tag) const {
  hid_t dataset = H5Dopen(file.top(), tag.c_str(), H5P_DEFAULT);

  if (dataset < 0) {
    LOG_ERR(string_format("Failed to open dataset, tag = %s [%s]",
                          tag.c_str(),
                          hierarchy().c_str()));
  }

  return dataset;
}

hid_t HDF5File::get_dataspace(const hid_t &dataset) const {
  hid_t dataspace = H5Dget_space(dataset);

  if (dataspace < 0) {
    LOG_ERR(string_format("Failed to open dataspace (%d).", dataset));
  }

  return dataspace;
}

std::string HDF5File::hierarchy() const {
  return zisa::concatenate(path.begin(), path.end(), "/");
}

} // namespace zisa
