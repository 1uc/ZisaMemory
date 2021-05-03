/* File I/O using HDF5.
 *
 * Authors: Luc Grosheintz <forbugrep@zoho.com>
 *    Date: 2014-09-03
 */

#if ZISA_HAS_HDF5 == 1

#include <zisa/io/hdf5_file.hpp>

#include <zisa/io/concatenate.hpp>
#include <zisa/io/hdf5.hpp>

namespace zisa {
std::recursive_mutex hdf5_mutex;

HDF5DataType::HDF5DataType(const hid_t &h5_type, size_t size)
    : size(size), h5_type(h5_type) {}

HDF5DataType::~HDF5DataType() {
  auto lock = std::lock_guard(hdf5_mutex);
  zisa::H5T::close(h5_type);
}

/// Return the HDF5 identifier of the data-type.
hid_t HDF5DataType::operator()() const {
  assert(h5_type > 0);
  return h5_type;
}

HDF5DataType make_hdf5_data_type(const hid_t &hdf5_data_type, size_t size) {
  auto lock = std::lock_guard(hdf5_mutex);
  hid_t h5_type = zisa::H5T::copy(hdf5_data_type);
  return HDF5DataType(h5_type, size);
}

#define ZISA_REGISTER_ERASED_DATA_TYPE(type, TYPE)                             \
  if (data_type == ErasedDataType::TYPE) {                                     \
    return make_hdf5_data_type<type>();                                        \
  }

HDF5DataType make_hdf5_data_type(const ErasedDataType &data_type) {
  ZISA_REGISTER_ERASED_DATA_TYPE(double, DOUBLE);
  ZISA_REGISTER_ERASED_DATA_TYPE(float, FLOAT);
  ZISA_REGISTER_ERASED_DATA_TYPE(unsigned long, UNSIGNED_LONG);
  ZISA_REGISTER_ERASED_DATA_TYPE(int, INT);
  ZISA_REGISTER_ERASED_DATA_TYPE(char, CHAR);

  LOG_ERR("Implement missing case.");
}

#undef ZISA_REGISTER_ERASED_DATA_TYPE

std::vector<hsize_t> make_hdf5_dims(std::size_t const *const dims, int rank) {
  auto hdf5_dims = std::vector<hsize_t>(zisa::integer_cast<std::size_t>(rank));
  std::copy(dims, dims + rank, hdf5_dims.begin());
  return hdf5_dims;
}

HDF5File::~HDF5File() {
  auto lock = std::lock_guard(hdf5_mutex);

  // all IDs not on the bottom are group IDs
  while (file.size() > 1) {
    zisa::H5G::close(file.top());
    file.pop();
  }

  // the bottom of the stack is a file ID
  zisa::H5F::close(file.top());
  file.pop();
}

void HDF5File::do_open_group(const std::string &group_name) {
  auto lock = std::lock_guard(hdf5_mutex);

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

void HDF5File::do_close_group() {
  auto lock = std::lock_guard(hdf5_mutex);

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

void HDF5File::do_switch_group(const std::string &group_name) {
  auto lock = std::lock_guard(hdf5_mutex);
  close_group();
  open_group(group_name);
}

bool HDF5File::do_group_exists(const std::string &group_name) const {
  auto lock = std::lock_guard(hdf5_mutex);
  return H5Lexists(file.top(), group_name.c_str(), H5P_DEFAULT);
}

std::string HDF5File::do_hierarchy() const {
  return zisa::concatenate(path.begin(), path.end(), "/");
}

hid_t HDF5File::open_dataset(const std::string &tag) const {
  auto lock = std::lock_guard(hdf5_mutex);
  hid_t dataset = H5Dopen(file.top(), tag.c_str(), H5P_DEFAULT);

  if (dataset < 0) {
    LOG_ERR(string_format("Failed to open dataset, tag = %s [%s]",
                          tag.c_str(),
                          hierarchy().c_str()));
  }

  return dataset;
}

hid_t HDF5File::get_dataspace(const hid_t &dataset) const {
  auto lock = std::lock_guard(hdf5_mutex);
  hid_t dataspace = H5Dget_space(dataset);

  if (dataspace < 0) {
    LOG_ERR(string_format("Failed to open dataspace (%d).", dataset));
  }

  return dataspace;
}

} // namespace zisa

#endif
