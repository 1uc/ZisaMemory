#if ZISA_HAS_NETCDF == 1
#include <zisa/io/netcdf_file.hpp>

namespace zisa {

#ifndef NC_UNSIGNED_LONG
#define NC_UNSIGNED_LONG (sizeof(unsigned long) == 4 ? NC_INT : NC_INT64)
#else
#error "`NC_UNSIGNED_LONG` has already been defined."
#endif

#define REGISTER_NETCDF_DATA_TYPE(DATA_TYPE, NETCDF_DATA_TYPE)                 \
  if (data_type == ErasedDataType::DATA_TYPE) {                                \
    return NETCDF_DATA_TYPE;                                                   \
  }

int make_netcdf_data_type(const ErasedDataType &data_type) {
  REGISTER_NETCDF_DATA_TYPE(CHAR, NC_CHAR);
  REGISTER_NETCDF_DATA_TYPE(FLOAT, NC_FLOAT);
  REGISTER_NETCDF_DATA_TYPE(DOUBLE, NC_DOUBLE);
  REGISTER_NETCDF_DATA_TYPE(INT, NC_INT);
  REGISTER_NETCDF_DATA_TYPE(UNSIGNED_LONG, NC_UNSIGNED_LONG);

  LOG_ERR("Implement missing case.");
}

#undef REGISTER_NETCDF_DATA_TYPE

NetCDFFileStructure::NetCDFFileStructure(const vector_dims_t &dims,
                                         const vector_vars_t &vars) {
  add_dims(dims);
  add_vars(vars);
}

const auto &NetCDFFileStructure::dims() const { return dims_; }

const auto &NetCDFFileStructure::vars() const { return vars_; }

void NetCDFFileStructure::add_dim(const dim_t &dim) { dims_.push_back(dim); }

void NetCDFFileStructure::add_dims(const vector_dims_t &dims) {
  for (const auto &dim : dims) {
    add_dim(dim);
  }
}

void NetCDFFileStructure::add_vars(const vector_vars_t &vars) {
  for (const auto &var : vars) {
    add_var(var);
  }
}

void NetCDFFileStructure::add_var(const var_t &var) { vars_.push_back(var); }

NetCDFFile::NetCDFFile(const std::string &filename,
                       const NetCDFFileStructure &structure) {
  auto file_id = zisa::io::netcdf::create(filename, NC_CLOBBER | NC_NETCDF4);

  file_ids_.push(file_id);
  path_.push_back(filename);

  const auto &dims = structure.dims();
  for (const auto &[name, extent] : dims) {
    dim_ids_[name] = zisa::io::netcdf::def_dim(file_id, name, extent);
  }

  const auto &vars = structure.vars();
  for (const auto &var : vars) {
    const auto &[var_name, dim_names, data_type] = var;

    auto type_id = make_netcdf_data_type(data_type);
    auto dim_ids = extract_dim_ids(dim_names);
    var_ids_[var_name]
        = zisa::io::netcdf::def_var(file_id, var_name, type_id, dim_ids);
  }
}

void NetCDFFile::do_open_group(const std::string & /* group_name */) {
  // This has strong ties to defining variables.
  LOG_ERR("Implement first.");
}

void NetCDFFile::do_close_group() { LOG_ERR("Implement first"); }

void NetCDFFile::do_switch_group(const std::string &group_name) {
  do_close_group();
  do_open_group(group_name);
}

bool NetCDFFile::do_group_exists(const std::string & /* group_name */) const {
  LOG_ERR("Implement first.");
}

std::string NetCDFFile::do_hierarchy() const {
  return zisa::concatenate(path_.begin(), path_.end(), "/");
}

std::vector<int>
NetCDFFile::extract_dim_ids(const std::vector<std::string> &dim_names) const {
  std::vector<int> dims;
  for (const auto &dim_name : dim_names) {
    dims.push_back(dim_ids_.at(dim_name));
  }

  return dims;
}

void NetCDFFile::do_unlink(const std::string & /* tag */) {
  LOG_ERR("Implement this first.");
}
}

#endif
