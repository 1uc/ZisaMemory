#if ZISA_HAS_NETCDF == 1

#include <zisa/io/netcdf.hpp>

namespace zisa::io::netcdf {

int create(std::string filename, int cmode) {
  int ncid = -1;
  auto status = nc_create(filename.c_str(), cmode, &ncid);

  LOG_ERR_IF(status != NC_NOERR, "Could not create file.");
  return ncid;
}

int def_dim(int file_id, const std::string &name, std::size_t extent) {
  int id = -1;
  auto status = nc_def_dim(file_id, name.c_str(), extent, &id);
  LOG_ERR_IF(status != NC_NOERR, "Failed to create dimension.");

  return id;
}

int def_var(int file_id,
            const std::string &name,
            int type_id,
            const std::vector<int> &dims) {

  auto id = -1;
  auto ndims = zisa::integer_cast<int>(dims.size());
  auto status
      = nc_def_var(file_id, name.c_str(), type_id, ndims, dims.data(), &id);
  LOG_ERR_IF(status != NC_NOERR, "Failed to create variable.");

  return id;
}

void put_var(int file_id, int var_id, void const *data) {
  auto status = nc_put_var(file_id, var_id, data);
  LOG_ERR_IF(status != NC_NOERR, "Failed to write variable.");
}
}

#endif
