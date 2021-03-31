#ifndef NETCDF_H
#define NETCDF_H

#include <netcdf.h>
#include <string>
#include <zisa/config.hpp>

namespace zisa::io::netcdf {

int create(std::string filename, int cmode) {
  int ncid = -1;
  auto status = nc_create(filename.c_str(), cmode, &ncid);

  LOG_ERR_IF(status != NC_NOERR, "Could not create file.");
  return ncid;
}

int def_dim(int file_id, const std::string &name, int extent) {
  int id = -1;
  auto status = nc_def_dim(file_id, name.c_str(), extent, &id);
  LOG_ERR_IF(status != NC_NOERR, "Failed to create dimension.");

  return id;
}

} // namespace zisa::io::netcdf

#endif // NETCDF_H
