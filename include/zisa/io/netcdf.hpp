// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef NETCDF_H
#define NETCDF_H

#if ZISA_HAS_NETCDF == 0
#error "Must be compiled with `ZISA_HAS_NETCDF == 1`".
#endif

#include <netcdf.h>
#include <string>
#include <vector>

#include <zisa/config.hpp>
#include <zisa/utils/integer_cast.hpp>

namespace zisa::io::netcdf {

int create(std::string filename, int cmode);

int def_dim(int file_id, const std::string &name, std::size_t extent);

int def_var(int file_id,
            const std::string &name,
            int type_id,
            const std::vector<int> &dims);

void put_var(int file_id, int var_id, void const *data);

} // namespace zisa::io::netcdf

#endif // NETCDF_H
