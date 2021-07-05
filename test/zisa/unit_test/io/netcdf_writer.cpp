// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#if ZISA_HAS_NETCDF

#include <zisa/testing/testing_framework.hpp>

#include <zisa/io/netcdf_file.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace zisa {
static array<double, 2> make_foo(int_t n_lat, int_t n_lon) {
  auto foo = zisa::array<double, 2>({n_lat, n_lon});
  for (int_t i = 0; i < n_lat; i++) {
    for (int_t j = 0; j < n_lon; ++j) {
      foo(i, j) = zisa::sin(double(i * j) / double(n_lat) + 0.1);
    }
  }

  return foo;
}

static array<int, 1> make_bar(int_t n_steps) {
  auto bar = zisa::array<int, 1>(n_steps);
  for (int_t i = 0; i < n_steps; ++i) {
    bar[i] = integer_cast<int>(i);
  }

  return bar;
}

void check_netcdf_writer() {
  int_t n_lat = 2;
  int_t n_lon = 3;
  int_t n_steps = 4;

  auto filename = "__netcdf-a.nc";

  // In NetCDF, a dimension has a name and an extent:
  using dim_t = std::tuple<std::string, std::size_t>;

  // Further, a array-valued variable consists of:
  //
  //  * a name
  //  * the names of its dimensions
  //  * a data type of its elements.
  //
  // Scalar variables are simply zero-dimensional variables.
  using var_t
      = std::tuple<std::string, std::vector<std::string>, ErasedDataType>;

  // Types can be erased as follows:
  auto erased_float = erase_data_type<float>();
  auto erased_double = erase_data_type<double>();
  auto erased_int = erase_data_type<int>();

  // There shall be three axes (aka. "dimensions"), a latitude ("lat"), a
  // longitude ("lon") and time ("time").
  // Futhermore, there will be two arrays, a 2D lat/lon array ("foo") of
  // doubles and a 1D time array ("bar") of `int`s.  Finally, we want to save a
  // scalar called "blubb".

  // The NetCDF file structure must therefore be:
  // clang-format off
  auto file_structure = NetCDFFileStructure(
      {
        dim_t{"time", n_steps},
        dim_t{"lat", n_lat},
        dim_t{"lon", n_lon}
      },
      {
        var_t{"foo", {"lat", "lon"}, erased_double},
        var_t{"bar", {"time"}, erased_int},
        var_t{"blubb", {}, erased_float}
      });
  // clang-format on

  auto writer = NetCDFSerialWriter(filename, file_structure);

  auto foo = make_foo(n_lat, n_lon);
  auto bar = make_bar(n_steps);

  save(writer, foo, "foo");
  save(writer, bar, "bar");
  writer.write_scalar(42.0f, "blubb");
}
}

TEST_CASE("NetCDF; write a file", "[io][netcdf]") {
  zisa::check_netcdf_writer();
}

#endif
