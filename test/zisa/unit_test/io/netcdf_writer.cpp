#include <zisa/testing/testing_framework.hpp>

#include <zisa/io/netcdf_file.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace zisa {

void check_netcdf_writer() {
  int_t n_lat = 2;
  int_t n_lon = 3;

  using dim_t = std::tuple<std::string, std::size_t>;
  using var_t
      = std::tuple<std::string, std::vector<std::string>, ErasedDataType>;

  auto erased_double = erase_data_type<double>();
  auto filename = "__netcdf-a.nc";

  auto file_structure
      = NetCDFFileStructure({dim_t{"lat", n_lat}, dim_t{"lon", n_lon}},
                            {var_t{"foo", {"lat", "lon"}, erased_double}});

  auto writer = NetCDFSerialWriter(filename, file_structure);

  auto foo = zisa::array<double, 2>({n_lat, n_lon});
  for (int_t i = 0; i < n_lat; i++) {
    for (int_t j = 0; j < n_lon; ++j) {
      foo(i, j) = zisa::sin(double(i * j) / double(n_lat) + 0.1);
    }
  }

  save(writer, foo, "foo");
}
}

TEST_CASE("NetCDF; write a file", "[io][netcdf]") {
  zisa::check_netcdf_writer();
}
