#if ZISA_HAS_HDF5 == 1
#include <zisa/testing/testing_framework.hpp>

#include <zisa/io/hdf5_serial_writer.hpp>

#include <zisa/memory/array.hpp>

TEST_CASE("hdf5; basic API", "[hdf5]") {
  auto writer_ = zisa::HDF5SerialWriter("__unit_tests_c.h5");
  auto &writer = static_cast<zisa::HierarchicalWriter &>(writer_);

  writer.open_group("foo");
  writer.open_group("bar");

  REQUIRE(writer.hierarchy() == "foo/bar");

  writer.switch_group("baz");
  REQUIRE(writer.hierarchy() == "foo/baz");

  writer.close_group();
  REQUIRE(writer.hierarchy() == "foo");

  REQUIRE(!writer.group_exists("foo"));
  REQUIRE(writer.group_exists("bar"));
  REQUIRE(writer.group_exists("baz"));
}

TEST_CASE("hdf5; read/write", "[array][hdf5]") {

  auto a = zisa::array<double, 3>({3ul, 3ul, 2ul});

  zisa::fill(a, 42.0);
  {
    auto writer = zisa::HDF5SerialWriter("__unit_tests_a.h5");
    zisa::save(writer, a, "a");
  }

  {
    auto reader = zisa::HDF5SerialReader("__unit_tests_a.h5");
    auto b = zisa::array<double, 3>::load(reader, "a");

    REQUIRE(a.shape() == b.shape());
    for (zisa::int_t i = 0; i < a.size(); ++i) {
      REQUIRE(a[i] == b[i]);
    }
  }
}
#endif
