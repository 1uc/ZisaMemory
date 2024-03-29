// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#include <zisa/memory/column_major.hpp>
#include <zisa/testing/testing_framework.hpp>
#include <zisa/utils/type_name.hpp>

#if ZISA_HAS_HDF5
#include <zisa/io/hdf5_serial_writer.hpp>
#endif

using namespace zisa;

bool implicit_conversion(const array_const_view<double, 3> &view, double *ptr) {
  return view.raw() == ptr;
}

TEST_CASE("array; basics", "[array]") {
  auto a = array<double, 3>({3ul, 3ul, 2ul}, device_type::cpu);
  auto b = array<double, 3>({3ul, 3ul, 2ul}, device_type::cpu);

  REQUIRE(a.raw() != static_cast<const decltype(b) &>(b).raw());
  REQUIRE(a.shape() == b.shape());

  auto b_view = array_view(b);
  auto bb_view = array_view<double, 3>(shape(b), raw_ptr(b));
  auto bc_view = array_const_view<double, 3>(shape(b), raw_ptr(b));

  REQUIRE(b.raw() == b_view.raw());
  REQUIRE(b.raw() == bb_view.raw());
  REQUIRE(b.raw() == bc_view.raw());

  b(1, 0, 1) = -42.0;
  REQUIRE(b_view(1, 0, 1) == -42.0);

  b_view(0, 0, 0) = 42.0;
  REQUIRE(b(0, 0, 0) == 42.0);

  REQUIRE(implicit_conversion(a, a.raw()));
}

#if ZISA_HAS_HDF5

template <class T>
void check_array() {
  auto filename = std::string("__unit_tests--array-to-hdf5.h5");
  auto label = "a";

  auto shape = zisa::shape_t<3>{3ul, 4ul, 2ul};

  auto a = array<T, 3>(shape);
  for (zisa::int_t i = 0; i < a.size(); ++i) {
    a[i] = T(i);
  }

  {
    auto writer = HDF5SerialWriter(filename);
    zisa::save(writer, a, label);
  }

  {
    auto reader = HDF5SerialReader(filename);
    auto dims = reader.dims(label);

    for (zisa::int_t k = 0; k < 3; ++k) {
      REQUIRE(shape[0] == static_cast<zisa::int_t>(dims[0]));
    }

    auto b = zisa::array<T, 3>::load(reader, label);
    REQUIRE(b == a);
  }

  zisa::delete_file(filename);
}

TEST_CASE("array; write to file", "[array]") {
  check_array<double>();
  check_array<bool>();
}

#endif

TEST_CASE("array; builds for general Indexing.", "[array]") {

  // The point is to check that `array<double, 3, Indexing>`
  // compiles fine. Despite the fact that `save` and `load` only
  // work if `Indexing == row_major`.

  auto shape = zisa::shape_t<3>{3ul, 4ul, 2ul};

  auto a = array<double, 3, column_major>(shape);
  for (zisa::int_t i = 0; i < a.size(); ++i) {
    a[i] = double(i);
  }
}

#if ZISA_HAS_CUDA

TEST_CASE("array; array_view (cuda)", "[cuda][array]") {
  auto a = array<double, 3>({3, 3, 2}, device_type::cuda);
  auto b = array<double, 3>({3, 3, 2}, device_type::cuda);

  REQUIRE(a.raw() != static_cast<const decltype(b) &>(b).raw());
  REQUIRE(a.shape() == b.shape());

  auto b_view = array_view<double, 3>(b);
  auto bb_view = array_view<double, 3>(shape(b), raw_ptr(b));
  auto bc_view = array_const_view<double, 3>(shape(b), raw_ptr(b));

  REQUIRE(b.raw() == b_view.raw());
  REQUIRE(b.raw() == bb_view.raw());
  REQUIRE(b.raw() == bc_view.raw());
  REQUIRE(implicit_conversion(a, a.raw()));
}

#endif
