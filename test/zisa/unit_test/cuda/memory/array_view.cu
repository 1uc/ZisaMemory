// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/array_view.hpp>

namespace zisa {

static void test_memory_location() {
  int n_elements = 6;
  auto shape = shape_t<2>{3, 2};

  double *h_ptr = (double *)malloc(n_elements * sizeof(double));
  double *d_ptr = nullptr;
  auto cuda_error = cudaMalloc(&d_ptr, n_elements * sizeof(double));
  REQUIRE(cuda_error == cudaSuccess);

  auto d_view = array_view<double, 2>(shape, d_ptr, device_type::cuda);
  auto h_view = array_view<double, 2>(shape, h_ptr, device_type::cpu);
  auto view = array_view<double, 2>(shape, h_ptr);

  REQUIRE(memory_location(d_view) == device_type::cuda);
  REQUIRE(memory_location(h_view) == device_type::cpu);
  REQUIRE(memory_location(view) == device_type::unknown);
}

}

TEST_CASE("array_view; memory_location", "[cuda][array]") {
  zisa::test_memory_location();
}
