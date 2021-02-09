#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/array_view.hpp>

TEST_CASE("array_view; memory_location", "[cuda][array]") {
  int n_elements = 6;
  auto shape = shape_t<2>{3, 2};

  double * h_ptr = (double *) malloc(n_elements * sizeof(double));
  double * d_ptr = nullptr;
  auto cuda_error = cudaMalloc(&d_ptr_a, n_elements * sizeof(double));
  REQUIRE(cuda_error == cudaSuccess);

  auto d_view = array_view<double, 2>(d_ptr, shape, device_type::cuda);
  auto h_view = array_view<double, 2>(h_ptr, shape, device_type::cpu);
  auto view = array_view<double, 2>(h_ptr, shape);

  REQUIRE(memory_location(d_view) == device_type::cuda);
  REQUIRE(memory_location(h_view) == device_type::cpu);
  REQUIRE(memory_location(view) == device_type::unknown);
}

