#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/copy_bytes.hpp>

namespace zisa {

__global__ void set_on_device(double *ptr, int n_elements, double value) {
  if (threadIdx.x < n_elements) {
    ptr[threadIdx.x] = value;
  }
}

void check_copy_bytes() {
  int n_elements = 5;
  std::size_t n_bytes = n_elements * sizeof(double);
  double value = 42.0;
  double off_value = 0.0;

  auto block_dims = dim3(1, 1, 1);
  auto thread_dims = dim3(32, 1, 1);

  double *h_ptr_a = (double *)malloc(n_elements * sizeof(double));
  double *h_ptr_b = (double *)malloc(n_elements * sizeof(double));

  double *d_ptr_a = nullptr;
  auto cuda_error = cudaMalloc(&d_ptr_a, n_elements * sizeof(double));
  REQUIRE(cuda_error == cudaSuccess);

  set_on_device<<<block_dims, thread_dims>>>(d_ptr_a, n_elements, value);
  copy_bytes(h_ptr_a, device_type::cpu, d_ptr_a, device_type::cuda, n_bytes);

  for (int i = 0; i < n_elements; ++i) {
    REQUIRE(h_ptr_a[i] == value);
  }

  set_on_device<<<block_dims, thread_dims>>>(d_ptr_a, n_elements, off_value);
  copy_bytes(d_ptr_a, device_type::cuda, h_ptr_a, device_type::cpu, n_bytes);
  copy_bytes(h_ptr_b, device_type::cpu, d_ptr_a, device_type::cuda, n_bytes);

  for (int i = 0; i < n_elements; ++i) {
    REQUIRE(h_ptr_b[i] == value);
  }

  free(h_ptr_a);
  free(h_ptr_b);
  cudaFree(d_ptr_a);
}

}

TEST_CASE("copy_bytes; simple", "[cuda]") { zisa::check_copy_bytes(); }
