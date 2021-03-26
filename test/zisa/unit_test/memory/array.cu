#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#include <zisa/memory/column_major.hpp>
#include <zisa/testing/testing_framework.hpp>

using namespace zisa;

__global__ void call_kernel(array_view<double, 2> y, array<double, 2> x) {
  if(threadIdx.x == 0) {
    x(0, 0) = 42.0;
    y(0, 0) = x(0, 0);
  }
}

TEST_CASE("array/array_view; API", "[cuda][array]") {
  auto a = array<double, 2>({3, 2}, device_type::cuda);
  auto b = array<double, 2>({3, 2}, device_type::cuda);

  auto b_view =  array_view<double, 2>(b);

  call_kernel<<<1, 1>>>(b_view, a);
}