#ifndef FILL_CUDA_IMPL_CUH_VEQOI
#define FILL_CUDA_IMPL_CUH_VEQOI

#include <zisa/config.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/utils/integer_cast.hpp>

namespace zisa {

template <class T>
__global__ void fill_kernel(T *const ptr, int_t n_elements, T value) {
  int_t i = threadIdx.x + blockDim.x * blockIdx.x;

  while (i < n_elements) {
    ptr[i] = value;
    i += gridDim.x * blockDim.x;
  }
}

template <class T>
void fill_cuda(T *const ptr, int_t n_elements, const T &value) {
  int thread_dims = 1024;
  int block_dims
      = zisa::min(div_up(integer_cast<int>(n_elements), thread_dims), 1024);

  fill_kernel<<<thread_dims, block_dims>>>(ptr, n_elements, value);
  ZISA_CHECK_CUDA_DEBUG;
}

}
#endif