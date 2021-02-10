#ifndef FILL_CUDA_HPP
#define FILL_CUDA_HPP

#include <zisa/config.hpp>

namespace zisa {

template <class T>
void fill_cuda(T *const ptr, int_t n_elements, const T &value);

#define ZISA_INSTANTIATE_FILL_CUDA(TYPE)                                       \
  extern template void fill_cuda<TYPE>(TYPE *const, int_t, const TYPE &);

ZISA_INSTANTIATE_FILL_CUDA(float)
ZISA_INSTANTIATE_FILL_CUDA(double)

ZISA_INSTANTIATE_FILL_CUDA(bool)
ZISA_INSTANTIATE_FILL_CUDA(char)
ZISA_INSTANTIATE_FILL_CUDA(short)
ZISA_INSTANTIATE_FILL_CUDA(long)
ZISA_INSTANTIATE_FILL_CUDA(int)
ZISA_INSTANTIATE_FILL_CUDA(int_t)

#undef ZISA_INSTANTIATE_FILL_CUDA

}

#endif // ZISAMEMORYSUPERBUILD_FILL_CUDA_HPP
