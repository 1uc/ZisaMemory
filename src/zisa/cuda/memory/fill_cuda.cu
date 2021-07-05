// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <zisa/cuda/memory/fill_cuda.hpp>

#include <zisa/cuda/memory/fill_cuda_impl.cuh>

namespace zisa {

#define ZISA_INSTANTIATE_FILL_CUDA(TYPE)                                       \
  template void fill_cuda<TYPE>(TYPE *const, int_t, const TYPE &);

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
