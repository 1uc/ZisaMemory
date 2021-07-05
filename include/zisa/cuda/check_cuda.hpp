// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef CHECK_CUDA_HPP_VQDUE
#define CHECK_CUDA_HPP_VQDUE

#include <zisa/config.hpp>

#include <cuda.h>

#if ZISA_HAS_CUDA
#define ZISA_CHECK_CUDA(status_code)                                           \
  LOG_ERR_IF(status_code != cudaSuccess, "CUDA reported an error.");
#endif

#endif
