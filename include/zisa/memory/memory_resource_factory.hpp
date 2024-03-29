// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef MEMORY_RESOURCE_FACTORY_H_GYHK3
#define MEMORY_RESOURCE_FACTORY_H_GYHK3

#include <memory>

#include "zisa/memory/host_memory_resource.hpp"
#if ZISA_HAS_CUDA
#include "zisa/cuda/memory/cuda_memory_resource.hpp"
#endif
#include "zisa/memory/memory_resource.hpp"

namespace zisa {

template <class T>
std::shared_ptr<memory_resource<T>>
make_memory_resource(const device_type &device) {

  if (device == device_type::cpu) {
    return std::make_shared<host_memory_resource<T>>();
  }

  if (device == device_type::cuda) {
#if ZISA_HAS_CUDA
    return std::make_shared<cuda_memory_resource<T>>();
#else
    LOG_ERR("`device_type::cuda` requires `ZISA_HAS_CUDA == 1`.");
#endif
  }

  LOG_ERR("Implement the missing memory resource.");
}

} // namespace zisa

#endif /* end of include guard */
