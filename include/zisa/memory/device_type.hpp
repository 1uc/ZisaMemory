// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef DEVICE_TYPE_H_SK3UI
#define DEVICE_TYPE_H_SK3UI

namespace zisa {

enum class device_type {
  cpu,    // CPU
  cuda,   // NVIDIA GPU through CUDA
  unknown // If it's unclear where the memory resides.
};

} // namespace zisa

#endif /* end of include guard */
