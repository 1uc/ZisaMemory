#ifndef FILL_HPP_ZMXAL
#define FILL_HPP_ZMXAL

#include <zisa/config.hpp>

#if ZISA_HAS_CUDA == 1
#include <zisa/cuda/memory/fill_cuda.hpp>
#endif

namespace zisa {

template <class T>
void fill_host(T *const ptr, int_t n_elements, const T &value) {
  std::fill(ptr, ptr + n_elements, value);
}

template <class T>
void fill(T *const ptr, device_type device, int_t n_elements, const T &value) {
#if ZISA_HAS_CUDA == 0
  ZISA_UNUSED(device);
  zisa::fill_host(ptr, n_elements, value);
#else
  if (device == device_type::cpu) {
    zisa::fill_host(ptr, n_elements, value);
  } else if (device == device_type::cuda) {
    zisa::fill_cuda(ptr, n_elements, value);
  } else {
    LOG_ERR("Invalid `device`.");
  }
#endif
}

}

#endif // FILL_HPP
