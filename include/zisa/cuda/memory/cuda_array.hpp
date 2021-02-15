#ifndef CUDA_ARRAY_HPP_QPEUJ
#define CUDA_ARRAY_HPP_QPEUJ

#include <zisa/memory/array.hpp>

namespace zisa {

template<class T, int n_dims>
array<T, n_dims> cuda_array(const shape_t<n_dims> &shape) {
  return array<T, n_dims>(shape, device_type::cuda);
}

}
#endif // CUDA_ARRAY_HPP
