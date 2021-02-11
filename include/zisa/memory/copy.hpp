#ifndef COPY_HPP
#define COPY_HPP

#include <zisa/config.hpp>
#include <zisa/memory/copy_bytes.hpp>
#include <zisa/memory/device_type.hpp>

namespace zisa {
namespace internal {

template <class T>
void copy(T *const dst,
          device_type dst_loc,
          T const *const src,
          device_type src_loc,
          int_t size) {

  LOG_ERR_IF(dst_loc == device_type::unknown,
             "Can't handle `dst_loc == unknown`.");

  LOG_ERR_IF(src_loc == device_type::unknown,
             "Can't handle `src_loc == unknown`.");

  if (dst_loc == device_type::cpu && src_loc == device_type::cpu) {
    std::copy(src, src + size, dst);
  }

#if ZISA_HAS_CUDA == 1
  else {
    LOG_ERR_IF(!std::is_trivially_copyable<T>::value,
               "Can't safely copy the bytes.");

    zisa::copy_bytes(
        (void *)dst, dst_loc, (void *)src, src_loc, size * sizeof(T));
  }
#endif
}

}
}

#endif // COPY_HPP
