#include <zisa/memory/copy_bytes.hpp>

#include <cstring>

namespace zisa {

#if ZISA_HAS_CUDA == 0
void copy_bytes(void *dst,
                const device_type &dst_loc,
                void *src,
                const device_type &src_loc,
                std::size_t n_bytes) {
  if (src_loc == device_type::cpu && dst_loc == device_type::cpu) {
    memcpy(dst, src, n_bytes);
  } else {
    LOG_ERR("Unknown combination of `device_type`.");
  }
}
#endif

}
