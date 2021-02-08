#include <zisa/memory/copy_bytes.hpp>

namespace zisa {

#if ZISA_HAS_CUDA == 1
void copy_bytes(void * dst, const device_type &dst_loc, void * src, const device_type &src_loc, std::size_t n_bytes) {
  if(src_loc == device_type::cpu && dst_loc == device_type::cpu) {
    cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToHost);
  }
  else if(src_loc == device_type::cpu && dst_loc == device_type::cuda) {
    cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToDevice);
  }
  else if(src_loc == device_type::cuda && dst_loc == device_type::cpu) {
    cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToHost);
  }
  else if(src_loc == device_type::cuda && dst_loc == device_type::cuda) {
    cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToDevice);
  }
  else {
    LOG_ERR("Unknown combination of `device_type`.");
  }
}
#endif

}
