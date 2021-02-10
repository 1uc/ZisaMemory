#ifndef CUDA_MEMORY_RESOURCE_HPP_VUEIW
#define CUDA_MEMORY_RESOURCE_HPP_VUEIW

#include <zisa/memory/memory_resource.hpp>

namespace zisa {

template <class T>
class cuda_memory_resource : public memory_resource<T> {
private:
  using super = memory_resource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

protected:
  virtual inline pointer do_allocate(size_type n) override {
    pointer ptr = nullptr;

    auto cudaError = cudaMalloc(&ptr, n * sizeof(T));
    LOG_ERR_IF(cudaError != cudaSuccess, cuda_error_message(cudaError));

    return ptr;
  }

  virtual inline void do_deallocate(pointer ptr, size_type) override {
    auto cudaError = cudaFree(ptr);
    LOG_ERR_IF(cudaError != cudaSuccess, cuda_error_message(cudaError));
  }

  virtual inline device_type do_device() const override {
    return device_type::cuda;
  }
};

template <class T>
device_type memory_location(const cuda_memory_resource<T> &resource) {
  return resource.device();
}

}

#endif
