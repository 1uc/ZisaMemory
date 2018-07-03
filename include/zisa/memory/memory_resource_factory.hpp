#ifndef MEMORY_RESOURCE_FACTORY_H_GYHK3
#define MEMORY_RESOURCE_FACTORY_H_GYHK3

#include <memory>

#include "zisa/memory/host_memory_resource.hpp"
#include "zisa/memory/memory_resource.hpp"

namespace zisa {

template <class T>
std::shared_ptr<memory_resource<T>>
make_memory_resource(const device_type &device) {

  if (device == device_type::cpu) {
    return std::make_shared<host_memory_resource<T>>();
  }

  return nullptr;
}

} // namespace zisa

#endif /* end of include guard */
