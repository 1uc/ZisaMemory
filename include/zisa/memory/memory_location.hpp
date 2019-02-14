#ifndef MEMORY_LOCATION_H_W181H
#define MEMORY_LOCATION_H_W181H

#include <memory>

#include "zisa/memory/device_type.hpp"

namespace zisa {

template <class T>
device_type memory_location(const std::allocator<T> &) {
  return device_type::cpu;
}

} // zisa
#endif /* end of include guard */
