#ifndef CONTIGUOUS_MEMORY_H_EIH5D
#define CONTIGUOUS_MEMORY_H_EIH5D

#include "zisa/memory/allocator.hpp"
#include "zisa/memory/contiguous_memory_base.hpp"

namespace zisa {

template <class T>
using contiguous_memory = contiguous_memory_base<T, allocator<T>>;

} // namespace zisa

#endif /* end of include guard */
