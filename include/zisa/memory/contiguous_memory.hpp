// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef CONTIGUOUS_MEMORY_H_EIH5D
#define CONTIGUOUS_MEMORY_H_EIH5D

#include "zisa/memory/allocator.hpp"
#include "zisa/memory/contiguous_memory_base.hpp"
#include "zisa/memory/scientific_constructor.hpp"

namespace zisa {

template <class T>
using contiguous_memory = contiguous_memory_base<T,
                                                 allocator<T>,
                                                 AllocatorEquivalence<T>,
                                                 ScientificConstructor<T>>;

} // namespace zisa

#endif /* end of include guard */
