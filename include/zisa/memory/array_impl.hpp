/*
 *
 */

#ifndef ARRAY_IMPL_H_1WC9M
#define ARRAY_IMPL_H_1WC9M

#include "zisa/memory/array_base.hpp"
#include "zisa/memory/array_decl.hpp"

namespace zisa {

template <class T, int n_dims, template <int N> class Indexing>
array<T, n_dims, Indexing>::array(const shape_type &shape, device_type device)
    : super(shape, contiguous_memory<T>(product(shape), device)) {}

} // namespace zisa

#endif /* end of include guard */
