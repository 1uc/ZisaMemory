// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#ifndef SCIENTIFIC_CONSTRUCTOR_HPP
#define SCIENTIFIC_CONSTRUCTOR_HPP

#include <zisa/config.hpp>
#include <zisa/memory/allocator.hpp>

namespace zisa {

namespace internal {
template <class T, class SFINAE = void>
struct ScientificConstructorImpl {
  static void default_construct(T *, int_t, const allocator<T> &) {
    // Do nothing.
  }

  static void copy_construct(T *const dst,
                             T const *const src,
                             int_t size,
                             const allocator<T> &alloc) {
    auto device = alloc.device();
    zisa::copy_bytes(
        (void *)dst, device, (void *)src, device, size * sizeof(T));
  }
};

template <class T>
struct ScientificConstructorImpl<
    T,
    typename std::enable_if<!std::is_trivially_copyable<T>::value>::type> {

  static void
  default_construct(T *const ptr, int_t size, const allocator<T> &alloc) {
    LOG_ERR_IF(alloc.device() != device_type::cpu,
               "Can't default construct this type.");

    for (int_t i = 0; i < size; ++i) {
      alloc->construct(ptr + i);
    }
  }

  static void copy_construct(T *const dst,
                             T const *const src,
                             int_t size,
                             const allocator<T> &alloc) {

    LOG_ERR_IF(alloc.device() != device_type::cpu,
               "Can't default construct this type.");

    for (int_t i = 0; i < size; ++i) {
      new (dst + i) T(src[i]);
    }
  }
};

}

/// Construction policy for `contiguous_memory_base`.
/** This policy implements the common memory allocation model in scientific
 * codes. In it's most naive form it's simply: allocate & maybe copy the bytes.
 * Clearly doesn't work well when allocating an array of arrays or an array of
 * smart pointers, since they need certain invariants to hold otherwise, they
 * free memory that was never allocated.
 *
 * For scientific codes, there's no need to default initialize an array of
 * doubles, and properly initializing a array of arrays on a GPU is done
 * manually.
 *
 * The rules for default construction are:
 *   - For trivially copyable types, only allocate the memory, i.e.
 *     the memory comes 'empty'.
 *
 *   - For all other types, for memory allocated on the CPU,
 *     perform proper default initialization.
 *
 *  The rules for copy construction are:
 *   - For trivially copyable types, copy the bytes.
 *
 *   - For all other types, for memory allocated on the CPU,
 *     perform proper copy construction.
 *
 *  Missing cases:
 *   - Non trivially copyable types not allocated on the CPU simply
 *     aren't handled.
 */
template <class T>
struct ScientificConstructor {
  static void
  default_construct(T *const ptr, int_t size, const allocator<T> &alloc) {
    zisa::internal::ScientificConstructorImpl<T>::default_construct(
        ptr, size, alloc);
  }

  static void copy_construct(T *const dst,
                             T const *const src,
                             int_t size,
                             const allocator<T> &alloc) {
    zisa::internal::ScientificConstructorImpl<T>::copy_construct(
        dst, src, size, alloc);
  }
};

}

#endif // SCIENTIFIC_CONSTRUCTOR_HPP
