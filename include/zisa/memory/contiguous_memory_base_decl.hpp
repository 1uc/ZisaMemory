#ifndef CONTIGUOUS_MEMORY_BASE_H_N7OAI
#define CONTIGUOUS_MEMORY_BASE_H_N7OAI

#include <memory>

#include "zisa/config.hpp"
#include "zisa/memory/array_traits.hpp"
#include "zisa/memory/device_type.hpp"
#include "zisa/meta/void_t.hpp"

namespace zisa {

template <class T, class Allocator, class Equivalence, class Construction>
class contiguous_memory_base {
public:
  using const_pointer =
      typename std::allocator_traits<Allocator>::const_pointer;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using size_type = typename std::allocator_traits<Allocator>::size_type;

public:
  contiguous_memory_base()
      : _raw_data(nullptr), _n_elements(0), _allocator(nullptr) {}

  explicit contiguous_memory_base(size_type n_elements,
                                  const Allocator &allocator = Allocator());

  contiguous_memory_base(const contiguous_memory_base &other);
  contiguous_memory_base(contiguous_memory_base &&other) noexcept;

  ~contiguous_memory_base();

  contiguous_memory_base &operator=(const contiguous_memory_base &other);
  contiguous_memory_base &operator=(contiguous_memory_base &&other) noexcept;

  ANY_DEVICE_INLINE T &operator[](size_type i);
  ANY_DEVICE_INLINE const T &operator[](size_type i) const;

  ANY_DEVICE_INLINE pointer raw() { return _raw_data; }
  ANY_DEVICE_INLINE const_pointer raw() const { return _raw_data; }

  ANY_DEVICE_INLINE pointer begin() { return _raw_data; }
  ANY_DEVICE_INLINE const_pointer begin() const { return _raw_data; }
  ANY_DEVICE_INLINE const_pointer cbegin() const { return _raw_data; }

  ANY_DEVICE_INLINE pointer end() { return _raw_data + _n_elements; }
  ANY_DEVICE_INLINE const_pointer end() const {
    return _raw_data + _n_elements;
  }
  ANY_DEVICE_INLINE const_pointer cend() const {
    return _raw_data + _n_elements;
  }

  ANY_DEVICE_INLINE size_type size() const { return _n_elements; }

  inline device_type device() const;

private:
  void allocate(size_type n_elements, Allocator allocator);
  void allocate(size_type n_elements);

  void free();
  void free_data();
  void free_allocator();

  Allocator *allocator();
  Allocator const *allocator() const;

  void default_construct();
  void copy_construct(const contiguous_memory_base &other);

  bool resize(const size_type &n_elements);

private:
  pointer _raw_data;
  size_type _n_elements;

  Allocator *_allocator;
};

// clang-format off
template <class ...Args>
ANY_DEVICE_INLINE
typename contiguous_memory_base<Args...>::pointer
raw_ptr(contiguous_memory_base<Args...> &a) {
  return a.raw();
}
// clang-format on

// clang-format off
template <class ...Args>
ANY_DEVICE_INLINE
typename contiguous_memory_base<Args...>::const_pointer
raw_ptr(const contiguous_memory_base<Args...> &a) {
  return a.raw();
}
// clang-format on

template <class... Args>
device_type device(const contiguous_memory_base<Args...> &a) {
  return a.device();
}

} // namespace zisa

#endif /* end of include guard */
