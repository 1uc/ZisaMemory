#ifndef CONTIGUOUS_MEMORY_BASE_H_N7OAI
#define CONTIGUOUS_MEMORY_BASE_H_N7OAI

#include <memory>

#include "zisa/config.hpp"
#include "zisa/memory/array_traits.hpp"
#include "zisa/memory/device_type.hpp"
#include "zisa/meta/void_t.hpp"

namespace zisa {

template <class T, class Allocator>
class contiguous_memory_base {
public:
  using const_pointer =
      typename std::allocator_traits<Allocator>::const_pointer;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using size_type = typename std::allocator_traits<Allocator>::size_type;

public:
  contiguous_memory_base()
      : _raw_data(nullptr), n_elements(0), _allocator(nullptr) {}
  contiguous_memory_base(size_type n_elements,
                         const Allocator &allocator = Allocator());
  template <class A>
  contiguous_memory_base(const contiguous_memory_base<T, A> &other);
  contiguous_memory_base(const contiguous_memory_base &other);
  contiguous_memory_base(contiguous_memory_base &&other);

  ~contiguous_memory_base();

  inline contiguous_memory_base &operator=(const contiguous_memory_base &other);
  inline contiguous_memory_base &operator=(contiguous_memory_base &&other);

  template <class A>
  inline contiguous_memory_base<T, Allocator> &
  operator=(const contiguous_memory_base<T, A> &other);

  ANY_DEVICE_INLINE T &operator[](size_type i);
  ANY_DEVICE_INLINE const T &operator[](size_type i) const;

  ANY_DEVICE_INLINE pointer raw() { return _raw_data; }
  ANY_DEVICE_INLINE const_pointer raw() const { return _raw_data; }

  ANY_DEVICE_INLINE pointer begin() { return _raw_data; }
  ANY_DEVICE_INLINE const_pointer begin() const { return _raw_data; }

  ANY_DEVICE_INLINE pointer end() { return _raw_data + n_elements; }
  ANY_DEVICE_INLINE const_pointer end() const { return _raw_data + n_elements; }

  ANY_DEVICE_INLINE size_type size() const { return n_elements; }

  inline device_type device() const { return memory_location(*allocator()); }

private:
  void allocate(size_type n_elements, Allocator allocator);
  void allocate(size_type n_elements);

  void free();
  void free_data();
  void free_allocator();

  Allocator *allocator();
  Allocator const *allocator() const;

  template <class TT, class A1, class A2>
  friend void copy(contiguous_memory_base<TT, A1> &dst,
                   const contiguous_memory_base<TT, A2> &src);

  template <class TT, class A1, class A2>
  friend void copy_construct(contiguous_memory_base<TT, A1> &dst,
                             const contiguous_memory_base<TT, A2> &src);

  void default_construct();

  template <class A>
  bool resize(const contiguous_memory_base<T, A> &other);
  bool resize(const size_type &n_elements);

private:
  pointer _raw_data;
  size_type n_elements;

  Allocator *_allocator;
};

template <class T, class Allocator>
ANY_DEVICE_INLINE
    typename array_traits<contiguous_memory_base<T, Allocator>>::pointer
    raw_ptr(contiguous_memory_base<T, Allocator> &a) {
  return a.raw();
}

template <class T, class Allocator>
ANY_DEVICE_INLINE
    typename array_traits<contiguous_memory_base<T, Allocator>>::const_pointer
    raw_ptr(const contiguous_memory_base<T, Allocator> &a) {
  return a.raw();
}

template <class T, class Allocator>
device_type device(const contiguous_memory_base<T, Allocator> &a) {
  return a.device();
}

} // namespace zisa

#endif /* end of include guard */
