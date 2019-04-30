/* Array traits.
 *
 */

#ifndef ARRAY_TRAITS_H_8G39N
#define ARRAY_TRAITS_H_8G39N

namespace zisa {

template <class Array>
struct array_traits {
  using pointer = typename Array::pointer;
  using const_pointer = typename Array::const_pointer;
  using size_type = typename Array::size_type;
};

struct default_dispatch_tag {};
struct split_array_dispatch_tag {};
struct bool_dispatch_tag {};

template <class T>
struct array_save_traits {
  using dispatch_tag = default_dispatch_tag;
  using scalar_type = T;
};

template <>
struct array_save_traits<bool> {
  using dispatch_tag = bool_dispatch_tag;
  using scalar_type = char;
};

} // namespace zisa

#endif /* end of include guard */
