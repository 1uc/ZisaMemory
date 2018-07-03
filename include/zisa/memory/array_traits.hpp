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

} // namespace zisa

#endif /* end of include guard */
