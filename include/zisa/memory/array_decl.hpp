#ifndef ARRAY_H_TH7WE
#define ARRAY_H_TH7WE

#include <zisa/config.hpp>

#include <zisa/memory/array_base_decl.hpp>
#include <zisa/memory/column_major.hpp>
#include <zisa/memory/contiguous_memory.hpp>
#include <zisa/memory/row_major.hpp>
#include <zisa/memory/shape.hpp>

#include <zisa/memory/array_view_fwd.hpp>

namespace zisa {

namespace detail {

template <class T, int n_dims, template <int N> class Indexing = row_major>
using array_super_
    = array_base<T,
                 Indexing<n_dims>,
                 contiguous_memory<T>,
                 shape_t<n_dims, typename contiguous_memory<T>::size_type>>;

} // namespace detail

template <class T, int n_dims, template <int N> class Indexing = row_major>
class array : public detail::array_super_<T, n_dims, Indexing> {
private:
  using super = detail::array_super_<T, n_dims, Indexing>;

protected:
  using shape_type = typename super::shape_type;

public:
  using super::super;

  array(const array &) = default;
  array(array &&) noexcept = default;

  explicit array(const array_const_view<T, n_dims, Indexing> &other);
  explicit array(const array_view<T, n_dims, Indexing> &other);

  array(T *raw_ptr, const shape_type &shape);
  explicit array(const shape_type &shape,
                 device_type device = device_type::cpu);

  array &operator=(const array &) = default;
  array &operator=(array &&) noexcept = default;

  array &operator=(const array_const_view<T, n_dims, Indexing> &);
  array &operator=(const array_view<T, n_dims, Indexing> &);

  [[nodiscard]] static array<T, n_dims, row_major> load(HDF5Reader &reader,
                                                        const std::string &tag);
};

template <class T, int n_dims, template <int N> class Indexing = column_major>
array<T, n_dims, Indexing> empty_like(const array<T, n_dims, Indexing> &other) {

  return array<T, n_dims, Indexing>(other.shape());
}

template <class T, int n_dims, template <int N> class Indexing>
void save(HDF5Writer &writer,
          const array<T, n_dims, Indexing> &arr,
          const std::string &tag);

} // namespace zisa
#endif /* end of include guard */
