#ifndef ARRAY_H_TH7WE
#define ARRAY_H_TH7WE

#include "zisa/config.hpp"

#include "zisa/memory/array_base_decl.hpp"
#include "zisa/memory/column_major.hpp"
#include "zisa/memory/contiguous_memory.hpp"
#include "zisa/memory/row_major.hpp"
#include "zisa/memory/shape.hpp"

namespace zisa {

namespace detail {

template <class T, int n_dims, template <int N> class Indexing = column_major>
using array_super_
    = array_base<T,
                 Indexing<n_dims>,
                 contiguous_memory<T>,
                 shape_t<n_dims, typename contiguous_memory<T>::size_type>>;

} // namespace detail

template <class T, int n_dims, template <int N> class Indexing = column_major>
class array : public detail::array_super_<T, n_dims, Indexing> {
private:
  using super = detail::array_super_<T, n_dims, Indexing>;

protected:
  using shape_type = typename super::shape_type;

public:
  using super::super;

  array(const shape_type &shape, device_type device = device_type::cpu);

  using super::operator=;
};

template <class T, int n_dims, template <int N> class Indexing = column_major>
array<T, n_dims, Indexing> empty_like(const array<T, n_dims, Indexing> &other) {

  return array<T, n_dims, Indexing>(other.shape());
}

template <class T, int n_dims>
void save(const HDF5Writer &writer,
          const array<T, n_dims> &arr,
          const std::string &tag);

template <class T, int n_dims>
void load(const HDF5Reader &reader,
          array<T, n_dims> &arr,
          const std::string &tag);

} // namespace zisa

#endif /* end of include guard */
