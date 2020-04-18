#ifndef ZISA_ARRAY_VIEW_FWD_HPP_UEIWQ
#define ZISA_ARRAY_VIEW_FWD_HPP_UEIWQ

namespace zisa {

template <class T, int n_dims, template <int> class Indexing = row_major>
class array_const_view;

template <class T, int n_dims, template <int> class Indexing = row_major>
class array_view;

}
#endif // ZISA_ARRAY_VIEW_FWD_HPP
