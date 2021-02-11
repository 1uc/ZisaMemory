#ifndef STD_ALLOCATOR_EQUIVALENCE_HPP_OQXUN
#define STD_ALLOCATOR_EQUIVALENCE_HPP_OQXUN

#include <memory>

namespace zisa {

template<class T>
struct STDAllocatorEquivalence {
  bool are_equivalent(std::allocator<T> &, std::allocator<T> &) {
    return true;
  }
};

}

#endif // STD_ALLOCATOR_EQUIVALENCE_HPP
