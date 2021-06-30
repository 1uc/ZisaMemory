#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/allocator.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/contiguous_memory.hpp>

namespace zisa {

void test_array_traits() {
  int_t n_elements = 10;

  using A = contiguous_memory<double>;
  typename std::enable_if<std::is_same<decltype(std::declval<A>().device()), device_type>::value, device_type>::type foo = device_type::unknown;
  REQUIRE(foo == device_type::unknown);


  // zisa::contiguous_memory
  auto alloc = allocator<double>(device_type::cpu);
  auto flat_memory = contiguous_memory<double>(n_elements, alloc);
  REQUIRE(array_traits<contiguous_memory<double>>::device(flat_memory)
          == device_type::cpu);

  // std::vector
  auto std_vector = std::vector<double>(n_elements);
  REQUIRE(array_traits<std::vector<double>>::device(std_vector)
          == device_type::unknown);
}

}

TEST_CASE("array_traits.device; SFINAE", "[array]") {
  zisa::test_array_traits();
}
