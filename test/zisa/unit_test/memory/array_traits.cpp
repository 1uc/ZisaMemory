// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/allocator.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/contiguous_memory.hpp>

namespace zisa {

void test_array_traits() {
  int_t n_elements = 10;

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
