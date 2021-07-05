// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <iostream>
#include <memory>
#include <zisa/testing/testing_framework.hpp>

#include <zisa/memory/contiguous_memory_base.hpp>
#include <zisa/memory/scientific_constructor.hpp>
#include <zisa/memory/std_allocator_equivalence.hpp>

TEST_CASE("array; STL allocator") {
  zisa::int_t n_elements = 15;

  using cmb
      = zisa::contiguous_memory_base<double,
                                     std::allocator<double>,
                                     zisa::STDAllocatorEquivalence<double>,
                                     zisa::ScientificConstructor<double>>;

  auto alloc = std::allocator<double>();
  auto a = cmb(n_elements, alloc);
  auto b = cmb(n_elements, alloc);

  a[2] = 42;

  SECTION("move constructor") {
    auto ptr = a.raw();

    b = std::move(a);

    REQUIRE(b.raw() == ptr);
    REQUIRE(a.raw() == nullptr);
  }
}
