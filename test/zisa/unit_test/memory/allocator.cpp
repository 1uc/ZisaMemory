#include <zisa/testing/testing_framework.hpp>

#include <memory>
#include <zisa/memory/allocator.hpp>
#include <zisa/memory/device_type.hpp>

TEST_CASE("allocator; memory_location", "[array][allocator]") {

  SECTION("host") {
    auto host_alloc = zisa::allocator<double>(zisa::device_type::cpu);
    REQUIRE(zisa::memory_location(host_alloc) == zisa::device_type::cpu);
  }

#if ZISA_HAS_CUDA == 1
  SECTION("cuda") {
    auto cuda_alloc = zisa::allocator<double>(zisa::device_type::cuda);
    REQUIRE(zisa::memory_location(cuda_alloc) == zisa::device_type::cuda);
  }
#endif
}