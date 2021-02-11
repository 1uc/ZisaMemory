#include <algorithm>
#include <iostream>
#include <memory>

#include <zisa/memory/contiguous_memory.hpp>
#include <zisa/memory/host_contiguous_memory.hpp>
#include <zisa/testing/testing_framework.hpp>

template <class T>
zisa::contiguous_memory<bool> check_equal(const zisa::contiguous_memory<T> &a,
                                          const zisa::contiguous_memory<T> &b) {
  assert(a.size() == b.size());

  auto ac = zisa::contiguous_memory<T>(a.size(), zisa::device_type::cpu);
  auto bc = zisa::contiguous_memory<T>(b.size(), zisa::device_type::cpu);

  zisa::copy(ac, a);
  zisa::copy(bc, b);

  auto is_equal = zisa::contiguous_memory<bool>(a.size());

  for (zisa::int_t i = 0; i < a.size(); ++i) {
    is_equal[i] = (ac[i] == bc[i]);
  }

  return is_equal;
}

void check_copyless_const_ref(const zisa::contiguous_memory<double> &a,
                              void *ptr) {
  REQUIRE(a.raw() == ptr);
}

void check_copyless_ref(zisa::contiguous_memory<double> &a, void *ptr) {
  REQUIRE(a.raw() == ptr);
}

template <class SRC>
void check_copy_construction(const SRC &src) {
  auto cpy = src;

  REQUIRE(cpy.raw() != src.raw());
  REQUIRE(cpy.device() == src.device());

  auto is_equal = check_equal(cpy, src);
  auto is_good
      = std::all_of(is_equal.begin(), is_equal.end(), [](bool x) { return x; });

  REQUIRE(is_good);
}

template <class SRC>
void check_move_construction(SRC src) {
  auto ptr = src.raw();

  auto cpy = std::move(src);

  REQUIRE(cpy.raw() == ptr);
  REQUIRE(src.raw() == nullptr);
}

TEST_CASE("contiguous_memory") {
  zisa::int_t n_elements = 15;

  auto a = zisa::contiguous_memory<double>(n_elements);
  std::fill(a.begin(), a.end(), 0.0);

  SECTION("copy-less upcast -- ref") {
    SECTION("host") { check_copyless_ref(a, a.raw()); }
  }

  SECTION("copy-less upcast -- const ref") {
    SECTION("host") { check_copyless_const_ref(a, a.raw()); }
  }

  SECTION("copy construction") {
    SECTION("host") { check_copy_construction(a); }
  }

  SECTION("move construction") {
    SECTION("host") { check_move_construction(a); }
  }
}

TEST_CASE("contiguous_memory; initialization", "[memory]") {
  zisa::int_t n_elements = 15;
  zisa::int_t n_arrays = 5;

  auto elem = zisa::contiguous_memory<double>(n_elements);
  std::fill(elem.begin(), elem.end(), 42.0);

  auto seq = zisa::contiguous_memory<zisa::contiguous_memory<double>>(n_arrays);

  for (zisa::int_t i = 0; i < n_arrays; ++i) {
    REQUIRE(seq[i].raw() == nullptr);
  }

  for (zisa::int_t i = 0; i < n_arrays; ++i) {
    elem[0] = 42.0 + 0.1 * double(i);
    seq[i] = elem;
  }
  //
  //  for (zisa::int_t i = 0; i < n_arrays; ++i) {
  //    elem[0] = 42.0 + 0.1 * double(i);
  //    REQUIRE(std::equal(seq[i].begin(), seq[i].end(), elem.begin(),
  //    elem.end()));
  //  }
}

#if (ZISA_HAS_CUDA == 1)

namespace zisa {

static void check_allocation_cuda() {
  int_t n_elements = 10;
  auto d_mem = zisa::contiguous_memory<double>(n_elements, device_type::cuda);

  REQUIRE(d_mem.raw() != nullptr);
  REQUIRE(d_mem.device() == device_type::cuda);
}

template <class SRC>
static void check_assignment_location(SRC b, SRC a, device_type device) {
  b = a;
  REQUIRE(b.device() == device);
}

static void check_assignment() {
  int_t na = 10;
  int_t nb = 10;

  auto a_cpu = zisa::contiguous_memory<double>(na, device_type::cpu);
  auto a_cuda = zisa::contiguous_memory<double>(na, device_type::cuda);

  auto b_cpu = zisa::contiguous_memory<double>(nb, device_type::cpu);
  auto b_cuda = zisa::contiguous_memory<double>(nb, device_type::cuda);

  SECTION(" cpu <- cpu") {
    check_assignment_location(b_cpu, a_cpu, device_type::cpu);
  }
  SECTION(" cuda <- cpu") {
    check_assignment_location(b_cuda, a_cpu, device_type::cpu);
  }
  SECTION(" cpu <- cuda ") {
    check_assignment_location(b_cpu, a_cuda, device_type::cuda);
  }
  SECTION(" cuda <- cuda ") {
    check_assignment_location(b_cuda, a_cuda, device_type::cuda);
  }
}

}

TEST_CASE("contiguous_memory; on-device", "[cuda][memory]") {
  zisa::check_allocation_cuda();

  zisa::int_t n_elements = 10;
  auto device = zisa::device_type::cuda;
  auto a = zisa::contiguous_memory<double>(n_elements, device);

  SECTION("copy-less upcast -- ref") {
    SECTION("device") { check_copyless_ref(a, a.raw()); }
  }

  SECTION("copy-less upcast -- const ref") {
    SECTION("device") { check_copyless_const_ref(a, a.raw()); }
  }

  SECTION("copy construction") {
    SECTION("device") { check_copy_construction(a); }
  }

  SECTION("move construction") {
    SECTION("device") { check_move_construction(a); }
  }

  SECTION("assignment") {
    SECTION("device") { zisa::check_assignment(); }
  }
}

#endif
