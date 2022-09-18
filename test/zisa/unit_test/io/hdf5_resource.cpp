#if ZISA_HAS_HDF5
#include <zisa/testing/testing_framework.hpp>

#include <zisa/io/hdf5_resource.hpp>

template <class Resource>
static void check_ownership(hid_t valid_hid) {
  int n_frees = 0;
  auto free_callback = [&n_frees](hid_t) { ++n_frees; };

  hid_t invalid_hid = -1;

  SECTION("Create & destroy") {
    { Resource(valid_hid, free_callback); }

    REQUIRE(n_frees == 1);
  }

  SECTION("Create invalid & not destroy") {
    { Resource(invalid_hid, free_callback); }

    REQUIRE(n_frees == 0);
  }

  SECTION("Move construct & destroy once.") {
    {
      // Avoid copy elision with explicit std::move.
      Resource(std::move(Resource(valid_hid, free_callback)));
    }

    REQUIRE(n_frees == 1);
  }

  SECTION("Create, move assign into invalid & destroy once.") {
    {
      auto place_holder = Resource(invalid_hid, [](hid_t) {});
      {
        // Avoid copy elision with explicit std::move.
        place_holder = std::move(Resource(valid_hid, free_callback));
      }

      REQUIRE(n_frees == 0);
      REQUIRE(*place_holder == valid_hid);
    }

    REQUIRE(n_frees == 1);
  }

  SECTION("Create, move assign into valid & destroy twice.") {
    {
      auto place_holder = Resource(valid_hid, free_callback);
      {
        // Avoid copy elision with explicit std::move.
        place_holder = std::move(Resource(valid_hid, free_callback));
      }

      REQUIRE(n_frees == 1);
      REQUIRE(*place_holder == valid_hid);
    }

    REQUIRE(n_frees == 2);
  }

  SECTION("Create, drop & don't destroy.") {
    {
      auto resource = Resource(valid_hid, free_callback);
      resource.drop_ownership();
    }
    REQUIRE(n_frees == 0);
  }
}

TEST_CASE("HDF5Resource; ownership", "[hdf5]") {
  hid_t valid_hid = zisa::H5T::copy(H5T_NATIVE_DOUBLE);
  check_ownership<zisa::HDF5Resource>(valid_hid);
  zisa::H5T::close(valid_hid);
}

TEST_CASE("HDF5Dataset; ownership", "[hdf5]") {
  hid_t valid_hid = zisa::H5T::copy(H5T_NATIVE_DOUBLE);
  check_ownership<zisa::HDF5Dataset>(valid_hid);
  zisa::H5T::close(valid_hid);
}

TEST_CASE("HDF5Dataspace; ownership", "[hdf5]") {
  hid_t valid_hid = zisa::H5T::copy(H5T_NATIVE_DOUBLE);
  check_ownership<zisa::HDF5Dataspace>(valid_hid);
  zisa::H5T::close(valid_hid);
}

TEST_CASE("HDF5Property; ownership", "[hdf5]") {
  hid_t valid_hid = zisa::H5T::copy(H5T_NATIVE_DOUBLE);
  check_ownership<zisa::HDF5Property>(valid_hid);
  zisa::H5T::close(valid_hid);
}

#endif