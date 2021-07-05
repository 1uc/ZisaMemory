// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#define CATCH_CONFIG_RUNNER
#include "zisa/testing/testing_framework.hpp"
#if ZISA_HAS_MPI
#include <mpi.h>
#endif

int main(int argc, char *argv[]) {
#if ZISA_HAS_MPI
  MPI_Init(&argc, &argv);
#endif
  int result = Catch::Session().run(argc, argv);
#if ZISA_HAS_MPI
  MPI_Finalize();
#endif
  return result;
}
