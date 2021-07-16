# ZisaMemory
[![Build Status](https://github.com/1uc/ZisaMemory/actions/workflows/basic_integrity_checks.yml/badge.svg)](https://github.com/1uc/ZisaMemory/actions)
[![Docs Status](https://github.com/1uc/ZisaMemory/actions/workflows/publish_docs.yml/badge.svg)](https://1uc.github.io/ZisaMemory)

ZisaMemory most notably contains multidimensional arrays and array views. Along
with the wrappers for HDF5 and NetCDF required to write these arrays to disk.

## Installation

### System Dependencies
If you want to use ZisaMemory with

  * CUDA
  * MPI
  * HDF5
  * NetCDF

support, you must install these before hand. Note that all of them are optional
and are turned off by default.

### Dependencies
Please ensure you have `conan` in your `PATH`. The recommened way of installing
Conan is

    $ pip install --user conan

Now, the remaining dependencies of ZisaMemory can be installed by

    $ bin/install_dependencies.sh COMPILER DIRECTORY DEPENDENCY_FLAGS

where `COMPILER` is the name of the C compiler with which you want to compile
ZisaMemory, e.g. when using GNU GCC it's `gcc`. The dependencies will be
installed into subdirectories of `DIRECTORY`. We'll cover `DEPENDENCY_FLAGS`
soon.

The script performs two actions:

  1. Install external dependencies, see `conanfile.txt`.
  2. Download, compile and install other Zisa dependencies.

The second step requires knowledge of which dependencies you need. These are
passed through by the following flags

  * `--zisa_has_cuda={0,1}` to request CUDA.
  * `--zisa_has_mpi={0,1}` to request MPI.

### Compiling
The script for installing the dependencies generates part of the CMake command
required for compiling ZisaMemory. You'll need to add flags to control which
dependencies should be used:

  * `-DZISA_HAS_HDF5={0,1}` for HDF5 I/O.
  * `-DZISA_HAS_NETCDF={0,1}` for NetCDF I/O.

Further, you should choose a build type:

  * `Debug` CMake built in.
  * `FastDebug` optimized build with debug symbols and assertions.
  * `Release` CMake built in.

Running an unoptimized build can be very slow. Therefore, if you only need
meaningful backtraces and want the assertions, `FastDebug` is a good choice.
Now, for more specialized setting we provide:

  * `GPERFTOOLS` adds the flags required for `gperftools`, a sampling profiler.
  * `ASAN` adds the flags for Clangs AdressSanitizer.
  * `UBSAN` adds the flags for Clangs undefined behaviour sanitizer.

all three are based on an optimized build with debug symbols. Note that `ASAN`
and `UBSAN` expect the Clang compilers. The details can be found in `cmake/`.

#### CLI
Finally, you can use `-B BUILD_DIR` to generate the configuration in the folder
`BUILD_DIR`, and then compile with

    $ cmake --build BUILD_DIR --parallel=$(nproc)

#### IDEs
The procedure is to first install the dependencies, and then add the required
CMake flags in the appropriate place in the settings of your IDE. Once that's
done, you should be able to open the project as a CMake project.

### Installing
The library can be installed by a `make install`.


## Contributing
Please note that Zisa libraries are not stable and are distributed as source
code. The motivation is that I've written code that solves a reoccuring problem
in scientific computing. If this variant of the solution works for you, you're
most welcome to incorporate this code, or parts of it, into your project. If you
want to share bug fixes please use pull requests. Maybe this library almost
works for you and you need to perform destructive or otherwise unmergeable
changes, you can do this for example in your own fork.

In summary please treat this library as code which you can use and modify to
solve your problems.
