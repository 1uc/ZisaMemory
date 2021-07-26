# ZisaMemory
[![Build Status](https://github.com/1uc/ZisaMemory/actions/workflows/basic_integrity_checks.yml/badge.svg?branch=main)](https://github.com/1uc/ZisaMemory/actions)
[![Docs Status](https://github.com/1uc/ZisaMemory/actions/workflows/publish_docs.yml/badge.svg?branch=main)](https://1uc.github.io/ZisaMemory)

ZisaMemory most notably contains multidimensional arrays and array views. Along
with the wrappers for HDF5 and NetCDF required to write these arrays to disk.

## Quickstart

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
installed into subdirectories of `DIRECTORY`. The available `DEPENDENCY_FLAGS`
are

  * `--zisa_has_cuda={0,1}` to request CUDA.
  * `--zisa_has_mpi={0,1}` to request MPI.

If this succeded the script will suggest part of the CMake command needed to
configure the build. Head over to the [project specific flags] for a list of
available flags.

[project specific flags]: https://1uc.github.io/ZisaMemory/md_building.html#cmake_flags
