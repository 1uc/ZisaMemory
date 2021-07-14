# ZisaMemory                                                         {#mainpage}
ZisaMemory is provides multidimensional arrays and array views. Along with the
code required to write arrays to HDF5 or NetCDF files.

## Quickstart
Start by cloning the repository

    $ git clone https://github.com/1uc/ZisaMemory.git

and change into the newly created directory. Then proceed to install the
dependencies:

    $ bin/install_dir.sh COMPILER DIRECTORY       \
                         [--zisa_has_mpi={0,1}]   \
                         [--zisa_has_cuda={0,1}]

they will be placed into a subdirectory of `DIRECTORY` and print
part of the CMake command needed to include the dependencies. `COMPILER` must
be replaced with the compiler you want to use.

If this worked continue by adding [project specific flags] to the CMake
command. If not it's not going to be a quick start. Please read
[Dependencies] and then [Building].

[project specific flags]: @ref cmake_flags
[Dependencies]: dependencies.md
[Building]: cmake.md
