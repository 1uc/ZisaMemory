#!/usr/bin/env bash

if [[ "$#" -ne 2 ]]
then
    echo "Usage: $0 COMPILER DESTINATION"
fi

component_name="ZisaMemory"
zisa_dependencies=("ZisaCore")

zisa_memory_root=$(realpath $(dirname $(readlink -f $0))/..)

compiler=$1
compiler_id=$(basename ${compiler})
compiler_version=$($compiler -dumpversion)

install_root=$2
install_dir=${install_root}/${compiler_id}/${compiler_version}
source_dir=${install_root}/sources/${component_name}_dependencies

conan_file=${zisa_memory_root}/conanfile.txt

if [[ -f $conan_file ]]
then
   mkdir -p ${install_dir}/conan && cd ${install_dir}/conan
   conan install $conan_file -s compiler.libcxx=libstdc++11
fi

mkdir -p ${source_dir}

for dep in $zisa_dependencies
do
    src_dir=${source_dir}/$dep

    # git clone git@github.com/1uc/${dep}.git ${src_dir}
    # FIXME
    cp -r ${HOME}/git/$dep/ ${src_dir}

    mkdir -p ${src_dir}/build-dep
    cd ${src_dir}/build-dep

    cmake -DCMAKE_INSTALL_PREFIX=${install_dir}/zisa \
          -DCMAKE_PREFIX_PATH=${install_dir}/zisa/lib/cmake/zisa \
          -DCMAKE_PROJECT_${component_name}_INCLUDE=${install_dir}/conan/conan_paths.cmake \
          -DCMAKE_BUILD_TYPE=Release \
          ..

    cmake --build .
    cmake --install .
done
