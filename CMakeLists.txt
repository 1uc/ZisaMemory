if(ZISA_HAS_CUDA)
  # The CUDA related stuff was *massively* improved in
  # 3.17 and 3.18. Therefore, if CUDA is requested, we
  # absolutely need the modern versions.
  cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
else()
  # If we're not building with CUDA something old suffices.
  cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
endif()

project(ZisaMemory VERSION 0.1 LANGUAGES C CXX)
if(ZISA_HAS_CUDA)
  enable_language(CUDA)
endif()

# Enables local find modules, e.g. for NetCDF.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")
include("cmake/sanitizers.cmake")
include("cmake/gperftools.cmake")
include("cmake/clang_tidy.cmake")

# -- Dependencies --------------------------------------------------------------
add_library(memory_dependencies INTERFACE)

# experimental/filesystem
target_link_libraries(memory_dependencies INTERFACE stdc++fs)

# Logging
if(ZISA_LOG_LEVEL)
  target_compile_definitions(memory_dependencies INTERFACE ZISA_LOG_LEVEL=${ZISA_LOG_LEVEL})
endif()

if(ZISA_HAS_HDF5)
  find_package(HDF5 REQUIRED)
  target_include_directories(memory_dependencies INTERFACE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(memory_dependencies INTERFACE ${HDF5_LIBRARIES})
  target_compile_definitions(memory_dependencies INTERFACE ZISA_HAS_HDF5=1)
endif()

if(ZISA_HAS_NETCDF)
  find_package(NetCDF REQUIRED)
  target_link_libraries(memory_dependencies INTERFACE NetCDF::NetCDF)
  target_compile_definitions(memory_dependencies INTERFACE ZISA_HAS_NETCDF=1)
endif()


# -- Zisa ----------------------------------------------------------------------
# -- core ----------------------------------------------------------------------
find_package(ZisaCore)

# -- memory --------------------------------------------------------------------
add_library(memory_generic_obj OBJECT)
set_target_properties(memory_generic_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(memory_generic_obj PROPERTIES CXX_STANDARD 17)
target_include_directories(memory_generic_obj PUBLIC
  $<BUILD_INTERFACE:${ZisaMemory_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
target_link_libraries(memory_generic_obj PUBLIC memory_dependencies)
target_link_libraries(memory_generic_obj PUBLIC Zisa::core)
target_link_libraries(memory_generic_obj PRIVATE memory_warning_flags)

if(ZISA_HAS_CUDA)
  add_library(memory_cuda_obj OBJECT)
  set_target_properties(memory_cuda_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(memory_cuda_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(memory_cuda_obj PROPERTIES CUDA_STANDARD 11)
  set_target_properties(memory_cuda_obj PROPERTIES CXX_STANDARD 17)
  target_include_directories(memory_cuda_obj PUBLIC
    $<BUILD_INTERFACE:${ZisaMemory_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
  target_link_libraries(memory_cuda_obj PUBLIC memory_dependencies)
  target_link_libraries(memory_cuda_obj PUBLIC Zisa::core)
  target_link_libraries(memory_cuda_obj PRIVATE memory_warning_flags)
endif()

add_library(memory STATIC)
set_target_properties(memory PROPERTIES OUTPUT_NAME zisa_memory)
target_sources(memory PRIVATE $<TARGET_OBJECTS:memory_generic_obj>)

if(ZISA_HAS_CUDA)
  target_sources(memory PRIVATE $<TARGET_OBJECTS:memory_cuda_obj>)
  set_target_properties(memory PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_include_directories(memory PUBLIC
  $<BUILD_INTERFACE:${ZisaMemory_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(memory PUBLIC memory_dependencies)
target_link_libraries(memory PUBLIC Zisa::core)
target_link_libraries(memory PRIVATE memory_warning_flags)

# -- clang tidy ----------------------------------------------------------------
if(CLANG_TIDY_EXE)
  set_target_properties(memory PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif()

# -- install -------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMemoryConfigVersion.cmake"
  VERSION ${ZisaMemory_VERSION}
  COMPATIBILITY SameMajorVersion
  )

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/cmake/ZisaMemoryConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMemoryConfig.cmake"
  @ONLY
)

install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMemoryConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMemoryConfigVersion.cmake"
  DESTINATION lib/cmake/zisa
  )

add_library(Zisa::memory ALIAS memory)
install(
  TARGETS memory
          memory_dependencies
          memory_warning_flags
  EXPORT ZisaMemoryTargets
  )
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ZisaMemoryTargets
  FILE ZisaMemoryTargets.cmake
  NAMESPACE Zisa::
  DESTINATION lib/cmake/zisa
  )

# -- Tests ---------------------------------------------------------------------
find_package(Catch2 REQUIRED)

add_executable(memory_unit_tests "")
set_target_properties(memory_unit_tests PROPERTIES OUTPUT_NAME unit_tests)
set_target_properties(memory_unit_tests PROPERTIES CXX_STANDARD 17)
if(ZISA_HAS_CUDA)
  set_target_properties(memory_unit_tests PROPERTIES CUDA_STANDARD 11)
endif()
target_include_directories(memory_unit_tests PUBLIC test)
target_link_libraries(memory_unit_tests PUBLIC Zisa::memory Catch2::Catch2)
target_link_libraries(memory_unit_tests PUBLIC Catch2::Catch2)
target_link_libraries(memory_unit_tests PRIVATE memory_warning_flags)

enable_testing()
add_test(NAME ZisaMemoryUnitTests COMMAND unit_tests)

# -- Subdirectories ------------------------------------------------------------
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
