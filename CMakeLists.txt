cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(ZisaMemory VERSION 0.1)

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")
include("cmake/sanitizers.cmake")
include("cmake/gperftools.cmake")
include("cmake/clang_tidy.cmake")

# -- Dependencies --------------------------------------------------------------
add_library(memory_dependencies INTERFACE)

find_package(HDF5 REQUIRED)
target_include_directories(memory_dependencies INTERFACE ${HDF5_INCLUDE_DIRS})
target_link_libraries(memory_dependencies INTERFACE ${HDF5_LIBRARIES})

# experimental/filesystem
target_link_libraries(memory_dependencies INTERFACE stdc++fs)

# Logging
if(ZISA_LOG_LEVEL)
  target_compile_definitions(memory_dependencies INTERFACE ZISA_LOG_LEVEL=${ZISA_LOG_LEVEL})
endif()

# -- Zisa ----------------------------------------------------------------------
# -- core ----------------------------------------------------------------------
find_package(ZisaCore)

# -- memory --------------------------------------------------------------------
add_library(memory SHARED)
set_target_properties(memory PROPERTIES CXX_STANDARD 17)
target_include_directories(memory PUBLIC
  $<BUILD_INTERFACE:${ZisaMemory_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(memory PUBLIC memory_dependencies)
target_link_libraries(memory PUBLIC Zisa::core)
# target_link_libraries(memory PRIVATE memory_warning_flags)

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

install(
  FILES
  "cmake/ZisaMemoryConfig.cmake"
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
target_include_directories(memory_unit_tests PUBLIC test)
target_link_libraries(memory_unit_tests PUBLIC Zisa::memory Catch2::Catch2)
target_link_libraries(memory_unit_tests PUBLIC Catch2::Catch2)
target_link_libraries(memory_unit_tests PRIVATE memory_warning_flags)

enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)

# -- Subdirectories ------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(test)
