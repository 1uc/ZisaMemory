target_sources(memory_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_file.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_serial_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchcial_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchical_file.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchical_reader.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_file.cpp
)

if(ZISA_HAS_CUDA)

endif()
