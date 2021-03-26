target_sources(memory_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_file.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_serial_writer.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_writer.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hdf5_writer_fwd.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchical_file.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchical_reader.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hierarchical_writer.hpp
)

if(ZISA_HAS_CUDA)

endif()
