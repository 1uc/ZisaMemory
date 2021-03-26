if(ZISA_HAS_CUDA)

target_sources(memory_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/check_cuda.hpp
)

endif()
add_subdirectory(memory)
