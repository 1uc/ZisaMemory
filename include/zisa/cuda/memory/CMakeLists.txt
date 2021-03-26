if(ZISA_HAS_CUDA)

target_sources(memory_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cuda_array.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cuda_memory_resource.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fill_cuda.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fill_cuda_impl.cuh
)

endif()
