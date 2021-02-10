if(ZISA_HAS_CUDA)

target_sources(memory_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_bytes.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fill_cuda.cu
)

endif()
