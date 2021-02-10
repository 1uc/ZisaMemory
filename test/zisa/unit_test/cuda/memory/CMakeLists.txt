target_sources(memory_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_view.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_bytes.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fill_cuda.cu
)

