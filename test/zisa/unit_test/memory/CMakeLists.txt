target_sources(memory_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/allocator.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_traits.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/block_allocator.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory_base.cpp
)

