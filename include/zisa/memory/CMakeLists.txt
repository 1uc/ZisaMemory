target_sources(memory_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/allocator.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_base.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_base_decl.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_base_fwd.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_cell_flags.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_decl.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_impl.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_stencil_family.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_traits.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_view.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/array_view_fwd.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/block_allocator.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/column_major.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory_base.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory_base_decl.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/contiguous_memory_base_impl.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_bytes.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/device_type.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fill.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/host_memory_resource.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/memory_location.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/memory_resource.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/memory_resource_factory.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/row_major.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/scientific_constructor.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/shape.hpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/std_allocator_equivalence.hpp
)

if(ZISA_HAS_CUDA)

endif()
