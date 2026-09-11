add_executable(runfasteig runfasteig.cpp)
target_link_libraries(runfasteig PRIVATE fasteig)

if(ENABLE_CUDA)
    target_compile_definitions(runfasteig PRIVATE FASTEIG_GPU_EXEC)
endif()

set_target_properties(runfasteig PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

