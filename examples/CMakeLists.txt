add_executable(runfasteig_cpu runfasteig.cpp)
target_link_libraries(runfasteig_cpu PRIVATE fasteig)
target_compile_definitions(runfasteig_cpu PRIVATE FASTEIG_CPU_EXEC)

if(ENABLE_CUDA)
    add_executable(runfasteig_gpu runfasteig.cpp)
    target_link_libraries(runfasteig_gpu PRIVATE fasteig)
    target_compile_definitions(runfasteig_gpu PRIVATE FASTEIG_GPU_EXEC)
    set_target_properties(runfasteig_gpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

set_target_properties(runfasteig_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

