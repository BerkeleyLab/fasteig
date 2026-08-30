file(GLOB FASTEIG_GPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

if(NOT FASTEIG_GPU_SOURCES)
    message(FATAL_ERROR "ENABLE_CUDA=ON but no sources found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# .cpp files here contain device code, so route them through nvcc.
# Renaming to .cu removes the need for this.
foreach(f IN LISTS FASTEIG_GPU_SOURCES)
    if(f MATCHES "\\.cpp$")
        set_source_files_properties("${f}" PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()

target_sources(fasteig PRIVATE ${FASTEIG_GPU_SOURCES})
