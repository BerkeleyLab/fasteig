# ------------------------------------------------------------------------------
# fasteig library
#
# src/core is always compiled (shared infrastructure + CPU backend).
# src/gpu is added on top when ENABLE_CUDA=ON.
# ------------------------------------------------------------------------------
file(GLOB FASTEIG_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

add_library(fasteig STATIC ${FASTEIG_SOURCES})

add_subdirectory(core)

if(ENABLE_CUDA)
    add_subdirectory(gpu)
    message(STATUS "fasteig backends: src/core (CPU) + src/gpu (CUDA)")
else()
    message(STATUS "fasteig backends: src/core (CPU)")
endif()

# Internal include paths, visible to every translation unit in the target
# regardless of which subdirectory it came from.
target_include_directories(fasteig PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>)

if(ENABLE_CUDA)
    target_include_directories(fasteig PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>)
endif()

# Public API headers
target_include_directories(fasteig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

target_link_libraries(fasteig PUBLIC fasteig_math_deps)
