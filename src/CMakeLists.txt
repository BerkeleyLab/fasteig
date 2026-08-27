file(GLOB FASTEIG_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

add_library(fasteig STATIC ${FASTEIG_SOURCES})

# must come after add_library so the target exists
add_subdirectory(core)

# PUBLIC, not PRIVATE: consumers (examples, tests) need the headers too
target_include_directories(fasteig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(fasteig PUBLIC fasteig_math_deps)
