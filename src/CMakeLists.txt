# Collect all source files in src directory
file(GLOB FASTEIG_SOURCES "*.cpp" "*.c")

# Create static library
add_library(fasteig STATIC ${FASTEIG_SOURCES})

# Specify include directories for this target
target_include_directories(fasteig PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link math libraries and Eigen via bundled interface target
target_link_libraries(fasteig PUBLIC fasteig_math_deps)
