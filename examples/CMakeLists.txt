# List of example source files
set(EXAMPLES
    runfasteig
)

# Create an executable for each example
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)

    # Include directories
    target_include_directories(${EXAMPLE} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    # Link against fasteig (PUBLIC linkage means Eigen and math libs are inherited)
    target_link_libraries(${EXAMPLE} PRIVATE fasteig)

    # Set output directory for executables
    set_target_properties(${EXAMPLE} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )
endforeach()

