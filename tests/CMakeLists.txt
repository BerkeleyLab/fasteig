# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path to the built executable
set(FASTEIG_EXE "${CMAKE_BINARY_DIR}/examples/runfasteig")

# Path to test scripts
set(TEST_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/tests/scripts")

# Path to matrices
set(MATRIX_DIR "${CMAKE_SOURCE_DIR}/examples/Sparse_Matrices")

# Default matrix
set(TEST_MATRIX "SiO" CACHE STRING "Matrix to use for tests (stem or full path)")

# ------------------------------------------------------------------------------
# Build guard - ensures the executable is compiled before any test runs
# ------------------------------------------------------------------------------
add_test(
    NAME "build_fasteig"
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target runfasteig
)
set_tests_properties("build_fasteig" PROPERTIES
    FIXTURES_SETUP  fasteig_built
    TIMEOUT         600
)

# ------------------------------------------------------------------------------
# Smoke tests - single solver, single case
# ------------------------------------------------------------------------------
set(SOLVERS Lanczos BlockLanczos KrylovSchur LOBPCG Davidson Chebyshev)
set(WHICH_OPTS SR LR)

foreach(SOLVER ${SOLVERS})
    foreach(WHICH ${WHICH_OPTS})
        add_test(
            NAME "smoke_${SOLVER}_${WHICH}"
            COMMAND ${Python3_EXECUTABLE}
                ${TEST_SCRIPTS_DIR}/run_case.py
                --solver      ${SOLVER}
                --which       ${WHICH}
                --matrix      ${TEST_MATRIX}
                --runfasteig  ${FASTEIG_EXE}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties("smoke_${SOLVER}_${WHICH}" PROPERTIES
            TIMEOUT           300
            LABELS            "smoke"
            FIXTURES_REQUIRED fasteig_built
        )
    endforeach()
endforeach()

if(NOT TARGET runfasteig)
    message(FATAL_ERROR "BUILD_TESTS requires BUILD_EXAMPLES=ON (runfasteig is the test driver).")
endif()
