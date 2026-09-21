# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

# ------------------------------------------------------------------------------
# Launch profiles and timeouts
# ------------------------------------------------------------------------------
set(FASTEIG_TEST_PROFILE "auto" CACHE STRING "Test launch profile: auto, local, perlmutter")
set_property(CACHE FASTEIG_TEST_PROFILE PROPERTY STRINGS auto local perlmutter)
set(FASTEIG_TEST_LAUNCHER "auto" CACHE STRING "Test launcher: auto, direct, srun")
set_property(CACHE FASTEIG_TEST_LAUNCHER PROPERTY STRINGS auto direct srun)
if(NOT FASTEIG_TEST_PROFILE MATCHES "^(auto|local|perlmutter)$" OR
   NOT FASTEIG_TEST_LAUNCHER MATCHES "^(auto|direct|srun)$")
    message(FATAL_ERROR "Invalid FASTEIG_TEST_PROFILE or FASTEIG_TEST_LAUNCHER")
endif()
set(FASTEIG_TEST_TIMEOUT 3600 CACHE STRING "Numerical test timeout in seconds")
if(NOT FASTEIG_TEST_TIMEOUT MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "FASTEIG_TEST_TIMEOUT must be a positive integer")
endif()
math(EXPR harness_timeout "${FASTEIG_TEST_TIMEOUT} + 120")

add_executable(fasteig_version_api_test version_api.cpp)
target_link_libraries(fasteig_version_api_test PRIVATE fasteig)

# ------------------------------------------------------------------------------
# Build guard - ensures the executables are compiled before any test runs
# ------------------------------------------------------------------------------
add_test(NAME build_fasteig COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG>)
set_tests_properties(build_fasteig PROPERTIES FIXTURES_SETUP fasteig_built TIMEOUT 600)

# ------------------------------------------------------------------------------
# Version/build-provenance API (xSDK M8)
# ------------------------------------------------------------------------------
add_test(NAME version_api COMMAND fasteig_version_api_test)
set_tests_properties(version_api PROPERTIES FIXTURES_REQUIRED fasteig_built LABELS version)

# ------------------------------------------------------------------------------
# Test scripts and shared solver list
# ------------------------------------------------------------------------------
set(test_scripts "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${test_scripts}/run_case.py" "${test_scripts}/run_suite.py")
execute_process(COMMAND ${Python3_EXECUTABLE} -B "${test_scripts}/run_suite.py" --list-solvers
    OUTPUT_VARIABLE solver_manifest RESULT_VARIABLE manifest_status OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT manifest_status EQUAL 0 OR solver_manifest STREQUAL "")
    message(FATAL_ERROR "Unable to read the default reference-test solver list")
endif()
string(REPLACE "\n" ";" solvers "${solver_manifest}")

# ------------------------------------------------------------------------------
# CPU/GPU tests - register only for executables built by this configuration
# ------------------------------------------------------------------------------
foreach(backend cpu gpu)
    if(NOT TARGET runfasteig_${backend})
        continue()
    endif()

    # --------------------------------------------------------------------------
    # Version/build-provenance CLI (xSDK M8)
    # --------------------------------------------------------------------------
    add_test(NAME version_cli_${backend} COMMAND runfasteig_${backend} --version)
    set_tests_properties(version_cli_${backend} PROPERTIES
        FIXTURES_REQUIRED fasteig_built PASS_REGULAR_EXPRESSION "FASTEIG 1\\.0\\.0" LABELS "version;${backend}")
    add_test(NAME build_info_cli_${backend} COMMAND runfasteig_${backend} --build-info)
    set_tests_properties(build_info_cli_${backend} PROPERTIES
        FIXTURES_REQUIRED fasteig_built PASS_REGULAR_EXPRESSION "Build System[ ]*: CMake" LABELS "version;${backend}")

    # --------------------------------------------------------------------------
    # Reference tests - SiO/SR, single solver, single case
    # --------------------------------------------------------------------------
    foreach(solver IN LISTS solvers)
        set(test_name "reference_${backend}_SiO_${solver}_SR")
        add_test(NAME ${test_name} COMMAND ${Python3_EXECUTABLE} -B "${test_scripts}/run_case.py"
            --solver ${solver} --which SR --matrix SiO
            --runfasteig $<TARGET_FILE:runfasteig_${backend}>
            --profile ${FASTEIG_TEST_PROFILE} --launcher ${FASTEIG_TEST_LAUNCHER}
            --timeout ${FASTEIG_TEST_TIMEOUT} --allow-unavailable
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(${test_name} PROPERTIES
            FIXTURES_REQUIRED fasteig_built TIMEOUT ${harness_timeout}
            SKIP_RETURN_CODE 77 RESOURCE_LOCK fasteig_compute
            LABELS "reference;${backend};standard;SiO;${solver};SR")
    endforeach()
endforeach()

