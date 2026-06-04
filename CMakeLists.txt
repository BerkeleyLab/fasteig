cmake_minimum_required(VERSION 3.20)
project(fasteig
    VERSION 1.0.0
    LANGUAGES CXX C
    DESCRIPTION "Fast eigenvalue solver library"
)

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "CMAKE_BUILD_TYPE must be 'Release' or 'Debug'. Got: '${CMAKE_BUILD_TYPE}'")
endif()

include(FetchContent)

set(PLAT "Generic" CACHE STRING "Platform name (e.g., Perlmutter, Frontier)")

# ------------------------------------------------------------------------------
# Compiler detection and flags
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
        $<$<CONFIG:Debug>:-Wpedantic>
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address>)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
        $<$<CONFIG:Debug>:-Wpedantic>
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address>)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(STATUS "Compiler: Intel LLVM (icpx) ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Release>:-fp-model=fast>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address>)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(STATUS "Compiler: Intel Classic (icc/icpc) ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Release>:-fp-model=fast>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
    )

else()
    message(WARNING "Untested compiler: ${CMAKE_CXX_COMPILER_ID}. Using default flags.")
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
    )
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(USE_MKL        "Use Intel MKL for BLAS/LAPACK"    OFF)
option(USE_LIBSCI     "Use Cray LibSci for BLAS/LAPACK"  OFF)
option(BUILD_EXAMPLES "Build example programs"            ON)
option(BUILD_TESTS    "Build and enable tests"           OFF)

if(USE_MKL AND USE_LIBSCI)
    message(FATAL_ERROR "USE_MKL and USE_LIBSCI are mutually exclusive. Choose one.")
endif()

# ------------------------------------------------------------------------------
# Eigen
# ------------------------------------------------------------------------------
set(EIGEN_DIR "" CACHE PATH "Path to existing Eigen3 installation")

if(EIGEN_DIR)
    if(EXISTS "${EIGEN_DIR}/Eigen/Core")
        message(STATUS "Using Eigen from: ${EIGEN_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_DIR}")
    elseif(EXISTS "${EIGEN_DIR}/eigen3/Eigen/Core")
        message(STATUS "Using Eigen from: ${EIGEN_DIR}/eigen3")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_DIR}/eigen3")
    else()
        message(FATAL_ERROR "EIGEN_DIR set to '${EIGEN_DIR}' but Eigen/Core not found there.")
    endif()
elseif(EXISTS "${CMAKE_SOURCE_DIR}/eigen/Eigen/Core")
    message(STATUS "Using previously cloned Eigen from: ${CMAKE_SOURCE_DIR}/eigen")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/eigen")
else()
    message(STATUS "EIGEN_DIR not set, cloning Eigen into ${CMAKE_SOURCE_DIR}/eigen ...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        SOURCE_DIR     "${CMAKE_SOURCE_DIR}/eigen"
    )
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/eigen")
    endif()
    message(STATUS "Eigen available at: ${CMAKE_SOURCE_DIR}/eigen")
endif()

# ------------------------------------------------------------------------------
# BLAS and LAPACK
# ------------------------------------------------------------------------------
if(USE_MKL)

    # Resolve MKLROOT from argument or environment
    if(NOT MKLROOT)
        if(DEFINED ENV{MKLROOT})
            set(MKLROOT "$ENV{MKLROOT}")
        else()
            message(FATAL_ERROR
                "USE_MKL=ON but MKLROOT is not set. "
                "Pass -DMKLROOT=/path/to/mkl or source the MKL setvars.sh script.")
        endif()
    endif()

    message(STATUS "Using MKL from: ${MKLROOT}")

    # Try CMake config first (MKL >= 2021 ships one)
    list(APPEND CMAKE_PREFIX_PATH "${MKLROOT}/lib/cmake/mkl")

    # LP64 is required: Eigen's Pardiso wrapper uses int throughout.
    # ILP64 would change MKL_INT to long long and break the pardiso() signature.
    # Do NOT add MKL_ILP64 to compile definitions.
    set(MKL_INTERFACE "lp64"       CACHE STRING "MKL integer interface" FORCE)
    set(MKL_THREADING "gnu_thread" CACHE STRING "MKL threading layer"   FORCE)

    set(MATH_COMPILE_DEFINITIONS
        EIGEN_USE_MKL_ALL
        USE_MKL
    )

    find_package(MKL QUIET CONFIG)
    if(MKL_FOUND)
        message(STATUS "Found MKL via CMake config")
        message(STATUS "  MKL root:     ${MKL_ROOT}")
        message(STATUS "  Interface:    ${MKL_INTERFACE}")
        message(STATUS "  Threading:    ${MKL_THREADING}")
        set(MATH_LINK_LIBRARIES MKL::MKL)
    else()
        # Manual configuration mirroring the known-good link line:
        #   -L${MKLROOT}/lib/intel64 -Wl,-rpath ${MKLROOT}/lib/intel64
        #   -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
        #   -lgomp -lpthread -lm -ldl
        message(STATUS "MKL CMake config not found, using manual link configuration (LP64 + GNU thread)")

        add_library(mkl_imported INTERFACE IMPORTED)
        set_target_properties(mkl_imported PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKLROOT}/include"
            INTERFACE_LINK_LIBRARIES
                "-L${MKLROOT}/lib/intel64;\
-Wl,-rpath,${MKLROOT}/lib/intel64;\
-Wl,--no-as-needed;\
-lmkl_intel_lp64;\
-lmkl_gnu_thread;\
-lmkl_core;\
-lgomp;\
-lpthread;\
-lm;\
-ldl"
        )
        set(MATH_LINK_LIBRARIES mkl_imported)
    endif()

elseif(USE_LIBSCI)

    include(CheckFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES "sci_gnu" "m")
    check_function_exists(dgemm_ HAVE_LIBSCI)

    if(HAVE_LIBSCI)
        message(STATUS "Cray LibSci detected via compiler wrappers")
        add_library(libsci_imported INTERFACE IMPORTED)
        set(MATH_LINK_LIBRARIES libsci_imported)
        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(LIBSCI QUIET cray-libsci)
        endif()

        if(LIBSCI_FOUND)
            message(STATUS "Found LibSci via pkg-config")
            add_library(libsci_imported INTERFACE IMPORTED)
            set_target_properties(libsci_imported PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${LIBSCI_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES      "${LIBSCI_LIBRARIES}")
            set(MATH_LINK_LIBRARIES libsci_imported)
            set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
        else()
            message(FATAL_ERROR
                "USE_LIBSCI=ON but LibSci not found. "
                "Ensure you are using Cray compiler wrappers (CC/cc) or set PKG_CONFIG_PATH.")
        endif()
    endif()

else()

    set(BLA_VENDOR "" CACHE STRING "BLAS vendor hint (e.g., OpenBLAS, Generic)")

    find_package(BLAS  QUIET)
    find_package(LAPACK QUIET)

    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found system BLAS:   ${BLAS_LIBRARIES}")
        message(STATUS "Found system LAPACK: ${LAPACK_LIBRARIES}")

        add_library(math_imported INTERFACE IMPORTED)
        set_target_properties(math_imported PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES};${BLAS_LIBRARIES}")

        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS)

        find_library(LAPACKE_LIB NAMES lapacke)
        if(LAPACKE_LIB)
            message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")
            set_property(TARGET math_imported APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES "${LAPACKE_LIB}")
            list(APPEND MATH_COMPILE_DEFINITIONS EIGEN_USE_LAPACKE)
        else()
            message(STATUS "LAPACKE not found, using BLAS only for Eigen acceleration")
        endif()

        set(MATH_LINK_LIBRARIES math_imported)

    else()
        if(EXISTS "${CMAKE_SOURCE_DIR}/openblas/CMakeLists.txt")
            message(STATUS "Using previously cloned OpenBLAS from: ${CMAKE_SOURCE_DIR}/openblas")
            add_subdirectory("${CMAKE_SOURCE_DIR}/openblas"
                             "${CMAKE_BINARY_DIR}/openblas")
        else()
            message(STATUS "System BLAS/LAPACK not found. Cloning OpenBLAS into ${CMAKE_SOURCE_DIR}/openblas ...")
            FetchContent_Declare(
                openblas
                GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
                GIT_TAG        v0.3.27
                GIT_SHALLOW    TRUE
                SOURCE_DIR     "${CMAKE_SOURCE_DIR}/openblas"
            )
            set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
            set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
            set(C_LAPACK             ON  CACHE BOOL "" FORCE)
            set(NOFORTRAN            ON  CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(openblas)
            message(STATUS "OpenBLAS cloned to: ${CMAKE_SOURCE_DIR}/openblas")
        endif()

        set(MATH_LINK_LIBRARIES openblas)
        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS)
        message(STATUS "OpenBLAS will be built from source")
    endif()

endif()

# ------------------------------------------------------------------------------
# Bundled math dependency target
# ------------------------------------------------------------------------------
add_library(fasteig_math_deps INTERFACE)

if(DEFINED MATH_LINK_LIBRARIES)
    target_link_libraries(fasteig_math_deps INTERFACE ${MATH_LINK_LIBRARIES})
endif()

if(DEFINED MATH_COMPILE_DEFINITIONS)
    target_compile_definitions(fasteig_math_deps INTERFACE ${MATH_COMPILE_DEFINITIONS})
endif()

target_link_libraries(fasteig_math_deps INTERFACE Eigen3::Eigen)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
endif()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Library output properties
# ------------------------------------------------------------------------------
if(TARGET fasteig)
    set_target_properties(fasteig PROPERTIES
        OUTPUT_NAME                "fasteig_${PLAT}"
        ARCHIVE_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== fasteig build configuration ===")
message(STATUS "  Platform:       ${PLAT}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  USE_MKL:        ${USE_MKL}")
message(STATUS "  USE_LIBSCI:     ${USE_LIBSCI}")
message(STATUS "  BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_TESTS:    ${BUILD_TESTS}")
message(STATUS "  Math defines:   ${MATH_COMPILE_DEFINITIONS}")
if(USE_MKL)
    message(STATUS "  MKLROOT:        ${MKLROOT}")
    message(STATUS "  MKL interface:  ${MKL_INTERFACE}")
    message(STATUS "  MKL threading:  ${MKL_THREADING}")
endif()
message(STATUS "===================================")
message(STATUS "")
