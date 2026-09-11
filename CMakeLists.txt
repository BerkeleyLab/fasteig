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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config generators only: pick a default and validate it.
if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
    endif()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR
            "CMAKE_BUILD_TYPE must be 'Release' or 'Debug'. Got: '${CMAKE_BUILD_TYPE}'")
    endif()
endif()

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# All third-party sources and downloads live here. Override to share a single
# download cache across several build directories.
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps" CACHE PATH
    "Directory for all downloaded third-party dependencies")
set(FASTEIG_DEPS_DIR "${FETCHCONTENT_BASE_DIR}")

set(PLAT "Generic" CACHE STRING "Platform name (e.g., Perlmutter, Frontier, spack)")

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(USE_MKL        "Use Intel MKL for BLAS/LAPACK"             OFF)
option(USE_LIBSCI     "Use Cray LibSci for BLAS/LAPACK"           OFF)
option(ENABLE_CUDA    "Build the NVIDIA GPU backend (src/gpu)"    OFF)
option(USE_CUDSS      "Use the NVIDIA cuDSS sparse direct solver" ON)
option(BUILD_EXAMPLES "Build example programs"                    ON)
option(BUILD_TESTS    "Build and enable tests"                    OFF)

# Package managers (Spack, EasyBuild, distro builds) inject their own -march and
# expect to control the target microarchitecture. Turn this off for those builds.
option(FASTEIG_ARCH_NATIVE "Add -march=native to Release builds"  ON)

if(USE_MKL AND USE_LIBSCI)
    message(FATAL_ERROR "USE_MKL and USE_LIBSCI are mutually exclusive. Choose one.")
endif()

if(USE_CUDSS AND NOT ENABLE_CUDA)
    message(STATUS "USE_CUDSS ignored because ENABLE_CUDA=OFF")
    set(USE_CUDSS OFF)
endif()

if(BUILD_TESTS AND NOT BUILD_EXAMPLES)
    message(FATAL_ERROR
        "BUILD_TESTS=ON requires BUILD_EXAMPLES=ON (runfasteig is the test driver).")
endif()

if(ENABLE_CUDA AND NOT EXISTS "${CMAKE_SOURCE_DIR}/src/gpu/CMakeLists.txt")
    message(FATAL_ERROR "ENABLE_CUDA=ON but src/gpu/CMakeLists.txt is missing.")
endif()

# ------------------------------------------------------------------------------
# CUDA language
# ------------------------------------------------------------------------------
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "ENABLE_CUDA=ON but no CUDA compiler was found. Load a CUDA module, put "
            "nvcc on PATH, or pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc.")
    endif()

    # nvcc must be paired with a host compiler it supports. icpx/icx are not
    # supported hosts, so fall back to the system g++ for device translation units.
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
        else()
            find_program(FASTEIG_HOST_CXX NAMES g++)
            if(FASTEIG_HOST_CXX)
                set(CMAKE_CUDA_HOST_COMPILER "${FASTEIG_HOST_CXX}")
                message(WARNING
                    "CXX compiler is ${CMAKE_CXX_COMPILER_ID}, which nvcc does not support "
                    "as a host compiler. Using ${FASTEIG_HOST_CXX} for CUDA sources.")
            endif()
        endif()
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
        endif()
    endif()

    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA toolkit:  ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR})")
    message(STATUS "CUDA host cxx: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA arch:     ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ------------------------------------------------------------------------------
# Compiler detection and flags
#
# Flags are wrapped in COMPILE_LANGUAGE genexes so that host-only options such as
# -march=native and -Wpedantic are never handed to nvcc, which rejects them.
# ------------------------------------------------------------------------------
set(FE_REL_FLAGS "")
set(FE_DBG_FLAGS "")
set(FE_DBG_LINK  "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    set(FE_REL_FLAGS -O3 -funroll-loops)
    set(FE_DBG_FLAGS -O0 -g -Wall -Wextra -Wpedantic)
    if(FASTEIG_ARCH_NATIVE)
        list(APPEND FE_REL_FLAGS -march=native)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    set(FE_REL_FLAGS -O3 -funroll-loops)
    set(FE_DBG_FLAGS -O0 -g -Wall -Wextra -Wpedantic)
    if(FASTEIG_ARCH_NATIVE)
        list(APPEND FE_REL_FLAGS -march=native)
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(STATUS "Compiler: Intel LLVM (icpx) ${CMAKE_CXX_COMPILER_VERSION}")
    set(FE_REL_FLAGS -O3 -funroll-loops -fp-model=fast)
    set(FE_DBG_FLAGS -O0 -g -Wall -Wextra)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(STATUS "Compiler: Intel Classic (icc/icpc) ${CMAKE_CXX_COMPILER_VERSION}")
    set(FE_REL_FLAGS -O3 -funroll-loops -fp-model=fast)
    set(FE_DBG_FLAGS -O0 -g -Wall -Wextra)
    if(FASTEIG_ARCH_NATIVE)
        list(APPEND FE_REL_FLAGS -march=native)
    endif()

else()
    message(WARNING "Untested compiler: ${CMAKE_CXX_COMPILER_ID}. Using default flags.")
    set(FE_REL_FLAGS -O3)
    set(FE_DBG_FLAGS -O0 -g)
endif()

# ASan interacts badly with CUDA device code; use compute-sanitizer for GPU builds.
if(NOT ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    list(APPEND FE_DBG_FLAGS -fsanitize=address)
    set(FE_DBG_LINK -fsanitize=address)
elseif(ENABLE_CUDA AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "AddressSanitizer disabled for CUDA builds; use compute-sanitizer.")
endif()

add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CXX,C>,$<CONFIG:Release>>:${FE_REL_FLAGS}>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX,C>,$<CONFIG:Debug>>:${FE_DBG_FLAGS}>"
)
if(FE_DBG_LINK)
    add_link_options("$<$<CONFIG:Debug>:${FE_DBG_LINK}>")
endif()

if(ENABLE_CUDA)
    # --expt-relaxed-constexpr is required whenever a .cu file includes Eigen.
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G;--expt-relaxed-constexpr;--expt-extended-lambda>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:SHELL:-Xcompiler -O3>"
    )
endif()

if(FASTEIG_ARCH_NATIVE AND (PLAT STREQUAL "Perlmutter" OR PLAT STREQUAL "Frontier"))
    message(STATUS
        "Note: -march=native reflects the login node, which may differ from the "
        "compute nodes on ${PLAT}. Consider -DFASTEIG_ARCH_NATIVE=OFF.")
endif()

# ------------------------------------------------------------------------------
# Eigen  ->  ${FASTEIG_DEPS_DIR}/eigen-src
# ------------------------------------------------------------------------------
set(EIGEN_DIR "" CACHE PATH "Path to an existing Eigen3 installation")
set(EIGEN_TAG "3.4.0" CACHE STRING "Eigen git tag to fetch when EIGEN_DIR is unset")

set(FASTEIG_EIGEN_INCLUDE "")

if(EIGEN_DIR)
    if(EXISTS "${EIGEN_DIR}/Eigen/Core")
        set(FASTEIG_EIGEN_INCLUDE "${EIGEN_DIR}")
    elseif(EXISTS "${EIGEN_DIR}/eigen3/Eigen/Core")
        set(FASTEIG_EIGEN_INCLUDE "${EIGEN_DIR}/eigen3")
    elseif(EXISTS "${EIGEN_DIR}/include/eigen3/Eigen/Core")
        set(FASTEIG_EIGEN_INCLUDE "${EIGEN_DIR}/include/eigen3")
    else()
        message(FATAL_ERROR "EIGEN_DIR set to '${EIGEN_DIR}' but Eigen/Core was not found there.")
    endif()
    message(STATUS "Using Eigen from: ${FASTEIG_EIGEN_INCLUDE}")

else()
    message(STATUS "Fetching Eigen ${EIGEN_TAG} into ${FASTEIG_DEPS_DIR} ...")
    # SOURCE_SUBDIR points at a directory with no CMakeLists.txt, so the content is
    # downloaded but Eigen's own project() is never added. Eigen is header-only, so
    # nothing is lost, and its cache variables stay out of our build.
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        ${EIGEN_TAG}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SOURCE_SUBDIR  "fasteig-no-configure"
    )
    FetchContent_MakeAvailable(eigen)
    set(FASTEIG_EIGEN_INCLUDE "${eigen_SOURCE_DIR}")
    message(STATUS "Eigen available at: ${FASTEIG_EIGEN_INCLUDE}")
endif()

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FASTEIG_EIGEN_INCLUDE}")
endif()

# ------------------------------------------------------------------------------
# BLAS and LAPACK
#
# FE_MATH_LIBS holds a plain list of link items for the exported package config.
# Every branch below must populate it, or downstream find_package(fasteig) users
# will get undefined BLAS symbols.
# ------------------------------------------------------------------------------
set(MATH_LINK_LIBRARIES      "")
set(MATH_COMPILE_DEFINITIONS "")
set(FE_MATH_LIBS             "")
set(FE_EXPORT_NEEDS_MKL      FALSE)
set(FASTEIG_BUNDLED_OPENBLAS FALSE)

# Escape hatch for package managers: skip find_package entirely and use exactly
# the link line provided. Spack resolves BLAS/LAPACK itself and passes them here.
set(FASTEIG_BLAS_LIBRARIES   "" CACHE STRING
    "Explicit BLAS link line (bypasses find_package)")
set(FASTEIG_LAPACK_LIBRARIES "" CACHE STRING
    "Explicit LAPACK link line (bypasses find_package)")

if(USE_MKL)

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
    list(APPEND CMAKE_PREFIX_PATH "${MKLROOT}/lib/cmake/mkl")

    # The threading layer must match the compiler's OpenMP runtime. Mixing
    # libgomp and libiomp5 in one process gives wrong results or deadlocks.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(FE_MKL_THREADING "intel_thread")
        set(FE_MKL_OMP_LIB   "-liomp5")
    else()
        set(FE_MKL_THREADING "gnu_thread")
        set(FE_MKL_OMP_LIB   "-lgomp")
    endif()

    # LP64 is required: Eigen's Pardiso wrapper uses int throughout. ILP64 would
    # change MKL_INT to long long and break the pardiso() signature.
    # Do NOT add MKL_ILP64 to the compile definitions.
    set(MKL_INTERFACE "lp64"                CACHE STRING "MKL integer interface" FORCE)
    set(MKL_THREADING "${FE_MKL_THREADING}" CACHE STRING "MKL threading layer"   FORCE)

    set(MATH_COMPILE_DEFINITIONS EIGEN_USE_MKL_ALL USE_MKL)

    find_package(MKL QUIET CONFIG)
    if(MKL_FOUND)
        message(STATUS "Found MKL via CMake config")
        message(STATUS "  MKL root:  ${MKL_ROOT}")
        message(STATUS "  Interface: ${MKL_INTERFACE}")
        message(STATUS "  Threading: ${MKL_THREADING}")
        set(MATH_LINK_LIBRARIES MKL::MKL)
        # Exported config re-resolves this via find_dependency(MKL).
        set(FE_EXPORT_NEEDS_MKL TRUE)
    else()
        message(STATUS
            "MKL CMake config not found, using manual link line (LP64 + ${FE_MKL_THREADING})")

        add_library(mkl_imported INTERFACE IMPORTED)
        set_target_properties(mkl_imported PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKLROOT}/include")
        target_link_options(mkl_imported INTERFACE
            "-L${MKLROOT}/lib/intel64"
            "-Wl,-rpath,${MKLROOT}/lib/intel64"
            "-Wl,--no-as-needed")
        target_link_libraries(mkl_imported INTERFACE
            -lmkl_intel_lp64
            "-lmkl_${FE_MKL_THREADING}"
            -lmkl_core
            ${FE_MKL_OMP_LIB}
            -lpthread
            -lm
            -ldl)
        set(MATH_LINK_LIBRARIES mkl_imported)

        # Flat list for the exported config. -L/-Wl items survive as link flags.
        set(FE_MATH_LIBS
            "-L${MKLROOT}/lib/intel64"
            "-Wl,-rpath,${MKLROOT}/lib/intel64"
            "-Wl,--no-as-needed"
            "-lmkl_intel_lp64"
            "-lmkl_${FE_MKL_THREADING}"
            "-lmkl_core"
            "${FE_MKL_OMP_LIB}"
            "-lpthread" "-lm" "-ldl")
    endif()

elseif(USE_LIBSCI)

    include(CheckFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES "sci_gnu" "m")
    check_function_exists(dgemm_ HAVE_LIBSCI)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if(HAVE_LIBSCI)
        message(STATUS "Cray LibSci detected via compiler wrappers")
        add_library(libsci_imported INTERFACE IMPORTED)
        # Link it explicitly: relying on the wrapper to inject it silently breaks
        # as soon as anyone configures without CC/cc.
        set_target_properties(libsci_imported PROPERTIES
            INTERFACE_LINK_LIBRARIES "sci_gnu;m")
        set(MATH_LINK_LIBRARIES libsci_imported)
        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
        set(FE_MATH_LIBS "sci_gnu" "m")
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
            set(FE_MATH_LIBS ${LIBSCI_LIBRARIES})
        else()
            message(FATAL_ERROR
                "USE_LIBSCI=ON but LibSci was not found. Ensure you are using the Cray "
                "compiler wrappers (CC/cc) or set PKG_CONFIG_PATH.")
        endif()
    endif()

elseif(FASTEIG_BLAS_LIBRARIES OR FASTEIG_LAPACK_LIBRARIES)

    message(STATUS "Using explicitly provided BLAS/LAPACK")
    message(STATUS "  BLAS:   ${FASTEIG_BLAS_LIBRARIES}")
    message(STATUS "  LAPACK: ${FASTEIG_LAPACK_LIBRARIES}")

    set(FE_MATH_LIBS ${FASTEIG_LAPACK_LIBRARIES} ${FASTEIG_BLAS_LIBRARIES})
    list(REMOVE_DUPLICATES FE_MATH_LIBS)

    add_library(math_imported INTERFACE IMPORTED)
    set_target_properties(math_imported PROPERTIES
        INTERFACE_LINK_LIBRARIES "${FE_MATH_LIBS}")
    set(MATH_LINK_LIBRARIES math_imported)
    set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS)

else()

    set(BLA_VENDOR "" CACHE STRING "BLAS vendor hint (e.g., OpenBLAS, Generic)")

    find_package(BLAS   QUIET)
    find_package(LAPACK QUIET)

    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found system BLAS:   ${BLAS_LIBRARIES}")
        message(STATUS "Found system LAPACK: ${LAPACK_LIBRARIES}")

        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS)

        # LAPACKE depends on LAPACK, so with static archives it must come first.
        find_library(LAPACKE_LIB NAMES lapacke)
        if(LAPACKE_LIB)
            message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")
            list(APPEND FE_MATH_LIBS "${LAPACKE_LIB}")
            list(APPEND MATH_COMPILE_DEFINITIONS EIGEN_USE_LAPACKE)
        else()
            message(STATUS "LAPACKE not found, using BLAS only for Eigen acceleration")
        endif()
        list(APPEND FE_MATH_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

        add_library(math_imported INTERFACE IMPORTED)
        set_target_properties(math_imported PROPERTIES
            INTERFACE_LINK_LIBRARIES "${FE_MATH_LIBS}")
        set(MATH_LINK_LIBRARIES math_imported)

    else()
        message(STATUS
            "System BLAS/LAPACK not found. Building OpenBLAS from source in ${FASTEIG_DEPS_DIR} ...")

        FetchContent_Declare(
            openblas
            GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
            GIT_TAG        v0.3.27
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )
        set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
        set(BUILD_STATIC_LIBS    ON  CACHE BOOL "" FORCE)
        set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
        set(C_LAPACK             ON  CACHE BOOL "" FORCE)
        set(NOFORTRAN            ON  CACHE BOOL "" FORCE)
        set(BUILD_TESTING        OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(openblas)

        # The target name differs between OpenBLAS versions and static/shared builds.
        if(TARGET openblas_static)
            set(FE_OPENBLAS_TARGET openblas_static)
        elseif(TARGET openblas)
            set(FE_OPENBLAS_TARGET openblas)
        else()
            message(FATAL_ERROR
                "OpenBLAS was fetched but neither the 'openblas' nor 'openblas_static' "
                "target exists. Check the OpenBLAS version pin.")
        endif()

        target_include_directories(${FE_OPENBLAS_TARGET} INTERFACE
            $<BUILD_INTERFACE:${openblas_SOURCE_DIR}>
            $<BUILD_INTERFACE:${openblas_SOURCE_DIR}/lapack-netlib/LAPACKE/include>
            $<BUILD_INTERFACE:${openblas_BINARY_DIR}>
            $<BUILD_INTERFACE:${openblas_BINARY_DIR}/generated>)

        set(MATH_LINK_LIBRARIES ${FE_OPENBLAS_TARGET})
        set(MATH_COMPILE_DEFINITIONS EIGEN_USE_BLAS)
        set(FASTEIG_BUNDLED_OPENBLAS TRUE)

        # The bundled archive is installed next to libfasteig, so the exported
        # config refers to it by name relative to the install prefix.
        set(FE_MATH_LIBS "\${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_LIBDIR}/libopenblas.a")
        message(STATUS "OpenBLAS source: ${openblas_SOURCE_DIR}")
    endif()

endif()

# ------------------------------------------------------------------------------
# cuDSS  ->  ${FASTEIG_DEPS_DIR}/cudss
# ------------------------------------------------------------------------------
set(FASTEIG_CUDSS_ROOT "")
set(CUDSS_TARGET "")

if(USE_CUDSS)

    set(CUDSS_DIR "" CACHE PATH
        "Path to an existing cuDSS install (directory containing include/ and lib/)")
    set(CUDSS_VERSION "0.8.0.10" CACHE STRING
        "cuDSS redistributable version to download when CUDSS_DIR is not set")
    set(CUDSS_URL_HASH "" CACHE STRING
        "SHA256 of the cuDSS tarball (recommended; empty skips verification)")

    if(CUDSS_DIR)
        if(NOT EXISTS "${CUDSS_DIR}/include/cudss.h")
            message(FATAL_ERROR
                "CUDSS_DIR='${CUDSS_DIR}' but include/cudss.h was not found there.")
        endif()
        set(FASTEIG_CUDSS_ROOT "${CUDSS_DIR}")
        message(STATUS "Using cuDSS from: ${FASTEIG_CUDSS_ROOT}")

    elseif(DEFINED ENV{CUDSS_ROOT} AND EXISTS "$ENV{CUDSS_ROOT}/include/cudss.h")
        set(FASTEIG_CUDSS_ROOT "$ENV{CUDSS_ROOT}")
        message(STATUS "Using cuDSS from CUDSS_ROOT: ${FASTEIG_CUDSS_ROOT}")

    elseif(EXISTS "${FASTEIG_DEPS_DIR}/cudss/include/cudss.h")
        set(FASTEIG_CUDSS_ROOT "${FASTEIG_DEPS_DIR}/cudss")
        message(STATUS "Using previously downloaded cuDSS: ${FASTEIG_CUDSS_ROOT}")

    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(FE_CUDSS_PLAT "linux-sbsa")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
            set(FE_CUDSS_PLAT "linux-x86_64")
        else()
            message(FATAL_ERROR
                "No cuDSS redistributable for CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'. "
                "Install cuDSS manually and pass -DCUDSS_DIR=...")
        endif()

        set(FE_CUDSS_ARCHIVE
            "libcudss-${FE_CUDSS_PLAT}-${CUDSS_VERSION}_cuda${CUDAToolkit_VERSION_MAJOR}-archive")
        set(FE_CUDSS_URL
            "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/${FE_CUDSS_PLAT}/${FE_CUDSS_ARCHIVE}.tar.xz")
        set(FE_CUDSS_TARBALL "${FASTEIG_DEPS_DIR}/${FE_CUDSS_ARCHIVE}.tar.xz")

        file(MAKE_DIRECTORY "${FASTEIG_DEPS_DIR}")
        message(STATUS "Downloading cuDSS ${CUDSS_VERSION} ...")
        message(STATUS "  ${FE_CUDSS_URL}")

        if(CUDSS_URL_HASH)
            file(DOWNLOAD "${FE_CUDSS_URL}" "${FE_CUDSS_TARBALL}"
                 SHOW_PROGRESS TLS_VERIFY ON
                 EXPECTED_HASH SHA256=${CUDSS_URL_HASH}
                 STATUS FE_CUDSS_DL_STATUS)
        else()
            message(WARNING
                "CUDSS_URL_HASH is not set: downloading cuDSS without checksum verification.")
            file(DOWNLOAD "${FE_CUDSS_URL}" "${FE_CUDSS_TARBALL}"
                 SHOW_PROGRESS TLS_VERIFY ON
                 STATUS FE_CUDSS_DL_STATUS)
        endif()

        list(GET FE_CUDSS_DL_STATUS 0 FE_CUDSS_DL_CODE)
        if(NOT FE_CUDSS_DL_CODE EQUAL 0)
            list(GET FE_CUDSS_DL_STATUS 1 FE_CUDSS_DL_MSG)
            file(REMOVE "${FE_CUDSS_TARBALL}")
            message(FATAL_ERROR
                "Failed to download cuDSS: ${FE_CUDSS_DL_MSG}\n"
                "  URL: ${FE_CUDSS_URL}\n"
                "Your CUDA toolkit is ${CUDAToolkit_VERSION}; cuDSS ${CUDSS_VERSION} may not "
                "ship a cuda${CUDAToolkit_VERSION_MAJOR} build.\n"
                "List available releases with\n"
                "  curl -s https://developer.download.nvidia.com/compute/cudss/redist/redistrib_<ver>.json\n"
                "then pass -DCUDSS_VERSION=<full.four.part.version> and -DCUDSS_URL_HASH=<sha256>,\n"
                "or install cuDSS yourself and pass -DCUDSS_DIR=<path>,\n"
                "or configure with -DUSE_CUDSS=OFF.")
        endif()

        file(ARCHIVE_EXTRACT
             INPUT       "${FE_CUDSS_TARBALL}"
             DESTINATION "${FASTEIG_DEPS_DIR}")

        file(REMOVE_RECURSE "${FASTEIG_DEPS_DIR}/cudss")
        file(RENAME "${FASTEIG_DEPS_DIR}/${FE_CUDSS_ARCHIVE}"
                    "${FASTEIG_DEPS_DIR}/cudss")

        set(FASTEIG_CUDSS_ROOT "${FASTEIG_DEPS_DIR}/cudss")
        message(STATUS "cuDSS extracted to: ${FASTEIG_CUDSS_ROOT}")
    endif()

    # Prefer the shipped package config so transitive dependencies are correct.
    find_package(cudss CONFIG QUIET
        PATHS "${FASTEIG_CUDSS_ROOT}"
        PATH_SUFFIXES lib/cmake/cudss lib64/cmake/cudss
        NO_DEFAULT_PATH)

    if(cudss_FOUND AND TARGET cudss)
        message(STATUS "Found cuDSS via its CMake config (version ${cudss_VERSION})")
        set(CUDSS_TARGET cudss)
        get_target_property(FE_CUDSS_LOC cudss IMPORTED_LOCATION)
        if(NOT FE_CUDSS_LOC)
            get_target_property(FE_CUDSS_LOC cudss IMPORTED_LOCATION_RELEASE)
        endif()
    else()
        message(STATUS "cuDSS CMake config not found; building imported target manually")

        find_library(CUDSS_LIBRARY
            NAMES cudss
            PATHS "${FASTEIG_CUDSS_ROOT}/lib" "${FASTEIG_CUDSS_ROOT}/lib64"
            PATH_SUFFIXES "${CUDAToolkit_VERSION_MAJOR}"
            NO_DEFAULT_PATH)

        if(NOT CUDSS_LIBRARY)
            message(FATAL_ERROR "libcudss was not found under ${FASTEIG_CUDSS_ROOT}/lib")
        endif()

        add_library(cudss_imported SHARED IMPORTED)
        set_target_properties(cudss_imported PROPERTIES
            IMPORTED_LOCATION             "${CUDSS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FASTEIG_CUDSS_ROOT}/include")
        target_link_libraries(cudss_imported INTERFACE CUDA::cudart CUDA::cublas)

        set(CUDSS_TARGET cudss_imported)
        set(FE_CUDSS_LOC "${CUDSS_LIBRARY}")
        message(STATUS "cuDSS library: ${CUDSS_LIBRARY}")
    endif()

    # cuDSS is a shared library outside the default loader path, so bake in an
    # RPATH. Without this every executable dies at startup with
    # "libcudss.so.0: cannot open shared object file".
    if(FE_CUDSS_LOC)
        get_filename_component(FE_CUDSS_LIBDIR "${FE_CUDSS_LOC}" DIRECTORY)
    else()
        set(FE_CUDSS_LIBDIR "${FASTEIG_CUDSS_ROOT}/lib")
    endif()
    list(APPEND CMAKE_BUILD_RPATH   "${FE_CUDSS_LIBDIR}")
    list(APPEND CMAKE_INSTALL_RPATH "${FE_CUDSS_LIBDIR}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

# ------------------------------------------------------------------------------
# Bundled dependency targets
# ------------------------------------------------------------------------------
add_library(fasteig_gpu_deps INTERFACE)

if(ENABLE_CUDA)
    target_link_libraries(fasteig_gpu_deps INTERFACE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver)
    target_compile_definitions(fasteig_gpu_deps INTERFACE FASTEIG_USE_CUDA)

    if(USE_CUDSS)
        target_link_libraries(fasteig_gpu_deps INTERFACE ${CUDSS_TARGET})
        target_compile_definitions(fasteig_gpu_deps INTERFACE FASTEIG_USE_CUDSS)
    endif()
endif()

add_library(fasteig_math_deps INTERFACE)

if(MATH_LINK_LIBRARIES)
    target_link_libraries(fasteig_math_deps INTERFACE ${MATH_LINK_LIBRARIES})
endif()

if(MATH_COMPILE_DEFINITIONS)
    target_compile_definitions(fasteig_math_deps INTERFACE ${MATH_COMPILE_DEFINITIONS})
endif()

target_link_libraries(fasteig_math_deps INTERFACE Eigen3::Eigen fasteig_gpu_deps)

# ------------------------------------------------------------------------------
# Version and build provenance (xSDK M8)
# ------------------------------------------------------------------------------
# Keep the public preprocessor version synchronized with project(VERSION).  The
# generated header below is private: the public API returns its embedded values
# without exposing build-tree paths to downstream consumers.
file(STRINGS "${CMAKE_SOURCE_DIR}/include/version.hpp"
    FASTEIG_PUBLIC_VERSION_LINE
    REGEX "^#define[ \\t]+FASTEIG_VERSION_STRING[ \\t]+\"")
string(REGEX MATCH "\"([^\"]+)\"" _fasteig_version_match
    "${FASTEIG_PUBLIC_VERSION_LINE}")
set(FASTEIG_PUBLIC_VERSION "${CMAKE_MATCH_1}")
if(NOT FASTEIG_PUBLIC_VERSION STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR
        "Public FASTEIG_VERSION_STRING (${FASTEIG_PUBLIC_VERSION}) does not match "
        "project version (${PROJECT_VERSION}).")
endif()

set(FASTEIG_GIT_COMMIT "" CACHE STRING
    "Git commit to embed (empty means detect it from the source checkout)")
set(FASTEIG_GIT_DIRTY "" CACHE STRING
    "Whether the embedded Git checkout is modified: 0, 1, or empty to detect")

find_package(Git QUIET)
if(NOT FASTEIG_GIT_COMMIT)
    set(FASTEIG_GIT_COMMIT "unknown")
    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --verify HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE _fasteig_git_result
            OUTPUT_VARIABLE _fasteig_git_commit
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_fasteig_git_result EQUAL 0 AND _fasteig_git_commit)
            set(FASTEIG_GIT_COMMIT "${_fasteig_git_commit}")
        endif()
    endif()
endif()

if(FASTEIG_GIT_DIRTY STREQUAL "")
    set(FASTEIG_GIT_DIRTY 0)
    if(Git_FOUND AND NOT FASTEIG_GIT_COMMIT STREQUAL "unknown")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" status --porcelain
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE _fasteig_status_result
            OUTPUT_VARIABLE _fasteig_git_status
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_fasteig_status_result EQUAL 0 AND NOT _fasteig_git_status STREQUAL "")
            set(FASTEIG_GIT_DIRTY 1)
        endif()
    endif()
elseif(FASTEIG_GIT_DIRTY)
    set(FASTEIG_GIT_DIRTY 1)
else()
    set(FASTEIG_GIT_DIRTY 0)
endif()

set(FASTEIG_BUILD_VERSION "${PROJECT_VERSION}")
if(NOT FASTEIG_GIT_COMMIT STREQUAL "unknown")
    string(SUBSTRING "${FASTEIG_GIT_COMMIT}" 0 12 FASTEIG_GIT_COMMIT_SHORT)
    string(APPEND FASTEIG_BUILD_VERSION "+git.${FASTEIG_GIT_COMMIT_SHORT}")
    if(FASTEIG_GIT_DIRTY)
        string(APPEND FASTEIG_BUILD_VERSION ".dirty")
    endif()
endif()

if(USE_MKL)
    set(FASTEIG_BUILD_USE_MKL 1)
else()
    set(FASTEIG_BUILD_USE_MKL 0)
endif()
if(USE_LIBSCI)
    set(FASTEIG_BUILD_USE_LIBSCI 1)
else()
    set(FASTEIG_BUILD_USE_LIBSCI 0)
endif()
if(ENABLE_CUDA)
    set(FASTEIG_BUILD_ENABLE_CUDA 1)
else()
    set(FASTEIG_BUILD_ENABLE_CUDA 0)
endif()
if(USE_CUDSS)
    set(FASTEIG_BUILD_USE_CUDSS 1)
else()
    set(FASTEIG_BUILD_USE_CUDSS 0)
endif()

if(CMAKE_CONFIGURATION_TYPES)
    set(_fasteig_flags_config "")
else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _fasteig_build_type_upper)
    set(_fasteig_flags_config "${CMAKE_CXX_FLAGS_${_fasteig_build_type_upper}}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(JOIN " " _fasteig_project_cxx_flags ${FE_DBG_FLAGS})
else()
    string(JOIN " " _fasteig_project_cxx_flags ${FE_REL_FLAGS})
endif()
string(STRIP
    "${CMAKE_CXX_FLAGS} ${_fasteig_flags_config} ${_fasteig_project_cxx_flags}"
    FASTEIG_CXX_FLAGS_TEXT)
string(REPLACE "\n" " " FASTEIG_CXX_FLAGS_TEXT "${FASTEIG_CXX_FLAGS_TEXT}")

set(FASTEIG_CMAKE_OPTIONS_TEXT
    "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE};PLAT=${PLAT};USE_MKL=${USE_MKL};USE_LIBSCI=${USE_LIBSCI};ENABLE_CUDA=${ENABLE_CUDA};USE_CUDSS=${USE_CUDSS};BUILD_EXAMPLES=${BUILD_EXAMPLES};BUILD_TESTS=${BUILD_TESTS};FASTEIG_ARCH_NATIVE=${FASTEIG_ARCH_NATIVE};EIGEN_DIR=${EIGEN_DIR};EIGEN_TAG=${EIGEN_TAG};FASTEIG_BLAS_LIBRARIES=${FASTEIG_BLAS_LIBRARIES};FASTEIG_LAPACK_LIBRARIES=${FASTEIG_LAPACK_LIBRARIES};CUDSS_DIR=${CUDSS_DIR};CUDSS_VERSION=${CUDSS_VERSION};CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

if(ENABLE_CUDA)
    set(FASTEIG_CUDA_COMPILER_TEXT "${CMAKE_CUDA_COMPILER}")
    set(FASTEIG_CUDA_VERSION_TEXT "${CUDAToolkit_VERSION}")
    set(FASTEIG_CUDA_ARCHITECTURES_TEXT "${CMAKE_CUDA_ARCHITECTURES}")
    if(CMAKE_CONFIGURATION_TYPES)
        set(_fasteig_cuda_flags_config "")
    else()
        set(_fasteig_cuda_flags_config
            "${CMAKE_CUDA_FLAGS_${_fasteig_build_type_upper}}")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_fasteig_project_cuda_flags
            "-O0 -g -G --expt-relaxed-constexpr --expt-extended-lambda")
    else()
        set(_fasteig_project_cuda_flags
            "-O3 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler -O3")
    endif()
    string(STRIP
        "${CMAKE_CUDA_FLAGS} ${_fasteig_cuda_flags_config} ${_fasteig_project_cuda_flags}"
        FASTEIG_CUDA_FLAGS_TEXT)
    string(REPLACE "\n" " " FASTEIG_CUDA_FLAGS_TEXT "${FASTEIG_CUDA_FLAGS_TEXT}")
else()
    set(FASTEIG_CUDA_COMPILER_TEXT "not enabled")
    set(FASTEIG_CUDA_VERSION_TEXT "not enabled")
    set(FASTEIG_CUDA_FLAGS_TEXT "not enabled")
    set(FASTEIG_CUDA_ARCHITECTURES_TEXT "not enabled")
endif()

set(FASTEIG_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${FASTEIG_GENERATED_INCLUDE_DIR}")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/fasteig_build_config.hpp.in"
    "${FASTEIG_GENERATED_INCLUDE_DIR}/fasteig_build_config.hpp"
    @ONLY)

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

target_include_directories(fasteig PRIVATE
    $<BUILD_INTERFACE:${FASTEIG_GENERATED_INCLUDE_DIR}>)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Library output properties
# ------------------------------------------------------------------------------
set(FE_LIB_SUFFIX "${PLAT}")
if(ENABLE_CUDA)
    string(APPEND FE_LIB_SUFFIX "_cuda")
endif()
set(FE_OUTPUT_NAME "fasteig_${FE_LIB_SUFFIX}")

if(TARGET fasteig)
    set_target_properties(fasteig PROPERTIES
        OUTPUT_NAME              "${FE_OUTPUT_NAME}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if(ENABLE_CUDA)
        set_target_properties(fasteig PROPERTIES
            CUDA_SEPARABLE_COMPILATION  ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
else()
    message(FATAL_ERROR
        "src/ did not create the 'fasteig' target. Check src/CMakeLists.txt.")
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
install(TARGETS fasteig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.hh"
        PATTERN "*.cuh")

if(BUILD_EXAMPLES AND TARGET runfasteig)
    install(TARGETS runfasteig RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# A bundled OpenBLAS is a build artifact of this project, so it must be installed
# alongside the library or downstream links will fail.
if(FASTEIG_BUNDLED_OPENBLAS AND TARGET ${FE_OPENBLAS_TARGET})
    install(TARGETS ${FE_OPENBLAS_TARGET}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# --- package config so downstream CMake projects can find_package(fasteig) ----
# The imported target is written by hand rather than via install(EXPORT):
# fasteig_math_deps links imported targets that belong to no export set, which
# makes export() fail outright.
set(FE_EXPORT_DEFS "${MATH_COMPILE_DEFINITIONS}")
if(ENABLE_CUDA)
    list(APPEND FE_EXPORT_DEFS FASTEIG_USE_CUDA)
    if(USE_CUDSS)
        list(APPEND FE_EXPORT_DEFS FASTEIG_USE_CUDSS)
    endif()
endif()

set(FE_EXPORT_EXTRA_LIBS "${FE_MATH_LIBS}")

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/fasteigConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/fasteigConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fasteig"
    PATH_VARS
        CMAKE_INSTALL_INCLUDEDIR
        CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/fasteigConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_BINARY_DIR}/fasteigConfig.cmake"
    "${CMAKE_BINARY_DIR}/fasteigConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fasteig")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== fasteig build configuration ===")
message(STATUS "  Version:        ${FASTEIG_BUILD_VERSION}")
message(STATUS "  Git commit:     ${FASTEIG_GIT_COMMIT}")
message(STATUS "  Source dirty:   ${FASTEIG_GIT_DIRTY}")
message(STATUS "  Platform:       ${PLAT}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CXX compiler:   ${CMAKE_CXX_COMPILER}")
message(STATUS "  C compiler:     ${CMAKE_C_COMPILER}")
message(STATUS "  C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  -march=native:  ${FASTEIG_ARCH_NATIVE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Deps directory: ${FASTEIG_DEPS_DIR}")
message(STATUS "  Eigen:          ${FASTEIG_EIGEN_INCLUDE}")
message(STATUS "  USE_MKL:        ${USE_MKL}")
message(STATUS "  USE_LIBSCI:     ${USE_LIBSCI}")
message(STATUS "  ENABLE_CUDA:    ${ENABLE_CUDA}")
if(ENABLE_CUDA)
    message(STATUS "  CUDA version:   ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA arch:      ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA host cxx:  ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "  USE_CUDSS:      ${USE_CUDSS}")
    if(USE_CUDSS)
        message(STATUS "  cuDSS root:     ${FASTEIG_CUDSS_ROOT}")
    endif()
endif()
message(STATUS "  BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_TESTS:    ${BUILD_TESTS}")
message(STATUS "  Math defines:   ${MATH_COMPILE_DEFINITIONS}")
if(USE_MKL)
    message(STATUS "  MKLROOT:        ${MKLROOT}")
    message(STATUS "  MKL interface:  ${MKL_INTERFACE}")
    message(STATUS "  MKL threading:  ${MKL_THREADING}")
endif()
message(STATUS "  Library name:   lib${FE_OUTPUT_NAME}.a")
message(STATUS "===================================")
message(STATUS "")

