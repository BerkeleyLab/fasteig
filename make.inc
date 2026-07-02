
MKLROOT = /opt/intel/oneapi/mkl/2023.1.0
MKL = -L${MKLROOT}/lib/intel64 -Wl,-rpath ${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

# Repository root.  This works whether make.inc is included from the repo root,
# src/, or examples/, and avoids accidentally linking against another checkout.
HOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# MKL / Pardiso toggle
# --------------------
# By default, we enable Eigen's MKL integration (PardisoSupport) by defining `EIGEN_USE_MKL_ALL`
# and linking MKL.
#
# To build WITHOUT MKL/Pardiso (use Eigen's native sparse LDLT), override `USE_MKL` and also avoid
# linking MKL when building examples:
#   make -C src MAKEINC=../make.inc clean
#   make -C src MAKEINC=../make.inc USE_MKL=""
#   make -C examples MAKEINC=../make.inc clean
#   make -C examples MAKEINC=../make.inc USE_MKL="" MKL=""

CXX = CC
CC  = cc

PLAT = Perlmutter
MAKE = /usr/bin/make

USE_MKL = -DEIGEN_USE_MKL_ALL
CXXFLAGS = $(USE_MKL) -O3 -I$(HOME)/include -I$(EIGEN) -I$(MKLROOT)/include -O3

EIGEN = /global/common/cori/software/eigen3/3.3.3/include/eigen3

# CUDA / GPU toggle
# -----------------
# Perlmutter GPU build example:
#   make clean
#   make USE_CUDA=1
#   make -C examples USE_CUDA=1
#
# This builds `examples/runfasteig_cpu` and `examples/runfasteig_gpu`.
# `runfasteig_gpu` maps input decks with `solver Lanczos` to the CUDA Lanczos backend.
USE_CUDA ?= 0
CUDA_PATH ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.5
CUDA_ARCH ?= 80
NVCC ?= nvcc
CUDA_INC = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/cuda/include -I$(CUDA_PATH)/math_libs/include
CUDA_LIBS = -L$(CUDA_PATH)/cuda/lib64 -L$(CUDA_PATH)/math_libs/lib64 \
            -lcusparse -lcusolver -lcublas -lcudart
CUDA_CXXFLAGS = -DUSE_CUDA $(CUDA_INC)
CUDA_NVCCFLAGS = -std=c++17 -O3 -DUSE_CUDA -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
                 -I$(HOME)/include -I$(EIGEN) $(CUDA_INC)

# cuDSS / CUDA sparse-direct toggle
# ---------------------------------
# CUDA builds use cuDSS automatically when `cudss.h` is available.  The default
# search first checks the checkout-local install created by `tools/install_cudss.sh`,
# then a sibling project-space install, then NVHPC math_libs.
# If cuDSS is installed elsewhere, pass CUDSS_PATH explicitly:
#   make -C examples USE_CUDA=1 CUDSS_PATH=/path/to/cudss
#
# If cuDSS is unavailable, CUDA still builds without CUDA SAI support:
#   make -C examples USE_CUDA=1 USE_CUDSS=0
CUDSS_EXTERN_CU12 := $(abspath $(HOME)/extern/cudss/python/nvidia/cu12)
CUDSS_EXTERN_CU13 := $(abspath $(HOME)/extern/cudss/python/nvidia/cu13)
CUDSS_PROJECT_CU12 := $(abspath $(HOME)/../cudss/python/nvidia/cu12)
CUDSS_PROJECT_CU13 := $(abspath $(HOME)/../cudss/python/nvidia/cu13)
ifeq ($(origin CUDSS_PATH),undefined)
  ifneq ($(wildcard $(CUDSS_EXTERN_CU12)/include/cudss.h),)
    CUDSS_PATH = $(CUDSS_EXTERN_CU12)
  else ifneq ($(wildcard $(CUDSS_EXTERN_CU13)/include/cudss.h),)
    CUDSS_PATH = $(CUDSS_EXTERN_CU13)
  else ifneq ($(wildcard $(CUDSS_PROJECT_CU12)/include/cudss.h),)
    CUDSS_PATH = $(CUDSS_PROJECT_CU12)
  else ifneq ($(wildcard $(CUDSS_PROJECT_CU13)/include/cudss.h),)
    CUDSS_PATH = $(CUDSS_PROJECT_CU13)
  else
    CUDSS_PATH = $(CUDA_PATH)/math_libs
  endif
endif
CUDSS_LIB64_LIBS := $(wildcard $(CUDSS_PATH)/lib64/libcudss.so) $(wildcard $(CUDSS_PATH)/lib64/libcudss.so.0)
CUDSS_LIB_LIBS := $(wildcard $(CUDSS_PATH)/lib/libcudss.so) $(wildcard $(CUDSS_PATH)/lib/libcudss.so.0)
CUDSS_LIBDIR ?= $(if $(strip $(CUDSS_LIB64_LIBS)),$(CUDSS_PATH)/lib64,$(CUDSS_PATH)/lib)
CUDSS_HEADER := $(wildcard $(CUDSS_PATH)/include/cudss.h)
CUDSS_SONAME := $(wildcard $(CUDSS_LIBDIR)/libcudss.so)
CUDSS_VERSIONED_SONAME := $(wildcard $(CUDSS_LIBDIR)/libcudss.so.0)

ifeq ($(origin CUDSS_LIB_NAME),undefined)
  ifneq ($(CUDSS_SONAME),)
    CUDSS_LIB_NAME = cudss
  else ifneq ($(CUDSS_VERSIONED_SONAME),)
    CUDSS_LIB_NAME = :libcudss.so.0
  else
    CUDSS_LIB_NAME = cudss
  endif
endif

ifeq ($(origin USE_CUDSS),undefined)
  ifeq ($(USE_CUDA),1)
    ifneq ($(CUDSS_HEADER),)
      ifneq ($(CUDSS_SONAME)$(CUDSS_VERSIONED_SONAME),)
        USE_CUDSS = 1
      else
        USE_CUDSS = 0
      endif
    else
      USE_CUDSS = 0
    endif
  else
    USE_CUDSS = 0
  endif
endif

CUDSS_INC = -I$(CUDSS_PATH)/include
CUDSS_LIBS = -L$(CUDSS_LIBDIR) -Wl,-rpath,$(CUDSS_LIBDIR) -l$(CUDSS_LIB_NAME)

ifeq ($(USE_CUDSS),1)
  ifeq ($(CUDSS_HEADER),)
    $(error USE_CUDSS=1 requested, but cudss.h was not found at $(CUDSS_PATH)/include/cudss.h. Set CUDSS_PATH to the cuDSS install prefix or use USE_CUDSS=0)
  endif
  ifeq ($(CUDSS_SONAME)$(CUDSS_VERSIONED_SONAME),)
    $(error USE_CUDSS=1 requested, but libcudss was not found in $(CUDSS_LIBDIR). Set CUDSS_PATH/CUDSS_LIBDIR to the cuDSS install prefix/lib dir or use USE_CUDSS=0)
  endif
  CUDA_INC += $(CUDSS_INC)
  CUDA_LIBS += $(CUDSS_LIBS)
  CUDA_CXXFLAGS += -DUSE_CUDSS
  CUDA_NVCCFLAGS += -DUSE_CUDSS
endif

ifeq ($(USE_CUDA),1)
  CXXFLAGS += $(CUDA_CXXFLAGS)
endif

LIBFASTEIG  = $(HOME)/libfasteig_$(PLAT).a

RANLIB   = ranlib

ECHO	 = echo

AR = ar

