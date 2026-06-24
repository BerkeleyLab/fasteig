
MKLROOT = /opt/intel/oneapi/mkl/2023.1.0
MKL = -L${MKLROOT}/lib/intel64 -Wl,-rpath ${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

HOME = /global/cfs/cdirs/m1027/Dong/fasteig
#HOME = /pscratch/sd/c/cyang/crd-fasteig

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

ifeq ($(USE_CUDA),1)
  CXXFLAGS += $(CUDA_CXXFLAGS)
endif

LIBFASTEIG  = $(HOME)/libfasteig_$(PLAT).a

RANLIB   = ranlib

ECHO	 = echo

AR = ar

