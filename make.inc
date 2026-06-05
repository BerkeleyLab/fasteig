
MKLROOT = /opt/intel/oneapi/mkl/2023.1.0
MKL = -L${MKLROOT}/lib/intel64 -Wl,-rpath ${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

HOME = /pscratch/sd/c/cyang/Github/fasteig
#HOME = /pscratch/sd/c/cyang/crd-fasteig

CXX = CC
CC  = cc

PLAT = Perlmutter
MAKE = /usr/bin/make

USE_MKL = -DEIGEN_USE_MKL_ALL 
CXXFLAGS = $(USE_MKL) -O3 -I$(HOME)/include -I$(EIGEN) -I$(MKLROOT)/include -O3

EIGEN = /global/common/cori/software/eigen3/3.3.3/include/eigen3

LIBFASTEIG  = $(HOME)/libfasteig_$(PLAT).a

RANLIB   = ranlib

ECHO	 = echo

AR = ar


