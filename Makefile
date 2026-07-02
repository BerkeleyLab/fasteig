MAKEINC ?= make.inc
MAKEINC_ABS := $(abspath $(MAKEINC))
include $(MAKEINC)
all: lib 
lib: libfasteig
examples: libfasteig
	$(MAKE) -C $(HOME)/examples MAKEINC=$(MAKEINC_ABS) USE_MKL="$(USE_MKL)" MKL="$(MKL)" USE_CUDA=$(USE_CUDA) USE_CUDSS=$(USE_CUDSS) CUDSS_PATH=$(CUDSS_PATH) CUDSS_LIBDIR=$(CUDSS_LIBDIR) CUDSS_LIB_NAME=$(CUDSS_LIB_NAME)
libfasteig:
	cd $(HOME)/src; \
        $(ECHO) Making lib in src; \
	$(MAKE) MAKEINC=$(MAKEINC_ABS) USE_MKL="$(USE_MKL)" MKL="$(MKL)" USE_CUDA=$(USE_CUDA) USE_CUDSS=$(USE_CUDSS) CUDSS_PATH=$(CUDSS_PATH) CUDSS_LIBDIR=$(CUDSS_LIBDIR) CUDSS_LIB_NAME=$(CUDSS_LIB_NAME); \
	cd ..; \
	$(RANLIB) $(LIBFASTEIG);
clean:
	cd $(HOME)/src; \
	$(MAKE) MAKEINC=$(MAKEINC_ABS) clean; \
	cd ..; 
	rm -f libfasteig_$(PLAT).a
	$(MAKE) -C $(HOME)/examples MAKEINC=$(MAKEINC_ABS) clean
help:
    @$(ECHO) "usage: make ?"

