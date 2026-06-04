MAKEINC ?= make.inc
MAKEINC_ABS := $(abspath $(MAKEINC))
include $(MAKEINC)

all: lib 

lib: libfasteig

libfasteig:
	cd $(HOME)/src; \
        $(ECHO) Making lib in src; \
	$(MAKE) MAKEINC=$(MAKEINC_ABS); \
	cd ..; \
	$(RANLIB) $(LIBFASTEIG);

clean:
	cd $(HOME)/src; \
	$(MAKE) MAKEINC=$(MAKEINC_ABS) clean; \
	cd ..; 
	rm -f libfasteig_$(PLAT).a

help:
    @$(ECHO) "usage: make ?"

