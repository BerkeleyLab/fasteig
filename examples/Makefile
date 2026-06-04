MAKEINC ?= ../make.inc
include $(MAKEINC)

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag

$(BUILD_TAG_FILE):
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

# List of source files (without .cpp extension)
SOURCES = runfasteig
EXECUTABLES = $(SOURCES:%=%)

# Rule to build all executables
all: $(EXECUTABLES)

# Build the library only if it doesn't exist.
# If you change library sources, rebuild explicitly with:
#   make -C ../src MAKEINC=$(MAKEINC)
$(LIBFASTEIG):
	$(MAKE) -C ../src MAKEINC=$(MAKEINC)

# Pattern rule for creating executables

runfasteig: runfasteig.o $(LIBFASTEIG)
	$(CXX) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) -o $@

MTX2FEI: MTX2FEI.o $(LIBFASTEIG)
	$(CXX) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) -o $@

# Pattern rule for compiling .cpp files into .o files
%.o: %.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $<

# Clean target
clean:
	rm -f $(EXECUTABLES) *.o *.d

-include *.d


