MAKEINC ?= ../make.inc
include $(MAKEINC)

# Notes
# -----
# By default, `make.inc` enables MKL/Pardiso via `-DEIGEN_USE_MKL_ALL` and links MKL.
# To build examples WITHOUT MKL/Pardiso, use:
#   make -C examples MAKEINC=../make.inc clean
#   make -C examples MAKEINC=../make.inc USE_MKL="" MKL=""

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag

$(BUILD_TAG_FILE):
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

EXECUTABLES = runfasteig_cpu
ifeq ($(USE_CUDA),1)
EXECUTABLES += runfasteig_gpu
endif

# Rule to build all executables
all: $(EXECUTABLES)

# Build the library only if it doesn't exist.
# If you change library sources, rebuild explicitly with:
#   make -C ../src MAKEINC=$(MAKEINC)
$(LIBFASTEIG):
	$(MAKE) -C ../src MAKEINC=$(MAKEINC)

# Pattern rule for creating executables

runfasteig_cpu: runfasteig_cpu.o $(LIBFASTEIG)
	$(CXX) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(if $(filter 1,$(USE_CUDA)),$(CUDA_LIBS),) -o $@

runfasteig_gpu: runfasteig_gpu.o $(LIBFASTEIG)
	$(CXX) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(CUDA_LIBS) -o $@

MTX2FEI: MTX2FEI.o $(LIBFASTEIG)
	$(CXX) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(if $(filter 1,$(USE_CUDA)),$(CUDA_LIBS),) -o $@

runfasteig_cpu.o: runfasteig.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

runfasteig_gpu.o: runfasteig.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -DFASTEIG_GPU_EXEC -MMD -MP -c $< -o $@

# Pattern rule for compiling .cpp files into .o files
%.o: %.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $<

# Clean target
clean:
	rm -f runfasteig runfasteig_cpu runfasteig_gpu *.o *.d

-include *.d

