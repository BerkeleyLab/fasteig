MAKEINC ?= ../make.inc
include $(MAKEINC)

CUDA_LINKER ?= $(CXX)
EXAMPLE_LINKER = $(if $(filter 1,$(USE_CUDA)),$(CUDA_LINKER),$(CXX))

# Notes
# -----
# By default, `make.inc` enables MKL/Pardiso via `-DEIGEN_USE_MKL_ALL` and links MKL.
# To build examples WITHOUT MKL/Pardiso, use:
#   make -C examples MAKEINC=../make.inc clean
#   make -C examples MAKEINC=../make.inc USE_MKL="" MKL=""

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag
.PHONY: force_build_tag

$(BUILD_TAG_FILE): force_build_tag
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; echo "USE_CUDA=$(USE_CUDA)"; echo "CUDA_PATH=$(CUDA_PATH)"; echo "CUDA_TOOLKIT_ROOT=$(CUDA_TOOLKIT_ROOT)"; echo "CUDA_MATH_ROOT=$(CUDA_MATH_ROOT)"; echo "CUDA_TOOLKIT_LIBDIR=$(CUDA_TOOLKIT_LIBDIR)"; echo "CUDA_MATH_LIBDIR=$(CUDA_MATH_LIBDIR)"; echo "CUDA_LINKER=$(CUDA_LINKER)"; echo "USE_CUDSS=$(USE_CUDSS)"; echo "CUDSS_PATH=$(CUDSS_PATH)"; echo "CUDSS_LIBDIR=$(CUDSS_LIBDIR)"; echo "CUDSS_LIB_NAME=$(CUDSS_LIB_NAME)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

EXECUTABLES = runfasteig_cpu
ifeq ($(USE_CUDA),1)
EXECUTABLES += runfasteig_gpu
endif

# Rule to build all executables
all: $(EXECUTABLES)

.PHONY: print-cuda-config
print-cuda-config:
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_TOOLKIT_ROOT=$(CUDA_TOOLKIT_ROOT)"
	@echo "CUDA_MATH_ROOT=$(CUDA_MATH_ROOT)"
	@echo "CUDA_TOOLKIT_LIBDIR=$(CUDA_TOOLKIT_LIBDIR)"
	@echo "CUDA_MATH_LIBDIR=$(CUDA_MATH_LIBDIR)"
	@echo "NVCC=$(NVCC)"
	@echo "CUDA_LINKER=$(CUDA_LINKER)"
	@echo "CUDA_NVJITLINK=$(if $(CUDA_NVJITLINK_SONAME),$(CUDA_NVJITLINK_SONAME),$(CUDA_NVJITLINK_VERSIONED))"
	@echo "CUDSS_PATH=$(CUDSS_PATH)"
	@echo "CUDSS_LIBDIR=$(CUDSS_LIBDIR)"

# Refresh the library before linking examples.  The archive lives outside this
# directory, so relying only on its timestamp easily links stale src objects.
.PHONY: libfasteig
libfasteig:
	$(MAKE) -C ../src MAKEINC=$(MAKEINC) USE_MKL="$(USE_MKL)" MKL="$(MKL)" USE_CUDA=$(USE_CUDA) USE_CUDSS=$(USE_CUDSS) CUDSS_PATH=$(CUDSS_PATH) CUDSS_LIBDIR=$(CUDSS_LIBDIR) CUDSS_LIB_NAME=$(CUDSS_LIB_NAME)

# Pattern rule for creating executables

runfasteig_cpu: runfasteig_cpu.o libfasteig
	$(EXAMPLE_LINKER) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(if $(filter 1,$(USE_CUDA)),$(CUDA_LIBS),) -o $@

runfasteig_gpu: runfasteig_gpu.o libfasteig
	$(EXAMPLE_LINKER) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(CUDA_LIBS) -o $@

MTX2FEI: MTX2FEI.o libfasteig
	$(EXAMPLE_LINKER) $< $(LIBFASTEIG) $(LDFLAGS) $(MKL) $(if $(filter 1,$(USE_CUDA)),$(CUDA_LIBS),) -o $@

runfasteig_cpu.o: runfasteig.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

runfasteig_gpu.o: runfasteig.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -DFASTEIG_GPU_EXEC -MMD -MP -c $< -o $@

# Pattern rule for compiling .cpp files into .o files
%.o: %.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $<

# Clean target
clean:
	rm -f runfasteig runfasteig_cpu runfasteig_gpu *.o *.d $(BUILD_TAG_FILE)

-include *.d

