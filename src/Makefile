MAKEINC ?= ../make.inc
include $(MAKEINC)
# Notes
# -----
# To build the library WITHOUT MKL/Pardiso (use Eigen's native sparse LDLT), override `USE_MKL`:
#   make -C src MAKEINC=../make.inc clean
#   make -C src MAKEINC=../make.inc USE_MKL=""
# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag
.PHONY: force_build_tag
$(BUILD_TAG_FILE): force_build_tag
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; echo "USE_CUDA=$(USE_CUDA)"; echo "CUDA_PATH=$(CUDA_PATH)"; echo "USE_CUDSS=$(USE_CUDSS)"; echo "CUDSS_PATH=$(CUDSS_PATH)"; echo "CUDSS_LIBDIR=$(CUDSS_LIBDIR)"; echo "CUDSS_LIB_NAME=$(CUDSS_LIB_NAME)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@
COMMON_OBJS = core/Executor.o core/Vector.o core/MultiVector.o core/CsrMatrix.o core/Operations.o core/LegacyOperationsOperator.o core/LanczosWorkspace.o core/KrylovSubspace.o core/Orthogonalization.o core/RitzExtraction.o mmio.o rmmdiis.o SPPC_Ind.o parseinput.o fasteig_solver.o LanczosSolver.o BlockLanczosSolver.o BlockKrylovSchurSolver.o KrylovSchurSolver.o DavidsonSolver.o LOBPCGSolver.o ChebyshevSolver.o SPPCSolver.o RMMDIISSolver.o fasteig_options.o fasteig_problem.o fasteig_logger.o
CUDA_OBJS =
ifeq ($(USE_CUDA),1)
CUDA_OBJS += gpu/CudaExecutor.o gpu/CudaVector.o gpu/CudaMultiVector.o gpu/CudaCsrMatrix.o gpu/CudaContext.o gpu/CudaLanczosWorkspace.o gpu/CudaLanczosKernels.o
ifeq ($(USE_CUDSS),1)
CUDA_OBJS += gpu/CudaDssSolver.o gpu/CudaShiftInvertOperator.o
endif
endif
OBJS = $(COMMON_OBJS) $(CUDA_OBJS)
# Auto-generate header dependencies (.d files) so edits to headers trigger rebuilds.
DEPFLAGS ?= -MMD -MP
%.o: %.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@
%.o: %.cu $(BUILD_TAG_FILE)
	$(NVCC) $(CUDA_NVCCFLAGS) $(DEPFLAGS) -c $< -o $@
all: $(OBJS)
	rm -f $(LIBFASTEIG)
	$(AR) $(ARFLAGS) $(LIBFASTEIG) $(OBJS)
	$(RANLIB) $(LIBFASTEIG)
clean:
	rm -f *.o *.d core/*.o core/*.d gpu/*.o gpu/*.d a.out $(BUILD_TAG_FILE)
-include $(OBJS:.o=.d)

