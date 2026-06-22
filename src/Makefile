MAKEINC ?= ../make.inc
include $(MAKEINC)

# Notes
# -----
# To build the library WITHOUT MKL/Pardiso (use Eigen's native sparse LDLT), override `USE_MKL`:
#   make -C src MAKEINC=../make.inc clean
#   make -C src MAKEINC=../make.inc USE_MKL=""

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag

$(BUILD_TAG_FILE):
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

COMMON_OBJS = KrylovSubspace.o mmio.o rmmdiis.o SPPC_Ind.o parseinput.o fasteig_solver.o LanczosSolver.o BlockLanczosSolver.o KrylovSchurSolver.o DavidsonSolver.o LOBPCGSolver.o ChebyshevSolver.o SPPCSolver.o RMMDIISSolver.o fasteig_options.o fasteig_problem.o fasteig_logger.o

CUDA_OBJS =
ifeq ($(USE_CUDA),1)
CUDA_OBJS += gpu/GpuLanczosBackend.o gpu/GpuKrylovSubspace.o gpu/GpuContext.o gpu/GpuCsrMatrix.o gpu/GpuLanczosKernels.o
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
	rm -f *.o *.d gpu/*.o gpu/*.d a.out core

-include $(OBJS:.o=.d)
