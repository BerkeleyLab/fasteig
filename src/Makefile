MAKEINC ?= ../make.inc
include $(MAKEINC)

.DEFAULT_GOAL := all

# Notes
# -----
# To build the library WITHOUT MKL/Pardiso (use Eigen's native sparse LDLT), override `USE_MKL`:
#   make -C src MAKEINC=../make.inc clean
#   make -C src MAKEINC=../make.inc USE_MKL=""

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag
GENERATED_DIR = .fasteig-generated
BUILD_CONFIG_HEADER = $(GENERATED_DIR)/fasteig_build_config.hpp
.PHONY: force_build_tag force_build_config

$(BUILD_TAG_FILE): force_build_tag
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; echo "USE_CUDA=$(USE_CUDA)"; echo "CUDA_PATH=$(CUDA_PATH)"; echo "CUDA_TOOLKIT_ROOT=$(CUDA_TOOLKIT_ROOT)"; echo "CUDA_MATH_ROOT=$(CUDA_MATH_ROOT)"; echo "CUDA_TOOLKIT_LIBDIR=$(CUDA_TOOLKIT_LIBDIR)"; echo "CUDA_MATH_LIBDIR=$(CUDA_MATH_LIBDIR)"; echo "USE_CUDSS=$(USE_CUDSS)"; echo "CUDSS_PATH=$(CUDSS_PATH)"; echo "CUDSS_LIBDIR=$(CUDSS_LIBDIR)"; echo "CUDSS_LIB_NAME=$(CUDSS_LIB_NAME)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

# Embed the exact Make configuration and Git revision in fasteig_version.o.
# The generated header is private and is rebuilt only when its contents change.
$(BUILD_CONFIG_HEADER): force_build_config $(BUILD_TAG_FILE)
	@mkdir -p $(GENERATED_DIR)
	@git_commit="$$(git -C .. rev-parse --verify HEAD 2>/dev/null || true)"; \
	git_dirty=0; \
	if test -n "$$(git -C .. status --porcelain 2>/dev/null)"; then git_dirty=1; fi; \
	if test -z "$$git_commit"; then git_commit=unknown; fi; \
	build_version="1.0.0"; \
	if test "$$git_commit" != unknown; then \
	  git_short="$$(printf '%s' "$$git_commit" | cut -c1-12)"; \
	  build_version="$$build_version+git.$$git_short"; \
	  if test "$$git_dirty" = 1; then build_version="$$build_version.dirty"; fi; \
	fi; \
	build_type=Custom; \
	case " $(CXXFLAGS) " in *" -DNDEBUG "*) build_type=Release;; *" -g "*) build_type=Debug;; esac; \
	compiler_version="$$( $(CXX) --version 2>/dev/null | head -n 1 )"; \
	if test -z "$$compiler_version"; then compiler_version=unknown; fi; \
	cuda_compiler="not enabled"; cuda_version="not enabled"; cuda_flags="not enabled"; cuda_arch="not enabled"; \
	if test "$(USE_CUDA)" = 1; then \
	  cuda_compiler="$(NVCC)"; \
	  cuda_version="$$( $(NVCC) --version 2>/dev/null | tail -n 1 )"; \
	  cuda_flags="$(CUDA_NVCCFLAGS)"; cuda_arch="$(CUDA_ARCH)"; \
	fi; \
	{ \
	  printf '%s\n' '#pragma once'; \
	  printf '#define FASTEIG_BUILD_VERSION R"fasteig(%s)fasteig"\n' "$$build_version"; \
	  printf '%s\n' '#define FASTEIG_BUILD_RELEASE_VERSION "1.0.0"'; \
	  printf '#define FASTEIG_BUILD_GIT_COMMIT R"fasteig(%s)fasteig"\n' "$$git_commit"; \
	  printf '#define FASTEIG_BUILD_GIT_DIRTY %s\n' "$$git_dirty"; \
	  printf '%s\n' '#define FASTEIG_BUILD_SYSTEM "Make"'; \
	  printf '#define FASTEIG_BUILD_TYPE R"fasteig(%s)fasteig"\n' "$$build_type"; \
	  printf '#define FASTEIG_BUILD_CONFIGURE_OPTIONS R"fasteig(%s)fasteig"\n' 'MAKEINC=$(MAKEINC);PLAT=$(PLAT);USE_MKL=$(USE_MKL);USE_LIBSCI=$(USE_LIBSCI);USE_CUDA=$(USE_CUDA);USE_CUDSS=$(USE_CUDSS)'; \
	  printf '%s\n' '#define FASTEIG_BUILD_CMAKE_VERSION "not used"'; \
	  printf '#define FASTEIG_BUILD_CMAKE_GENERATOR R"fasteig(make.inc: %s)fasteig"\n' '$(MAKEINC)'; \
	  printf '#define FASTEIG_BUILD_CXX_COMPILER_VERSION R"fasteig(%s)fasteig"\n' "$$compiler_version"; \
	  printf '#define FASTEIG_BUILD_CXX_COMPILER_PATH R"fasteig(%s)fasteig"\n' '$(CXX)'; \
	  printf '#define FASTEIG_BUILD_CXX_FLAGS R"fasteig(%s)fasteig"\n' '$(CXXFLAGS)'; \
	  printf '%s\n' '#define FASTEIG_BUILD_CXX_STANDARD "17"'; \
	  printf '#define FASTEIG_BUILD_PLATFORM R"fasteig(%s)fasteig"\n' '$(PLAT)'; \
	  printf '#define FASTEIG_BUILD_USE_MKL %s\n' '$(if $(strip $(USE_MKL)),1,0)'; \
	  printf '#define FASTEIG_BUILD_USE_LIBSCI %s\n' '$(if $(strip $(USE_LIBSCI)),1,0)'; \
	  printf '#define FASTEIG_BUILD_ENABLE_CUDA %s\n' '$(if $(filter 1,$(USE_CUDA)),1,0)'; \
	  printf '#define FASTEIG_BUILD_USE_CUDSS %s\n' '$(if $(and $(filter 1,$(USE_CUDA)),$(filter 1,$(USE_CUDSS))),1,0)'; \
	  printf '#define FASTEIG_BUILD_CUDA_COMPILER R"fasteig(%s)fasteig"\n' "$$cuda_compiler"; \
	  printf '#define FASTEIG_BUILD_CUDA_VERSION R"fasteig(%s)fasteig"\n' "$$cuda_version"; \
	  printf '#define FASTEIG_BUILD_CUDA_FLAGS R"fasteig(%s)fasteig"\n' "$$cuda_flags"; \
	  printf '#define FASTEIG_BUILD_CUDA_ARCHITECTURES R"fasteig(%s)fasteig"\n' "$$cuda_arch"; \
	} > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

COMMON_OBJS = core/Executor.o core/Vector.o core/MultiVector.o core/CsrMatrix.o core/Operations.o core/LegacyOperationsOperator.o core/LanczosWorkspace.o core/KrylovSubspace.o core/Orthogonalization.o core/RitzExtraction.o mmio.o rmmdiis.o SPPC_Ind.o parseinput.o fasteig_solver.o LanczosSolver.o BlockLanczosSolver.o BlockKrylovSchurSolver.o KrylovSchurSolver.o DavidsonSolver.o LOBPCGSolver.o ChebyshevSolver.o SPPCSolver.o RMMDIISSolver.o fasteig_options.o fasteig_problem.o fasteig_logger.o fasteig_version.o

COMMON_OBJS += core/LinearOperator.o

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

fasteig_version.o: fasteig_version.cpp $(BUILD_TAG_FILE) $(BUILD_CONFIG_HEADER)
	$(CXX) $(CXXFLAGS) -I$(GENERATED_DIR) $(DEPFLAGS) -c $< -o $@

%.o: %.cu $(BUILD_TAG_FILE)
	$(NVCC) $(CUDA_NVCCFLAGS) $(DEPFLAGS) -c $< -o $@

all: $(OBJS)
	rm -f $(LIBFASTEIG)
	$(AR) $(ARFLAGS) $(LIBFASTEIG) $(OBJS)
	$(RANLIB) $(LIBFASTEIG)

clean:
	rm -f *.o *.d core/*.o core/*.d gpu/*.o gpu/*.d a.out $(BUILD_TAG_FILE) $(BUILD_CONFIG_HEADER) $(BUILD_CONFIG_HEADER).tmp
	rmdir $(GENERATED_DIR) 2>/dev/null || true

-include $(OBJS:.o=.d)

