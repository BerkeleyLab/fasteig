MAKEINC ?= ../make.inc
include $(MAKEINC)

# Force recompilation when the selected toolchain/config changes (e.g. Perlmutter -> Mac).
BUILD_TAG_FILE = .fasteig_build_tag

$(BUILD_TAG_FILE):
	@{ echo "PLAT=$(PLAT)"; echo "CXX=$(CXX)"; echo "CXXFLAGS=$(CXXFLAGS)"; } > $@.tmp
	@cmp -s $@ $@.tmp 2>/dev/null && rm -f $@.tmp || mv $@.tmp $@

#DMR
OBJS  = KrylovSubspace.o mmio.o getevbounds.o chebfsit.o rmmdiis.o file_reader.o SPPC_Ind.o CholeskyQR.o Display2Vecs.o parseinput.o CreateMatrix.o fasteig_solver.o LanczosSolver.o BlockLanczosSolver.o KrylovSchurSolver.o DavidsonSolver.o LOBPCGSolver.o ChebyshevSolver.o SPPCSolver.o RMMDIISSolver.o fasteig_options.o fasteig_problem.o fasteig_logger.o ortho.o ReadMatrix.o

#OBJS  = KrylovSubspace.o krylov_schur.o mmio.o getevbounds.o chebfsit.o rmmdiis.o file_reader.o SPPC_Ind.o CholeskyQR.o Display2Vecs.o parseinput.o CreateMatrix.o fasteig_solver.o LanczosSolver.o fasteig_options.o LOBPCG.o ortho.o

# Auto-generate header dependencies (.d files) so edits to headers trigger rebuilds.
DEPFLAGS ?= -MMD -MP

%.o: %.cpp $(BUILD_TAG_FILE)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $<


all: $(OBJS)
	$(AR) $(ARFLAGS) $(LIBFASTEIG) $(OBJS)
	$(RANLIB) $(LIBFASTEIG)

clean:
	rm -f *.o *.d a.out core

-include $(OBJS:.o=.d)
		
