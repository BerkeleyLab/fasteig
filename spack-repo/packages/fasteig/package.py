import os

from spack.package import *


class Fasteig(CMakePackage, CudaPackage):
    """Fast eigenvalue solver library."""

    git = "https://github.com/BerkeleyLab/fasteig"

    # Track real releases with annotated tags in the repo
    # (git tag v1.0.0 && git push --tags).
    version("main", branch="main", preferred=True)
    version("1.0.0", tag="v1.0.0")

    # CMakeLists.txt only accepts Debug/Release, not the full CMake set.
    variant(
        "build_type",
        default="Release",
        values=("Debug", "Release"),
        description="CMake build type",
    )

    variant("mkl", default=False, description="Use Intel oneAPI MKL for BLAS/LAPACK")
    variant("libsci", default=False, description="Use Cray LibSci for BLAS/LAPACK")
    # cuda / cuda_arch variants come from CudaPackage
    variant("cudss", default=False,
            description="Use the NVIDIA cuDSS sparse direct solver (requires +cuda)")
    variant("examples", default=True, description="Build example programs")
    variant("tests", default=False, description="Build and enable tests")
    variant(
        "arch_native",
        default=False,
        description="Let fasteig add -march=native (Spack already sets "
        "microarch flags via the compiler wrapper)",
    )
    variant(
        "cudss_root",
        default="none",
        values=any,
        description="Path to an existing cuDSS install, used when Spack "
        "has no 'cudss' package available",
    )

    conflicts("+mkl", when="+libsci", msg="USE_MKL and USE_LIBSCI are mutually exclusive")
    conflicts("+tests", when="~examples",
              msg="tests require +examples (runfasteig is the test driver)")
    conflicts("+cudss", when="~cuda", msg="cudss requires +cuda")

    # Languages used by the project (CMakeLists: LANGUAGES CXX C).
    # Spack 1.x needs these to attach a compiler such as %gcc.
    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("cmake@3.20:", type="build")
    depends_on("eigen@3.4", type="link")

    with when("~mkl ~libsci"):
        depends_on("blas")
        depends_on("lapack")

    # LP64 is mandatory (Eigen's Pardiso wrapper uses int), and the CMakeLists
    # links a threaded MKL layer, so the OpenMP flavor is required.
    depends_on("intel-oneapi-mkl ~ilp64 threads=openmp", when="+mkl")

    depends_on("cray-libsci", when="+libsci")

    depends_on("cuda@11:", when="+cuda")
    # If a "cudss" package becomes available, uncomment this and drop the
    # cudss_root fallback below.
    # depends_on("cudss", when="+cudss")

    def _oneapi_mkl_root(self):
        """Return the MKL component directory (<prefix>/mkl/<version>)."""
        mkl = self.spec["intel-oneapi-mkl"]

        root = getattr(mkl.package, "component_prefix", None)
        if root and os.path.isfile(join_path(str(root), "include", "mkl.h")):
            return str(root)

        candidates = [
            join_path(mkl.prefix, "mkl", str(mkl.version)),
            join_path(mkl.prefix, "mkl", "latest"),
            str(mkl.prefix),  # external pointing straight at MKLROOT
        ]
        for c in candidates:
            if os.path.isfile(join_path(c, "include", "mkl.h")):
                return c

        raise InstallError(
            "Could not locate the MKL component directory under {0}. "
            "Looked for include/mkl.h in: {1}".format(mkl.prefix, ", ".join(candidates))
        )

    def cmake_args(self):
        spec = self.spec

        args = [
            self.define("PLAT", "spack"),
            self.define_from_variant("ENABLE_CUDA", "cuda"),
            self.define_from_variant("USE_CUDSS", "cudss"),
            self.define_from_variant("BUILD_EXAMPLES", "examples"),
            self.define_from_variant("BUILD_TESTS", "tests"),
            self.define_from_variant("FASTEIG_ARCH_NATIVE", "arch_native"),
            self.define_from_variant("USE_MKL", "mkl"),
            self.define_from_variant("USE_LIBSCI", "libsci"),
            # A Spack eigen prefix matches the <prefix>/include/eigen3 layout
            # that the CMakeLists checks for.
            self.define("EIGEN_DIR", spec["eigen"].prefix),
        ]

        if spec.satisfies("+cuda"):
            archs = [a for a in spec.variants["cuda_arch"].value if a != "none"]
            if archs:
                args.append(self.define("CMAKE_CUDA_ARCHITECTURES", archs))

        if spec.satisfies("+mkl"):
            mklroot = self._oneapi_mkl_root()
            args.append(self.define("MKLROOT", mklroot))
            args.append(self.define("MKL_DIR", join_path(mklroot, "lib", "cmake", "mkl")))
        elif spec.satisfies("~mkl ~libsci"):
            args.append(self.define("FASTEIG_BLAS_LIBRARIES", spec["blas"].libs.ld_flags))
            args.append(self.define("FASTEIG_LAPACK_LIBRARIES", spec["lapack"].libs.ld_flags))
        # +libsci: the CMakeLists finds it via the Cray wrappers or pkg-config.

        if spec.satisfies("+cudss"):
            cudss_root = spec.variants["cudss_root"].value
            if "cudss" in spec:
                args.append(self.define("CUDSS_DIR", spec["cudss"].prefix))
            elif cudss_root != "none":
                args.append(self.define("CUDSS_DIR", cudss_root))
            else:
                raise InstallError(
                    "fasteig+cudss needs either a Spack 'cudss' dependency or "
                    "cudss_root=<path> pointing at an existing install."
                )

        return args
