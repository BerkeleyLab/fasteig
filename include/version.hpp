#pragma once

#include <string_view>

// Release-version macros are available without linking the library.  Keep these
// values synchronized with project(... VERSION ...) in the top-level
// CMakeLists.txt when preparing a release.
#define FASTEIG_VERSION_MAJOR 1
#define FASTEIG_VERSION_MINOR 0
#define FASTEIG_VERSION_PATCH 0
#define FASTEIG_VERSION_STRING "1.0.0"

#define FASTEIG_VERSION_ENCODE(major, minor, patch) \
  ((major) * 1000000 + (minor) * 1000 + (patch))
#define FASTEIG_VERSION \
  FASTEIG_VERSION_ENCODE(FASTEIG_VERSION_MAJOR, FASTEIG_VERSION_MINOR, \
                         FASTEIG_VERSION_PATCH)

#define FASTEIG_VERSION_EQ(major, minor, patch) \
  (FASTEIG_VERSION == FASTEIG_VERSION_ENCODE((major), (minor), (patch)))
#define FASTEIG_VERSION_NE(major, minor, patch) \
  (FASTEIG_VERSION != FASTEIG_VERSION_ENCODE((major), (minor), (patch)))
#define FASTEIG_VERSION_LT(major, minor, patch) \
  (FASTEIG_VERSION < FASTEIG_VERSION_ENCODE((major), (minor), (patch)))
#define FASTEIG_VERSION_LE(major, minor, patch) \
  (FASTEIG_VERSION <= FASTEIG_VERSION_ENCODE((major), (minor), (patch)))
#define FASTEIG_VERSION_GT(major, minor, patch) \
  (FASTEIG_VERSION > FASTEIG_VERSION_ENCODE((major), (minor), (patch)))
#define FASTEIG_VERSION_GE(major, minor, patch) \
  (FASTEIG_VERSION >= FASTEIG_VERSION_ENCODE((major), (minor), (patch)))

namespace fasteig {

/**
 * Build-time provenance embedded in the fasteig library.
 *
 * String views refer to static storage owned by the library. They remain valid
 * for the lifetime of the process and require no caller-managed allocation.
 */
struct BuildInfo {
  std::string_view version;
  std::string_view release_version;
  std::string_view git_commit;
  bool git_dirty;

  std::string_view build_system;
  std::string_view build_type;
  std::string_view configure_options;
  std::string_view cmake_version;
  std::string_view cmake_generator;

  std::string_view cxx_compiler_id;
  std::string_view cxx_compiler_version;
  std::string_view cxx_compiler_path;
  std::string_view cxx_flags;
  std::string_view cxx_standard;
  std::string_view platform;

  bool mkl_enabled;
  bool libsci_enabled;
  bool cuda_enabled;
  bool cudss_enabled;

  std::string_view cuda_compiler;
  std::string_view cuda_version;
  std::string_view cuda_flags;
  std::string_view cuda_architectures;
};

/** Return the release version plus Git revision for development builds. */
std::string_view version() noexcept;

/** Return complete build and compiler provenance for this library binary. */
const BuildInfo& build_info() noexcept;

}  // namespace fasteig

