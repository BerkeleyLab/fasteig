#include "version.hpp"

#include <iostream>
#include <string_view>

namespace {

int fail(const char* message) {
  std::cerr << "version_api: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  static_assert(FASTEIG_VERSION_EQ(1, 0, 0),
                "version macros must match the current release");
  static_assert(FASTEIG_VERSION_GE(1, 0, 0), "version comparison failed");
  static_assert(FASTEIG_VERSION_LT(2, 0, 0), "version comparison failed");

  const auto runtime_version = fasteig::version();
  const auto& info = fasteig::build_info();

  if (runtime_version.empty()) return fail("runtime version is empty");
  if (runtime_version != info.version) return fail("API version fields disagree");
  if (info.release_version != FASTEIG_VERSION_STRING) {
    return fail("runtime and preprocessor release versions disagree");
  }
  if (info.git_commit.empty()) return fail("Git revision is empty");
  if (info.build_system.empty()) return fail("build system is empty");
  if (info.configure_options.empty()) return fail("configure options are empty");
  if (info.cxx_compiler_id.empty()) return fail("compiler ID is empty");
  if (info.cxx_compiler_version.empty()) return fail("compiler version is empty");
  if (info.cxx_flags.empty()) return fail("compiler flags are empty");
  if (info.cxx_standard != "17") return fail("unexpected C++ standard");

  std::cout << runtime_version << '\n';
  return 0;
}

