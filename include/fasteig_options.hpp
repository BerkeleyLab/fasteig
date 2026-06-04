#pragma once

#include "fasteig.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>

/**
 * @brief Solver configuration for FASTEIG.
 *
 * Design intent:
 * - Strongly-typed fields for solver code
 * - Optional key/value string interface for config parsing and logging
 * - Lightweight (no ownership of the main problem matrix)
 */
class fasteig_options {
public:
  // -------- Defaults (reasonable, conservative) --------
  fasteig_options() = default;

  // Construct with explicit dimension (helps auto-initialize v0 sizing if desired)
  explicit fasteig_options(FASTEIG_INT n) : n_(n) {}

  // -------- Key/value interface (boundary layer) --------
  // Returns false if key is unknown; throws std::invalid_argument for invalid values.
  bool set(std::string_view key, std::string_view value);

  // Throws std::out_of_range if key is unknown.
  std::string get(std::string_view key) const;

  // Bulk configure from key/value map (unknown keys are ignored unless strict=true)
  bool apply(const std::unordered_map<std::string, std::string>& kv, bool strict = false);

  // -------- Typed interface (preferred in solver code) --------
  FASTEIG_INT n() const noexcept { return n_; }
  FASTEIG_INT nev() const noexcept { return nev_; }
  FASTEIG_INT maxiter() const noexcept { return maxiter_; }
  FASTEIG_INT maxdim() const noexcept { return maxdim_; }
  FASTEIG_INT maxblks() const noexcept { return maxblks_; }
  FASTEIG_INT degree() const noexcept { return degree_; }
  FASTEIG_INT blksize() const noexcept { return blksize_; }
  const std::string& which() const noexcept { return which_; }
  const std::string& solver() const noexcept { return solver_; }
  FASTEIG_DOUBLE shift() const noexcept { return shift_; }
  FASTEIG_DOUBLE tol() const noexcept { return tol_; }
  std::optional<unsigned int> seed() const noexcept { return seed_; }
  int verbosity() const noexcept { return verbosity_; }
  bool verbose() const noexcept { return verbosity_ > 0; }
  FASTEIG_INT print_every() const noexcept { return print_every_; }
  std::string logfile() const noexcept {return logprefix_;}
  std::string output_prefix() const noexcept { return logprefix_; }
  std::string output_format() const noexcept { return output_format_; }

  std::string output_path(std::string_view solver_name) const {
    if (logprefix_.empty()) return {};
    const std::string s = std::string(solver_name);
    return logprefix_ + "_" + s + ".log";
  }

  const MatrixXd& v0() const noexcept { return v0_; }
  MatrixXd& v0() noexcept { return v0_; }

  // -------- Semantic helpers --------

  /**
   * @brief Populate options from a parsed fasteig_params (common driver path).
   *
   * This is the intended "bridge" between input decks and solver options.
   * It does not take ownership of param data.
   */
  void configure_from(const fasteig_params& p);

  /**
   * @brief Ensure internal consistency.
   *
   * Throws std::invalid_argument on invalid configuration.
   */
  void validate() const;

  /**
   * @brief Ensure v0 has the expected shape; if empty, initialize randomly.
   *
   * If v0 is empty (0x0) -> initializes random n x blocksize.
   * If v0 has wrong shape -> throws.
   */
  void ensure_initial_vectors(FASTEIG_INT n, FASTEIG_INT blocksize);

private:
  // -------- Internal state --------
  std::string logprefix_ = "logfile";
  std::string output_format_ = "text";
  FASTEIG_INT n_ = 0;

  FASTEIG_INT nev_ = 10;
  FASTEIG_INT maxiter_ = 1000;
  FASTEIG_INT maxdim_ = 0;     // 0 => let algorithm decide / unused in some solvers
  FASTEIG_INT maxblks_ = 10;   // Davidson uses this
  FASTEIG_INT degree_ = 10;    // Chebyshev uses this
  FASTEIG_INT blksize_ = 1;

  std::string which_ = "LM";
  std::string solver_ = "Lanczos";

  FASTEIG_DOUBLE shift_ = 0.0;
  FASTEIG_DOUBLE tol_ = 1.0e-8;
  std::optional<unsigned int> seed_;

  int verbosity_ = 1;
  FASTEIG_INT print_every_ = 0;

  MatrixXd v0_; // optional initial block vectors (typically n x blocksize)

  // -------- Parsing helpers --------
  static std::string lower(std::string_view s) {
    std::string out(s.begin(), s.end());
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return out;
  }

  static FASTEIG_INT parse_int(std::string_view key, std::string_view v);
  static FASTEIG_DOUBLE parse_double(std::string_view key, std::string_view v);
  static bool parse_bool(std::string_view key, std::string_view v);
};

