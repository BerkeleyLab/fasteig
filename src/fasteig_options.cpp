#include "fasteig.hpp"
#include "fasteig_options.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>

// ==================================
// Implementation of options set/get
// ==================================

FASTEIG_INT fasteig_options::parse_int(std::string_view key, std::string_view v) {
  try {
    size_t pos = 0;
    int value = std::stoi(std::string(v), &pos);
    if (pos != v.size()) throw std::invalid_argument("");
    return static_cast<FASTEIG_INT>(value);
  } catch (...) {
    throw std::invalid_argument("Invalid integer for key '" + std::string(key) +
                                "': '" + std::string(v) + "'");
  }
}

FASTEIG_DOUBLE fasteig_options::parse_double(std::string_view key, std::string_view v) {
  try {
    size_t pos = 0;
    double value = std::stod(std::string(v), &pos);
    if (pos != v.size()) throw std::invalid_argument("");
    return static_cast<FASTEIG_DOUBLE>(value);
  } catch (...) {
    throw std::invalid_argument("Invalid double for key '" + std::string(key) +
                                "': '" + std::string(v) + "'");
  }
}

bool fasteig_options::parse_bool(std::string_view key, std::string_view v) {
  const std::string s = lower(v);
  if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
  if (s == "0" || s == "false" || s == "no"  || s == "off") return false;
  throw std::invalid_argument("Invalid boolean for key '" + std::string(key) +
                              "': '" + std::string(v) + "'");
}

bool fasteig_options::set(std::string_view key, std::string_view value) {
  const std::string k = lower(key);

  if (k == "n") {
    n_ = parse_int(key, value);
  } else if (k == "nev") {
    nev_ = parse_int(key, value);
  } else if (k == "maxiter") {
    maxiter_ = parse_int(key, value);
  } else if (k == "maxdim") {
    maxdim_ = parse_int(key, value);
  } else if (k == "maxblks") {
    maxblks_ = parse_int(key, value);
  } else if (k == "degree") {
    degree_ = parse_int(key, value);
  } else if (k == "which") {
    which_ = std::string(value);
  } else if (k == "solver") {
    solver_ = std::string(value);
  } else if (k == "shift") {
    shift_ = parse_double(key, value);
  } else if (k == "blksize") {
    blksize_ = parse_int(key, value);
  } else if (k == "tol") {
    tol_ = parse_double(key, value);
  } else if (k == "seed") {
    const long long s = std::stoll(std::string(value));
    if (s < 0) {
      seed_.reset();
    } else if (s > static_cast<long long>(std::numeric_limits<unsigned int>::max())) {
      throw std::invalid_argument("seed out of range for unsigned int");
    } else {
      seed_ = static_cast<unsigned int>(s);
    }
  //DMR
  } else if (k == "verbosity") {
    verbosity_ = static_cast<int>(parse_int(key, value));
    if (verbosity_ <= 1) verbosity_ = 1;
    if (verbosity_ >= 2) verbosity_ = 2;
  } else if (k == "print_every") {
    print_every_ = parse_int(key, value);
  } else if (k == "verbose") {
    (void)parse_bool(key, value);
    verbosity_ = 1;
  } else if (k == "logprefix" || k == "logfile") {
    logprefix_ = std::string(value);
  } else if (k == "output_prefix") {
    logprefix_ = std::string(value);
  } else if (k == "output_format") {
    output_format_ = std::string(value);
  //} else if (k == "verbose") {
  //  verbose_ = parse_bool(key, value);
  } else if (k == "logprefix" || k == "logfile") {
    logprefix_ = std::string(value);
  } else {
    return false;
  }

  return true;
}

std::string fasteig_options::get(std::string_view key) const {
  const std::string k = lower(key);

  if (k == "n") return std::to_string(n_);
  if (k == "nev") return std::to_string(nev_);
  if (k == "maxiter") return std::to_string(maxiter_);
  if (k == "maxdim") return std::to_string(maxdim_);
  if (k == "maxblks") return std::to_string(maxblks_);
  if (k == "degree") return std::to_string(degree_);
  if (k == "which") return which_;
  if (k == "solver") return solver_;
  if (k == "shift") return std::to_string(shift_);
  if (k == "blksize") return std::to_string(blksize_);
  if (k == "tol") return std::to_string(tol_);
  if (k == "seed") return seed_.has_value() ? std::to_string(*seed_) : "-1";
  //DMR
  if (k == "verbosity") return std::to_string(verbosity_);
  if (k == "print_every") return std::to_string(print_every_);
  if (k == "verbose") return (verbosity_ > 0) ? "true" : "false";
  //if (k == "verbose") return verbose_ ? "true" : "false";
  if (k == "logprefix" || k == "logfile") return logprefix_;
  //DMR
  if (k == "output_prefix") return logprefix_;
  if (k == "output_format") return output_format_;

  throw std::out_of_range("Unknown option key: " + std::string(key));
}

bool fasteig_options::apply(const std::unordered_map<std::string, std::string>& kv, bool strict) {
  for (const auto& [k, v] : kv) {
    const bool ok = set(k, v);
    if (!ok && strict) {
      throw std::invalid_argument("Unknown option key: " + k);
    }
  }
  return true;
}

void fasteig_options::configure_from(const fasteig_params& p) {
  // Treat parsed param as authoritative where provided.
  logprefix_ = p.logprefix;
  //DMR
  output_format_ = p.output_format;
  n_ = p.n;

  if (p.nev > 0)      nev_ = p.nev;
  if (p.maxiter > 0)  maxiter_ = p.maxiter;
  if (p.maxdim > 0)   maxdim_ = p.maxdim;
  if (p.degree > 0)   degree_ = p.degree;
  if (p.blksize > 0)  blksize_ = p.blksize;

  which_  = p.which;
  solver_ = p.solver;
  shift_  = p.shift;
  tol_    = p.tol;
  if (p.seed >= 0) seed_ = static_cast<unsigned int>(p.seed);

  //DMR
  if (p.verbosity >= 0) verbosity_ = p.verbosity;
  if (verbosity_ < 1) verbosity_ = 1;
  if (verbosity_ > 2) verbosity_ = 2;
  if (p.print_every > 0) print_every_ = p.print_every;

  // A reasonable default for Davidson if user didn't specify maxblks explicitly.
  if (maxblks_ <= 0) {
    maxblks_ = std::max<FASTEIG_INT>(10, nev_);
  }
}

void fasteig_options::validate() const {
  if (n_ < 0) throw std::invalid_argument("options.n must be >= 0");
  if (nev_ <= 0) throw std::invalid_argument("options.nev must be > 0");
  if (maxiter_ <= 0) throw std::invalid_argument("options.maxiter must be > 0");
  if (tol_ <= 0) throw std::invalid_argument("options.tol must be > 0");
  if (degree_ <= 0) throw std::invalid_argument("options.degree must be > 0");
  if (maxblks_ < 0) throw std::invalid_argument("options.maxblks must be >= 0");
  if (blksize_ <= 0) throw std::invalid_argument("options.blksize must be > 0");
  //DMR
  if (verbosity_ < 1 || verbosity_ > 2) throw std::invalid_argument("options.verbosity must be 1 or 2");
  if (print_every_ < 0) throw std::invalid_argument("options.print_every must be >= 0");
}

void fasteig_options::ensure_initial_vectors(FASTEIG_INT n, FASTEIG_INT blocksize) {
  if (n <= 0 || blocksize <= 0) {
    throw std::invalid_argument("ensure_initial_vectors: n and blocksize must be > 0");
  }

  if (v0_.size() == 0) {
    if (seed_.has_value()) {
      std::srand(static_cast<unsigned int>(*seed_));
    }
    v0_ = MatrixXd::Random(static_cast<int>(n), static_cast<int>(blocksize));
    return;
  }

  if (v0_.rows() != static_cast<int>(n) || v0_.cols() != static_cast<int>(blocksize)) {
    std::ostringstream oss;
    oss << "Initial vectors v0 have wrong shape. Expected "
        << n << " x " << blocksize << " but got "
        << v0_.rows() << " x " << v0_.cols();
    throw std::invalid_argument(oss.str());
  }
}

