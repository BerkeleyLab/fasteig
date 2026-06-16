#include "fasteig.hpp"
#include "fasteig_options.hpp"
#include "fasteig_solver.hpp"

#include "BlockLanczosSolver.hpp"
#include "ChebyshevSolver.hpp"
#include "DavidsonSolver.hpp"
#include "KrylovSchurSolver.hpp"
#include "LanczosSolver.hpp"
#include "LOBPCGSolver.hpp"
#include "RMMDIISSolver.hpp"
#include "SPPCSolver.hpp"

#include "fasteig_problem.hpp"

namespace fasteig {
namespace {

bool has_prefix(const std::string& s, const char* prefix, size_t n) {
  return s.size() >= n && std::strncmp(s.c_str(), prefix, n) == 0;
}

std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return s;
}

MatrixXd apply_block_matvec_fallback(const operations& ops, const Eigen::Ref<const MatrixXd>& V) {
  MatrixXd AV(V.rows(), V.cols());
  if (ops.block_matvec) {
    ops.block_matvec(V, AV);
    return AV;
  }
  if (!ops.matvec) {
    throw std::invalid_argument("RayleighRitz: ops has no matvec or block_matvec.");
  }
  VectorXd av(V.rows());
  for (int j = 0; j < V.cols(); ++j) {
    ops.matvec(V.col(j), av);
    AV.col(j) = av;
  }
  return AV;
}

}  // namespace

fasteig_solver::fasteig_solver(::fasteig_problem& problem,
                               const fasteig_options& options)
    : problem_(problem),
      opsA_(problem.opsA()),
      opsB_(problem.opsB()),
      opsSAI_(problem.opsSAI()),
      eigvals_(problem.eigenvalues()),
      resnrms_(problem.residual_norms()),
      eigvecs_(problem.eigenvectors()),
      options_(options) {}

const operations& fasteig_solver::active_ops() const noexcept {
  return problem_.solver_ops(options_);
}

void fasteig_solver::compute_final_residual_norms() {
  resnrms_ = compute_residual_norms(opsA_, eigvals_, eigvecs_);
}

VectorXd fasteig_solver::compute_residual_norms(const operations& ops,
                                                const VectorXd& eigvals,
                                                const MatrixXd& eigvecs) {
  return compute_residual_norms(ops, /*opsB=*/nullptr, eigvals, eigvecs);
}

VectorXd fasteig_solver::compute_residual_norms(const operations& opsA,
                                                const operations* opsB,
                                                const VectorXd& eigvals,
                                                const MatrixXd& eigvecs) {
  const int nev = static_cast<int>(eigvals.size());
  if (eigvecs.cols() < nev) {
    throw std::invalid_argument("compute_residual_norms: eigvecs has fewer columns than eigvals.");
  }
  if (!opsA.matvec && !opsA.block_matvec) {
    throw std::invalid_argument("compute_residual_norms: opsA has no matvec or block_matvec.");
  }

  VectorXd residuals(nev);
  VectorXd av(eigvecs.rows());
  VectorXd bv(eigvecs.rows());

  for (int i = 0; i < nev; ++i) {
    const auto v = eigvecs.col(i);
    if (opsA.matvec) {
      opsA.matvec(v, av);
    } else {
      MatrixXd V(v.rows(), 1);
      V.col(0) = v;
      MatrixXd AV(v.rows(), 1);
      opsA.block_matvec(V, AV);
      av = AV.col(0);
    }

    if (opsB && opsB->matvec) {
      opsB->matvec(v, bv);
      const double num = (av - eigvals(i) * bv).norm();
      const double denom = (1.0 + std::abs(eigvals(i))) * bv.norm();
      residuals(i) = (denom > 0.0) ? (num / denom) : num;
    } else {
      const double num = (av - eigvals(i) * v).norm();
      const double denom = (1.0 + std::abs(eigvals(i))) * v.norm();
      residuals(i) = (denom > 0.0) ? (num / denom) : num;
    }
  }
  return residuals;
}

FASTEIG_INT fasteig_solver::CholeskyQR(MatrixXd& X, MatrixXd& Q, MatrixXd& R) {
  const int ncols = X.cols();

  Eigen::MatrixXd Gmat = X.transpose() * X;
  Eigen::LLT<Eigen::MatrixXd> llt(Gmat);
  if (llt.info() != Eigen::Success) {
    std::cerr << "CholeskyQR: Gram matrix is not positive definite." << std::endl;
    return -1;
  }

  const Eigen::MatrixXd L = llt.matrixL();
  R = L.transpose();
  Q = X * R.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(ncols, ncols));
  return 0;
}

std::unique_ptr<fasteig_solver> make_fasteig_solver(const std::string& solver,
                                                    ::fasteig_problem& problem,
                                                    const fasteig_options& options) {
  if (has_prefix(solver, "Lanczos", 3)) {
    return std::make_unique<LanczosSolver>(problem, options);
  }
  if (has_prefix(solver, "BlockLanczos", 8)) {
    return std::make_unique<BlockLanczosSolver>(problem, options);
  }
  if (has_prefix(solver, "KrylovSchur", 7)) {
    return std::make_unique<KrylovSchurSolver>(problem, options);
  }
  if (has_prefix(solver, "Davidson", 3)) {
    return std::make_unique<DavidsonSolver>(problem, options);
  }
  if (has_prefix(solver, "LOBPCG", 3)) {
    return std::make_unique<LOBPCGSolver>(problem, options);
  }
  if (has_prefix(solver, "Chebyshev", 3)) {
    return std::make_unique<ChebyshevSolver>(problem, options);
  }
  if (has_prefix(solver, "SPPC", 3)) {
    return std::make_unique<SPPCSolver>(problem, options);
  }
  if (has_prefix(solver, "RMM-DIIS", 3)) {
    return std::make_unique<RMMDIISSolver>(problem, options);
  }
  if (has_prefix(solver, "auto", 3)) {
    return nullptr;
  }

  std::cerr << "Must choose a valid solver (Lanczos, BlockLanczos, KrylovSchur, Davidson, "
               "LOBPCG, Chebyshev, SPPC, RMM-DIIS or auto)\n";
  return nullptr;
}

void fasteig_solver::RayleighRitz(const operations &ops,
                                  MatrixXd &V,
                                  VectorXd &Ritzvals,
                                  MatrixXd &Ritzvecs,
                                  MatrixXd &eigenvectors,
                                  MatrixXd &residuals) {
  MatrixXd AV = V;
  int ncols = 0;

  ops.block_matvec(V, AV);
  Eigen::MatrixXd Gmat = V.transpose() * AV;
  Gmat = (Gmat.transpose() + Gmat) / 2.0;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Gmat);
  eigenvectors = solver.eigenvectors();

  Ritzvecs = V * eigenvectors;
  Ritzvals = solver.eigenvalues();

  residuals = AV * eigenvectors;

  ncols = residuals.cols();
  for (size_t j = 0; j < static_cast<size_t>(ncols); ++j) {
    residuals.col(static_cast<int>(j)) =
        residuals.col(static_cast<int>(j)) - Ritzvecs.col(static_cast<int>(j)) * Ritzvals[static_cast<int>(j)];
  }
}

void fasteig_solver::RayleighRitz(const operations& ops,
                                  const Eigen::Ref<const MatrixXd>& V,
                                  FASTEIG_SIZET nev,
                                  std::string which,
                                  VectorXd& Ritzvals,
                                  MatrixXd& Ritzvecs,
                                  MatrixXd& residuals) {
  if (nev == 0) throw std::invalid_argument("RayleighRitz: nev must be > 0.");
  if (V.cols() == 0) throw std::invalid_argument("RayleighRitz: V must have at least 1 column.");

  const int m = V.cols();
  const int n = V.rows();
  const int nev_i = static_cast<int>(std::min<FASTEIG_SIZET>(nev, static_cast<FASTEIG_SIZET>(m)));

  MatrixXd AV = apply_block_matvec_fallback(ops, V);
  Eigen::MatrixXd Gmat = V.transpose() * AV;
  Gmat = (Gmat.transpose() + Gmat) / 2.0;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Gmat);
  const auto D = solver.eigenvalues();
  const auto C = solver.eigenvectors();

  std::vector<size_t> Ind(m, 0);
  const std::string sigma = lower_copy(std::move(which));
  if (sigma == "be") {
    int pos = 0;
    for (int i = 0; i < m / 2; ++i) {
      Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(m - 1 - i);
      Ind[static_cast<size_t>(pos++)] = static_cast<size_t>(i);
    }
    if (m % 2 == 1) {
      Ind[static_cast<size_t>(pos)] = static_cast<size_t>(m / 2);
    }
  } else {
    Ind = sort_indices(D, sigma);
  }

  MatrixXd Csel(m, nev_i);
  for (int j = 0; j < nev_i; ++j) {
    Csel.col(j) = C.col(static_cast<int>(Ind[static_cast<size_t>(j)]));
  }

  Ritzvals.resize(nev_i);
  for (int j = 0; j < nev_i; ++j) {
    Ritzvals(j) = D(static_cast<int>(Ind[static_cast<size_t>(j)]));
  }

  Ritzvecs.noalias() = V * Csel;
  residuals.noalias() = AV * Csel;
  for (int j = 0; j < nev_i; ++j) {
    residuals.col(j).noalias() -= Ritzvecs.col(j) * Ritzvals(j);
  }

  (void)n;
}

}  // namespace fasteig

