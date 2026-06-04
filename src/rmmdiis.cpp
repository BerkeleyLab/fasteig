//********************************************************************************
//
//  Name
//    rmm_diis
//
//  Description
//    The Residual Minimization Method with Direct Inversion of the Iterative 
//    Subspace (RMM-DIIS) refines an approximate eigenpair of a symmetric matrix. 
//    This method iteratively minimizes the residual error while accelerating 
//    convergence through subspace extrapolation.
//
//  Arguments
//    v0     Double precision, Eigen::VectorXd of size n. (INPUT)
//           On INPUT:
//           - Initial approximation to the eigenvector.
//           - Must be nonzero and normalized before calling the function.
//           On OUTPUT:
//           - Not modified.
//
//    ops    Structure. (INPUT)
//           A structure containing matrix operations, including:
//           - `matvec(v, Av)`: A user-defined matrix-vector multiplication 
//              function that computes Av = A * v.
//           - `CG(r, shift, w)`: A user-defined preconditioned conjugate gradient
//              solver that solves (R - shift * I) w = r.
//
//    max_it size_t. (INPUT)
//           The maximum number of allowed iterations.
//
//    eigvec Double precision, Eigen::VectorXd of size n. (OUTPUT)
//           On OUTPUT:
//           - The computed approximate eigenvector.
//
//    eigval Double precision. (OUTPUT)
//           On OUTPUT:
//           - The computed approximate eigenvalue.
//
//    tol    Double precision. (INPUT)
//           Convergence tolerance for the relative residual norm.
//           The iteration stops when ||Av - λv|| / |λ| < tol.
//
//  Exceptions
//    Throws std::runtime_error if:
//    - `v0.size() == 0`, indicating an invalid input vector.
//    - `max_it == 0`, meaning no iterations are allowed.
//
//  Algorithm
//    1. Normalize the input vector and compute its Rayleigh quotient as an 
//       initial eigenvalue approximation.
//    2. Compute the residual r = Av - θv and apply a preconditioner to obtain w1.
//    3. Iteratively refine the eigenvector by extrapolating from previous 
//       iterates using DIIS acceleration.
//    4. Construct a two-dimensional generalized eigenvalue problem to update 
//       the eigenpair.
//    5. Check convergence and terminate when the relative residual norm is 
//       below the given tolerance.
//
//  References
//    1. Pulay, P. (1980). "Convergence acceleration of iterative sequences. 
//       The case of SCF iteration." *Chemical Physics Letters*, 73(2), 393-398.
//    2. Kresse, G., & Furthmüller, J. (1996). "Efficient iterative schemes for 
//       ab initio total-energy calculations using a plane-wave basis set." 
//       *Physical Review B*, 54(16), 11169.
//
//  Authors
//    Dong Min Roh
//    Chao Yang
//    Applied Mathematics and Computational Research Division 
//    Lawrence Berkeley National Laboratory
//
//********************************************************************************

#include "fasteig.hpp"

void rmm_diis( const VectorXd v0,
	       const operations& ops,
	       size_t max_it,
	       VectorXd& eigvec,
	       double& eigval,
	       double tol ) {

  const size_t n = v0.size();                                                                                                                 
  std::cout << std::endl;
  std::cout << "***** Starting RMM-DIIS Algorithm *****" << std::endl;
  std::cout << std::endl;
  std::cout << "Refining an eigenpair of size " << n << " symmetric real matrix/matrices" << std::endl;
  std::cout << "Maximum number of iterations: " << max_it << std::endl;
  std::cout << "Tolerance: " << tol << std::endl;
  std::cout << std::endl;

  VectorXd v(n), Av(n), r(n), w(n), Aw(n), w1(n);
  double theta, resnrm, shift;

  MatrixXd V(n, max_it+1), AV(n, max_it+1), W(n, max_it);
  MatrixXd Q(n, 2), AQ(n, 2);

  // Initial computation
  v = v0.normalized();
  ops.matvec(v, Av);
  theta = v.dot(Av);
  r = Av - theta * v;
  resnrm = r.norm() / abs(theta);

  shift = theta - 2 * resnrm * r.norm() * 1e-2;
  ops.CG(r, shift, w1);

  V.col(0) = v;
  AV.col(0) = Av;

  std::cout << "Iteration " << 0 << std::endl;
  std::cout << "Eigenvalue = " << theta << ", Residual = " << resnrm << std::endl;
  
  if ( resnrm < tol ) {
    eigvec = v;
    eigval = theta;
    std::cout << "Found eigenvalue = " << theta << " after " << 0 << " iterations. " << std::endl;
    return;
  }

  for ( size_t i = 0; i < max_it; ++i ) {
	  
    if ( i > 0 ) {
 
      shift = theta - 2 * resnrm * r.norm() * 1e-2;
      ops.CG(r, shift, w);

      W.col(i-1) = w1 - w;
      MatrixXd W_mat = W.leftCols(i);
      VectorXd a = W_mat.householderQr().solve(w1);
      double a1 = 1 - a.sum();
      auto V_sub = V.block(0, 1, n, i);
      v = a1 * V.col(0) + V.middleCols(1, i) * a;
      auto nrmv = v.norm();
      v = v / nrmv;
      Av = a1 * AV.col(0) / nrmv + AV.middleCols(1, i) * a / nrmv;
      theta = v.dot(Av);

    }
    
    r = Av - theta * v;
    
    // Preconditioning
    shift = theta - 2 * r.norm() / abs(theta) * r.norm() * 1e-2;
    ops.CG(r, shift, w);
	  
    Q.col(0) = v;
    Q.col(1) = w;
    AQ.col(0) = Av;
    ops.matvec(w, Aw);
    AQ.col(1) = Aw;

    MatrixXd T = Q.transpose() * AQ;
    MatrixXd G = Q.transpose() * Q;

    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(T, G);
    auto D = solver.eigenvalues();
    auto S = solver.eigenvectors();

    v = Q * S.col(0);
    theta = D(0);
    Av = AQ * S.col(0);

    V.col(i+1) = v;
    AV.col(i+1) = Av;

    r = Av - theta * v;
    resnrm = r.norm() / abs(theta);
    
    std::cout << "Iteration " << i+1 << std::endl;
    std::cout << "Eigenvalue = " << theta << ", Residual = " << resnrm << std::endl;

    // Stopping Criteria
    if ( resnrm < tol ) {
      eigvec = v;
      eigval = theta;
      return;
    }

  }

  eigvec = v;
  eigval = theta;
  std::cout << "Did not converge within " << max_it << " iterations" << std::endl;

}
