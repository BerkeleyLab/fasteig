//********************************************************************************
//
//  Name
//    SPPC_Ind
//
//  Description
//    The SPPC_Ind function performs the Shifted Perturbative Polynomial Correction (SPPC) method 
//    for computing eigenpairs of a given matrix. This method refines an initial guess for 
//    eigenvectors and eigenvalues using a perturbative approach, iterating until convergence 
//    is achieved or a maximum number of perturbations is reached.
//
//  Arguments
//    V0     Double precision, Eigen::MatrixXd of size n by m. (INPUT)
//           - The initial set of eigenvectors, where each column represents an eigenvector.
//
//    L0     Double precision, Eigen::MatrixXd of size m by m. (INPUT)
//           - The initial set of eigenvalues, stored as a diagonal matrix.
//
//    ops    Structure. (INPUT)
//           - A structure containing matrix operations, including:
//             - `matvec(v, Av)`: A user-defined matrix-vector multiplication function that computes Av = A * v.
//             - `V21`: A matrix used for computing perturbations.
//             - `H0`: A sparse symmetric matrix representing a component of the system matrix.
//             - `minres(A, b, x)`: A function to solve the linear system Ax = b.
//
//    ns     Integer. (INPUT)
//           - The number of degrees of freedom in the submatrix split.
//
//    m      Integer. (INPUT)
//           - The number of eigenpairs to be computed.
//
//    P      Integer. (INPUT)
//           - The maximum number of perturbative iterations allowed.
//
//    X      Double precision, Eigen::MatrixXd of size n by m. (OUTPUT)
//           - The computed eigenvectors, where each column represents an eigenvector.
//
//    EIGS   Double precision, Eigen::MatrixXd of size P+1 by m. (OUTPUT)
//           - A matrix storing the computed eigenvalues at each perturbation iteration.
//
//    RESID  Double precision, Eigen::MatrixXd of size P+1 by m. (OUTPUT)
//           - A matrix storing the residual norm at each perturbation iteration.
//
//    tol    Double precision. (INPUT)
//           - The convergence tolerance for the relative residual of eigenpair approximation.
//
//    normalize  Boolean. (INPUT)
//           - If true, normalizes the computed vectors at each iteration to improve numerical stability.
//
//  Exceptions
//    Throws std::runtime_error if:
//    - `V0.cols() != m`, indicating that the input eigenvector matrix does not have the expected number of columns.
//    - Eigen decomposition fails when computing the Ritz pairs.
//
//  Algorithm
//    1. Initialize matrices and vectors for storing perturbative corrections.
//    2. Loop over each eigenpair and refine it using the perturbation expansion.
//    3. Compute initial projections and residuals to check for immediate convergence.
//    4. Iteratively apply perturbative corrections up to order `P`:
//       - Compute perturbative eigenvalues and eigenvectors.
//       - Orthogonalize vectors against the previous basis to ensure stability.
//       - Construct a reduced Hamiltonian matrix and solve for updated eigenvalues.
//       - Check residuals and determine if the computed eigenvector has converged.
//    5. Store the final computed eigenvectors in `X`.
//
//  References
//    0. ********TO BE ADDED********
//    1. Yang, Chao. "A study of perturbative methods for electronic structure calculations." 
//       *SIAM Journal on Scientific Computing*, 2009.
//    2. Saad, Yousef. *Numerical Methods for Large Eigenvalue Problems.* SIAM, 2011.
//
//  Authors
//    Dong Min Roh  
//    Chao Yang  
//    Applied Mathematics and Computational Research Division  
//    Lawrence Berkeley National Laboratory
//
//********************************************************************************

#include "fasteig.hpp"

void SPPC_Ind( const MatrixXd& V0,
	       const MatrixXd& L0,
	       const operations& ops,
	       int ns,
	       int m,
	       int P,
	       MatrixXd& X,
	       MatrixXd& EIGS,
	       MatrixXd& RESID,
	       double tol,
	       bool normalize ) {

  const int n = V0.rows();
  if (V0.cols() != m) throw std::runtime_error("Error: V0 must have exactly m columns.");

  // Perform SPPC for each eigenpair
  for ( size_t i = 0; i < m; ++i ) {

    std::cout << std::endl;
    std::cout << "-- Computing the eigenpair number " << i+1 << " --" << std::endl;
    std::cout << std::endl;

    MatrixXd V(n, P+1);				// Perturbative vectors
    VectorXd L(P+1);				// Perturbative eigenvalue approximations
    MatrixXd N(n, P+1);				// Orthonormal basis
    MatrixXd HN(n, P+1);			// H*N vectorx
    VectorXd BETA(P+1);				// Norms of orthogonalized vectors

    SpMat I(ns, ns);
    I.setIdentity();

    // Initial eigenpair data
    double l0 = L0(i, i);
    VectorXd v0 = V0.col(i);
    V.col(0) = v0;
    L(0) = l0;
    BETA(0) = 1.0;
    N.col(0) = v0;

    VectorXd temp(n);
    ops.matvec(v0, temp);
    HN.col(0) = temp;

    double hproj = N.col(0).dot(HN.col(0));
    EIGS(0, i) = hproj;
    RESID(0, i) = (HN.col(0) - hproj * N.col(0)).norm() / std::abs(hproj);

    if ( RESID(0, i) < tol ) {
      X.col(i) = v0;
      continue;
    }

    // To be used for perturbative calculations
    VectorXd V21v0 = ops.V21 * v0.head(ns);	// V*v0

    // Initialize projection matrix H_proj.
    MatrixXd H_proj(1, 1);
    H_proj(0, 0) = hproj;

    VectorXd v_EC;				// eigenvector approximation

    size_t p;
    for ( p = 1; p <= P; ++p ) {

      if ( p == 1 ) {
        
	// First perturbation (exact values known)
	double l1 = 0.0;
	VectorXd v1(n);
	v1.head(ns).setZero();
	v1.tail(n-ns) = V21v0 / l0;
	V.col(p) = v1;
	L(p) = l1;
      
      }
      else {
        
	VectorXd v_prev = V.col(p-1);
        VectorXd v1 = v_prev.head(ns);
        VectorXd v2 = v_prev.tail(n-ns);
        double l_val = v2.dot(V21v0);
        L(p) = l_val;

        // RHS
	VectorXd tvec(p);	
	tvec.head(p-1) = N.block(0, 0, n, p-1).transpose() * v_prev;
	tvec(p-1) = BETA(p-1);
	VectorXd b = HN.block(0, 0, n, p) * tvec;

	VectorXd subVec = VectorXd::Zero(n);
	subVec.head(ns) = ops.H0.selfadjointView<Eigen::Lower>() * v1;
	b = b - subVec;

	b = -b;
	for ( size_t r = 2; r <= p; ++r ) {
	  b += L(r) * V.col(p-r);
	}
	if ( normalize ) {
	  double normb = b.norm();
	  b /= normb;
	}
	else std::cout << "Norm of the right hand side = " << b.norm() << std::endl;
	VectorXd b1 = b.head(ns);
	VectorXd b2 = b.tail(n-ns);

	// Solve linear system
	SpMat A = ops.H0 - l0 * I;
	VectorXd y1(ns);
	ops.minres(A, b1, y1);
	
	VectorXd proj = v0.head(ns) * (v0.head(ns).dot(y1));
	y1 -= proj;
		// Reorthogonalize
	proj = v0.head(ns) * (v0.head(ns).dot(y1));
	y1 -= proj;

	VectorXd y2 = -b2 / l0;
	
	VectorXd v_new(n);
	v_new << y1, y2;
	V.col(p) = v_new;

      }

      // Orthogonalize w.r.t. prvious subspace
      VectorXd v_temp = V.col(p);
      VectorXd h = N.block(0, 0, n, p).transpose() * v_temp;
      v_temp = v_temp - N.block(0, 0, n, p) * h;
      VectorXd s = N.block(0, 0, n, p).transpose() * v_temp;
      h += s;
      v_temp = v_temp - N.block(0, 0, n, p) * s;
      double beta = v_temp.norm();
      v_temp /= beta;

      BETA(p) = beta;
      N.col(p) = v_temp;
      ops.matvec(v_temp, temp);
      HN.col(p) = temp;

      // Compute Ritz pair
      MatrixXd H_proj_new(p+1, p+1);
      H_proj_new.topLeftCorner(p, p) = H_proj;
      for ( size_t j = 0; j < p; ++j ) {
        double val = N.col(j).dot(HN.col(p));
	H_proj_new(j, p) = val;
	H_proj_new(p, j) = val;
      }
      H_proj_new(p, p) = N.col(p).dot(HN.col(p));
      H_proj = (H_proj_new + H_proj_new.transpose()) / 2.0;

      Eigen::SelfAdjointEigenSolver<MatrixXd> es(H_proj);
      if ( es.info() != Eigen::Success ) throw std::runtime_error("Eigen decomposition failed in SPPC_Ind.");
      VectorXd evals = es.eigenvalues();
      MatrixXd evecs = es.eigenvectors();
      int minIndex;
      evals.minCoeff(&minIndex);
      double l_EC = evals(minIndex);
      VectorXd pre_v_EC = evecs.col(minIndex);
      v_EC = N.block(0, 0, n, p + 1) * pre_v_EC;

      // Check convergence
      double resid = (HN.block(0, 0, n, p + 1) * pre_v_EC - l_EC * v_EC).norm() / std::abs(l_EC);
      EIGS(p, i) = l_EC;
      RESID(p, i) = resid;
      std::cout << "Perturbation/Iteration " << p << ": Relative residual \t" << resid << "\n";

      if ( resid < tol ) {
        X.col(i) = v_EC;
	break;
      }

    }
    if ( p > P ) X.col(i) = v_EC;

  }

}

                

