//********************************************************************************
//
//  Name
//    getevbounds
//
//  Description
//    Use Lanczos to get an lower and upper bound of the spectrum of op
//
//  Arguments
//    n         Integer (INPUT)
//              Dimension of the matrix to be diagonalized
//
//    ops       Structure. (INPUT)
//              Operations structure with a defined matrix-vector multiplication - Matvec.
//              Matvec is defined by the user outside of lanczos.
//
//    evlb      Double precision. (OUTPUT)
//              On OUTPUT:
//              A lower bound of the spectrum.
//
//    evub      Double precision. (OUTPUT)
//              On OUTPUT:
//              An upper bound of the spectrum.
//
//
//  References
//    1. Stewart, Gilbert W. "A Krylov--Schur algorithm for large eigenproblems." SIAM Journal on Matrix Analysis and Applications 23.3 (2002): 601-614.
//
//  Authors
//    Dong Min Roh
//    Chao Yang
//    Applied Mathematics and Computational Research Division 
//    Lawrence Berkeley National Laboratory
//
//********************************************************************************

#include "fasteig.hpp"
#include "fasteig_solver.hpp"

#include <limits>

void getevbounds( int n, const operations& ops, double &evlb, double &evub ) {

  int k = 20; // number of Lanczos iterations, hard coded for now

  MatrixXd Q(n, k);
  MatrixXd H(k, k);

  VectorXd v0 = Eigen::VectorXd::Random(n);

  H.setZero();

  double beta;

  KrylovSubspace( v0, Q, H, ops, 0, k );

  beta = v0.norm();

  // Eigen::MatrixXd H1 = H.block(0, 0, k, k);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H.block(0,0,k,k));

  VectorXd eigenvalues = solver.eigenvalues();

  // std::cout << "Lanczos Ritz values:\n" << eigenvalues << std::endl;

  MatrixXd eigenvectors = solver.eigenvectors();

  double t, x;

  double pos_inf = std::numeric_limits<double>::infinity();  // Positive infinity
  double neg_inf = -std::numeric_limits<double>::infinity(); // Negative infinity

  evlb = pos_inf;
  evub = neg_inf;

  double val = 0.0;

  for (int j = 0; j < k; j++) {
     t = fabs(beta*eigenvectors(k-1,j));
     x = eigenvalues(j)-t;
     if (x < evlb) evlb = x;
        x = eigenvalues(j)+t;
     if (x > evub) evub = x;
  }
}
