#include "fasteig.hpp"

int CholeskyQR(MatrixXd &X, MatrixXd &Q, MatrixXd &R) 
{
	int info = 0;
	int ncols = X.cols();

	Eigen::MatrixXd Gmat = X.transpose() * X;
        Eigen::LLT<Eigen::MatrixXd> llt(Gmat);
        if (llt.info() == Eigen::Success) {
           // Get the lower triangular matrix L such that Gmat = L * L.transpose()
           Eigen::MatrixXd L = llt.matrixL();
           R = L.transpose();
           Q = X*R.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(ncols,ncols));
        } else {
           std::cerr << "Decomposition failed. Matrix may not be positive definite." << std::endl;
           info = -1;
        }
	return info;
}

