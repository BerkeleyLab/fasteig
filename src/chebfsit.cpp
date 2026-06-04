#include "fasteig.hpp"
#include "fasteig_solver.hpp"

void chebyshev_filter(const operations &ops, MatrixXd &V, int degree, double lambda_min, double lambda_max) {
    MatrixXd V_old = V;
    MatrixXd V_new = V;
    MatrixXd AV    = V;

    ops.block_matvec(V,V_new);

    double rat = 1.0e3; // the ration of Cheby(xlb) and Cheby(lb) is hard coded for now.
    double xlb = std::cosh(std::acosh(rat)/degree); // xlb maps to the low end of the spectrum
						    //
    // [lb,ub] is mapped [-1,1]
    double ub = lambda_max;
    double lb = (2.0*lambda_min + lambda_max*xlb - lambda_max)/(1.0+xlb); 

    double c = 2.0 / (ub - lb);
    double d = -(ub + lb) / (ub - lb);
    
    V_new = c * V_new + d * V;
    for (int k = 2; k <= degree; ++k) {
        ops.block_matvec(V_new, AV);
        MatrixXd V_tmp = 2.0 * (c * AV + d * V_new) - V_old;
        // can use pointer swap to avoid copying data
        V_old = V_new;
        V_new = V_tmp;
    }
    V = V_new;
}

int chebyfsit(const operations &ops, MatrixXd &V, int maxiter, int degree, VectorXd &ritz_vals, MatrixXd &ritz_vecs, MatrixXd &residuals) {

    double lambda_max, lambda_min;

    int info = 0;

    int nrows = V.rows();
    int ncols = V.cols();
    Eigen::MatrixXd X = V;
    Eigen::MatrixXd eigvecs(ncols,ncols);

    std::cout << "Chebyshev filtering for low the of the spectrum... \n" << std::endl;
    getevbounds(nrows, ops, lambda_min, lambda_max);
    std::cout << "Spectral bounds from Lanczos: " << std::endl;
    std::cout << "lb = " << lambda_min << " ub = " << lambda_max << std::endl;
    
    for (int iter = 0; iter < maxiter; ++iter) {
        std::cout << "Cheby filter iter = " << iter << std::endl;

        chebyshev_filter(ops, X, degree, lambda_min, lambda_max);
        // Cholesky QR
	     Eigen::MatrixXd Gmat = X.transpose() * X;
	     Eigen::LLT<Eigen::MatrixXd> llt(Gmat);
	     if (llt.info() == Eigen::Success) {
           // Get the lower triangular matrix L such that Gmat = L * L.transpose()
           Eigen::MatrixXd L = llt.matrixL();
	        ritz_vecs = X*L.transpose().triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(ncols,ncols));
	        X = ritz_vecs;
        } else {
           std::cerr << "Decomposition failed. Matrix may not be positive definite." << std::endl;
	        info = -1;
	        break;
        }
    } // end for iter
    if (info == 0) fasteig::fasteig_solver::RayleighRitz(ops, X, ritz_vals, ritz_vecs, eigvecs, residuals);

    return info;
}
