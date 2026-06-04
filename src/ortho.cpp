#include "fasteig.hpp"
#include "LOBPCGSolver.hpp"

using namespace Eigen;

/*
********************************************************************************

Name
    svqb

Description
    Modifies the matrix 'U' such that it is an orthogonal matrix, i.e. <U,U> = Id
    
Arguments

    U       [in/out]    Double precision, Eigen::MatrixXd array of size n by b1.
                        On entry, the matrix that is to be orthogonalized.
                        On exit, the orthogonalized matrix.

    tau     [in]        Double. Relative threshold for dropping/replacing singular
                        values, see 'method' below.
                        Default: 1e-14
                        
    method  [in]        char. Sets how small singular values of the Gram matrix
                        are handled. 
                        
                        If 'R', singular values below t = tau * max_i |sigma_i|
                                are replaced by t
                        If 'D', singular values below t = tau * max_i |sigma_i|
                                are dropped (e.g. set to zero)

    solver  [out]       Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver, used
                        to compute an SVD

Reference:  Duersch, J.A., Shao, M., Yang, C., & Gu, M. (2017). A Robust and 
            Efficient Implementation of LOBPCG. SIAM J. Sci. Comput., 40, C655-C676.
            
            (cf. Algorithm 6)

TODO

    Reduce number of intermediate array allocations


********************************************************************************
*/
void svqb(
        MatrixXd& U, 
        double tau, 
        char method,
        SelfAdjointEigenSolver<MatrixXd>& solver) {

    // Overlap matrix
    MatrixXd W = U.transpose() * U;
    
    // Rescale diagonal to 1
    VectorXd d = W.diagonal();
    d = d.array().sqrt().inverse();
    auto D = d.asDiagonal();
    W.applyOnTheLeft(D);
    W.applyOnTheRight(D);

    // SVD of W (symmetric matrix => svd = eigendecomp)
    solver.compute(W);
    auto theta = solver.eigenvalues();
    auto Z = solver.eigenvectors();   
    
    // Replace or drop eigenvalues that are too small
    double threshold = tau * theta.lpNorm<Infinity>();
    double newval = (method == 'R') ? threshold : 0.0;
    theta = (theta.array() < threshold).select(newval, theta);
        
    // Compute theta.^(-1/2)
    theta = theta.array().sqrt().inverse();
    // Remove any Infs caused by dropping in eigenvalue above
    theta = theta.array().isInf().select(0, theta); 
    
    // Undo projection and diagonal scaling
    // U <-- U * d^(-1/2) * Z * theta^(-1/2)
    U.applyOnTheRight(D);
    U.applyOnTheRight(Z);
    U.applyOnTheRight(theta.asDiagonal());

}


/*
********************************************************************************

    Name
        ortho

    Description
        Modifies the matrix 'U' such that
            (a) U is an orthogonal matrix, i.e. <U,U> = Id
            (b) Each column of U is orthogonal to the columns of V, i.e. V' * U = 0
        The matrix V itself should be orthogonal on input
        
    Arguments

        U       [in/out]    Double precision, Eigen::MatrixXd array of size n by b1.
                            On entry, the matrix that is to be orthogonalized wrt 'V'
                            On exit, the orthogonalized matrix

        V       [in]        Double precision, Eigen::MatrixXd array of size n by b2.
                            Matrix with which 'U' is to be orthogonalized.

        tol     [in]        Double. Sets orthogonalization tolerance. The function 
                            breaks out of its inner and output loops when
                                |U'U - Id| < tol * |U|  and  |V'U| < tol*|U|,
                            respectively, or prematurely if 'maxiters1' or 'maxiters2' 
                            has been exceeded.

        solver  [out]       Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver, used
                            by svqb() subroutine to compute an SVD

        maxiter1 [in]       Int. Max number of outer loop iterations
                            Default: 3

        maxiter2 [in]       Int. Max number of inner loop iterations (calls to svqb())
                            Default: 3

        tau     [in]        Double. Relative threshold for dropping/replacing singular
                            values during the svqb() function call.
                            Default: 1e-14

    Reference:  Duersch, J.A., Shao, M., Yang, C., & Gu, M. (2017). A Robust and 
                Efficient Implementation of LOBPCG. SIAM J. Sci. Comput., 40, C655-C676.
                
                (cf. Algorithm 5)

    TODO

        Reduce number of intermediate array allocations

********************************************************************************
*/

void ortho(
        MatrixXd& U, 
        const MatrixXd& V, 
        double tol, 
        SelfAdjointEigenSolver<MatrixXd>& solver,
        int maxiter1,
        int maxiter2,
        double tau) {
    
    // Save copy of the identity matrix, used
    // to check for convergence
    int b = U.cols();
    auto Id = MatrixXd::Identity(b,b);
    
    double normU, orth_err;
    for (int i = 0; i < maxiter1; i++) {

        //printf("ortho(): outer iter i = %d\n", i+1);

        // Remove project of U onto V
        U = U - V * (V.transpose() * U);

        // Orthogonalize what remains of U
        for (int j = 0; j < maxiter2; j++) {
            //printf("ortho(): inner iter j = %d\n", j+1);

            // svqb with replacement on iter j = 0
            //           drop        on iter > 0
            char method = (j == 0) ? 'R' : 'D';
            svqb(U, tau, method, solver);
            // Check for convergence of U'U to Id
            normU = U.norm();
            orth_err = (U.transpose() * U - Id).norm(); 
            if ( orth_err < tol*normU ) { break; }
        }

        // Check for convergence of V'U to 0
        orth_err = (V.transpose() * U).norm();
        if ( orth_err < tol*normU ) { break; }
    
    }

}

