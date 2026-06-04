#include "fasteig.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <vector>
#include <iostream>

using namespace Eigen;

/*
 * SPPC_Comb:
 * -----------
 * This routine computes m approximate eigenpairs of the full symmetric matrix H
 * via the Combined Subspace Projection with Perturbative Corrections (SPPC_Comb) algorithm.
 *
 * INPUTS:
 *  H      : (n x n) full symmetric matrix (used via ops.matvec)
 *  V0     : (n x m) initial subspace initializer.
 *  L0     : (m x m) diagonal matrix with eigenvalues of H₀ (leading submatrix).
 *  ns     : size of the leading submatrix H₀ (first ns rows/cols of H).
 *  m      : number of desired eigenpairs.
 *  P      : maximum number of allowed perturbations/iterations.
 *  normalize : if true, normalize the right-hand side in the perturbative step.
 *  ops    : operations structure (which defines ops.matvec and ops.minres and stores H0 and V21)
 *  TOL    : desired accuracy (tolerance).
 *
 * OUTPUTS:
 *  X      : (n x m) final approximate eigenvectors.
 *  V      : (n x m*(P+1)) collection of perturbative correction vectors.
 *  N      : (n x m*(P+1)) orthonormal basis spanning the subspace.
 *  EIGS   : ((P+1) x m) eigenvalue approximations at each iteration.
 *  RESID  : ((P+1) x m) relative residuals at each iteration.
 */
void SPPC_Comb( const MatrixXd &V0,
                const MatrixXd &L0,
                int ns,
                int m,
                int P,
                bool normalize,
                const operations &ops,
                MatrixXd &X,
                MatrixXd &V,
                MatrixXd &N,
                MatrixXd &EIGS,
                MatrixXd &RESID,
                double TOL )
{
    int n = V0.rows();
    if(V0.cols() != m)
        throw std::runtime_error("Error: V0 must have exactly m columns.");
    
    // Extract leading submatrix H0 and coupling block V21 from H.
    // (H0 and V21 are stored in ops.)
    MatrixXd H0 = ops.H0.toDense();  // convert sparse to dense for small H0 operations
    MatrixXd V21;
    if(ops.V21.rows() > 0)
        V21 = ops.V21.toDense();
    else
        V21 = MatrixXd::Zero(0, ns);

    // Allocate outputs.
    // X: final eigenvectors, size (n x m)
    X = MatrixXd::Zero(n, m);
    // EIGS and RESID have (P+1) rows and m columns.
    EIGS = MatrixXd::Zero(P+1, m);
    RESID = MatrixXd::Zero(P+1, m);
    // V and N store all perturbative vectors and the corresponding ONB.
    V = MatrixXd::Zero(n, m*(P+1));
    N = MatrixXd::Zero(n, m*(P+1));
    // Lvec holds the perturbative eigenvalue approximations (vector of length m*(P+1)).
    VectorXd Lvec = VectorXd::Zero(m*(P+1));

    // R_cell stores the R factors from the QR factorizations (one per perturbation).
    std::vector<MatrixXd> R_cell(P+1);
    R_cell[0] = MatrixXd::Ones(1,1);  // MATLAB: R_cell{1} = 1;

    bool CONVERGED = true;
    
    // Set initial eigenvalue approximations from L0.
    for(int i = 0; i < m; i++){
        Lvec(i) = L0(i,i);
    }
    
    // Initialize first block of V and N with V0.
    V.block(0, 0, n, m) = V0;
    N.block(0, 0, n, m) = V0;
    
    // Compute initial HN = H * N(:,1:m) using ops.matvec.
    MatrixXd HN = MatrixXd::Zero(n, m*(P+1));
    {
        VectorXd temp = VectorXd::Zero(n);
        ops.matvec(V0.col(0), temp);  // dummy call for dimension; we'll fill block-wise
        for(int j = 0; j < m; j++){
            VectorXd tempj = VectorXd::Zero(n);
            ops.matvec( N.col(j), tempj );
            HN.col(j) = tempj;
        }
    }
    
    // Compute initial Ritz pairs.
    MatrixXd H_proj = N.block(0, 0, n, m).transpose() * HN.block(0, 0, n, m);
    H_proj = (H_proj + H_proj.transpose())/2.0;
    SelfAdjointEigenSolver<MatrixXd> es(H_proj);
    MatrixXd pre_V_EC = es.eigenvectors();
    MatrixXd L_EC = es.eigenvalues().asDiagonal();
    // (Assume eigenvalues are in ascending order.)
    MatrixXd V_EC = N.block(0, 0, n, m) * pre_V_EC;
    MatrixXd RES = HN.block(0, 0, n, m) * pre_V_EC.topRows(m) - V_EC * L_EC.topLeftCorner(m,m);
    for(int i = 0; i < m; i++){
        EIGS(0,i) = L_EC(i,i);
        RESID(0,i) = RES.col(i).norm() / std::abs(L_EC(i,i));
        if(RESID(0,i) >= TOL)
            CONVERGED = false;
    }
    std::cout << "Perturbation/Iteration 0: Relative residual";
    for(int i = 0; i < m; i++){
        std::cout << "\t" << RESID(0,i);
    }
    std::cout << std::endl;
    if(CONVERGED) {
        X = V_EC.block(0,0,n,m);
        return;
    } else {
        CONVERGED = true;
    }
    
    // Precompute V21v0s = V21 * V0(1:ns, :)
    MatrixXd V21v0s = V21 * V0.topRows(ns);
    
    // Main perturbation loop: for p = 2 to P+1 (MATLAB indexing)
    int p;
    for(p = 2; p <= P+1; p++){
        // For block indices, the new block will fill columns (p-1)*m to p*m-1 of V and N.
        if(p == 2){
            // First perturbation: for each eigenpair i,
            // v1 = [zeros(ns,1); V21v0s(:, i) / Lvec(i)]
            for(int i = 0; i < m; i++){
                VectorXd v1 = VectorXd::Zero(n);
                v1.tail(n - ns) = V21v0s.col(i) / Lvec(i);
                V.col(m + i) = v1;
                Lvec(m + i) = 0;  // first perturbation eigenvalue = 0.
            }
        } else {
            // For p >= 3:
            // Compute HV = HN(:, 1:((p-1)*m)) * T,
            // where T is constructed from previous blocks.
            // (For brevity, we mimic the MATLAB code by forming HV externally.)
            int prevCols = (p-1)*m;
            // Here we assume T = [N(:,1:((p-2)*m))' * V(:,((p-2)*m+1):(p-1)*m); R_cell[p-1]]
            // For simplicity we form HV as:
            MatrixXd T = MatrixXd::Zero((p-1)*m, m);
            if(p > 2){
                // Compute the first (p-2)*m rows: for each i in 0..(p-2)*m - 1, T(i,j) = N.col(i).dot( V.col((p-2)*m+j) )
                for(int j = 0; j < m; j++){
                    for(int i = 0; i < (p-2)*m; i++){
                        T(i,j) = N.col(i).dot(V.col((p-2)*m + j));
                    }
                }
                // Append R_cell[p-1] as the last block of rows.
                T.block((p-2)*m, 0, R_cell[p-1].rows(), m) = R_cell[p-1];
            }
            MatrixXd HV = HN.block(0, 0, n, prevCols) * T;
            
            for(int i = 0; i < m; i++){
                double l0_val = L0(i,i);
                VectorXd v0_i = V0.col(i);
                // Use the precomputed V21v0s for this eigenpair.
                // Let y = operator defined as: y(x) = H0*x - l0_val*x.
                // (We assume that such an operator is provided by ops, as in SPPC_Ind.)
                // Retrieve current perturbative vector: v = V(:, (p-2)*m + i)
                VectorXd v = V.col((p-2)*m + i);
                VectorXd v1 = v.head(ns);
                VectorXd v2 = v.tail(n - ns);
                double l_val = v2.dot(V21v0s.col(i));
                Lvec((p-1)*m + i) = l_val;
                
                VectorXd b = HV.col(i);
                VectorXd subVec = VectorXd::Zero(n);
                subVec.head(ns) = H0 * v1;
                b = - (b - subVec);
                // Accumulate corrections from previous perturbations:
                for(int r = 0; r < p-2; r++){
                    b += Lvec(m*(p - r - 1) + i) * V.col(m*r + i);
                }
                if(normalize){
                    double normb = b.norm();
                    if(normb > 0)
                        b /= normb;
                } else {
                    std::cout << "Norm of RHS = " << b.norm() << std::endl;
                }
                VectorXd b1 = b.head(ns);
                VectorXd b2 = b.tail(n - ns);
                // Solve (H0 - l0_val I)*y1 = b1 using MINRES via ops.minres.
                MatrixXd A_op = H0 - l0_val * MatrixXd::Identity(ns, ns);
                SpMat A_sparse = A_op.sparseView();
                VectorXd y1(ns);
                ops.minres(A_sparse, b1, y1);
                // Reorthogonalize y1 against v0_i(1:ns).
                VectorXd temp = v0_i.head(ns) * (v0_i.head(ns).dot(y1));
                y1 -= temp;
                temp = v0_i.head(ns) * (v0_i.head(ns).dot(y1));
                y1 -= temp;
                VectorXd y2 = -b2 / l0_val;
                VectorXd new_v(n);
                new_v << y1, y2;
                V.col((p-1)*m + i) = new_v;
            }
        } // end p>=3 branch
        
        // 2. Orthogonalize new block of V (columns (p-1)*m to p*m-1) against previous N.
        MatrixXd cur_V = V.block(0, (p-1)*m, n, m);
        MatrixXd h = N.block(0, 0, n, (p-1)*m).transpose() * cur_V;
        cur_V = cur_V - N.block(0, 0, n, (p-1)*m) * h;
        MatrixXd s = N.block(0, 0, n, (p-1)*m).transpose() * cur_V;
        h = h + s;
        cur_V = cur_V - N.block(0, 0, n, (p-1)*m) * s;
        HouseholderQR<MatrixXd> qr(cur_V);
        MatrixXd Q_temp = qr.householderQ() * MatrixXd::Identity(n, m);
        MatrixXd R = qr.matrixQR().topRows(m).triangularView<Upper>();
        R_cell[p-1] = R;
        N.block(0, (p-1)*m, n, m) = Q_temp;
        {
            VectorXd temp = VectorXd::Zero(n);
            ops.matvec(Q_temp.col(0), temp);
            for(int j = 0; j < m; j++){
                VectorXd tempj = VectorXd::Zero(n);
                ops.matvec( Q_temp.col(j), tempj );
                HN.col((p-1)*m + j) = tempj;
            }
        }
        
        // 3. Expand the projected matrix H_proj.
        int totalCols = (p-1)*m;
        MatrixXd H_proj_new(totalCols + m, totalCols + m);
        H_proj_new.topLeftCorner(totalCols, totalCols) = H_proj;
        MatrixXd A_top = N.block(0, 0, n, totalCols).transpose() * HN.block(0, (p-1)*m, n, m);
        MatrixXd A_bot = HN.block(0, (p-1)*m, n, m).transpose() * N.block(0, 0, n, totalCols);
        MatrixXd A_diag = N.block(0, (p-1)*m, n, m).transpose() * HN.block(0, (p-1)*m, n, m);
        H_proj_new.topRightCorner(totalCols, m) = A_top;
        H_proj_new.bottomLeftCorner(m, totalCols) = A_bot;
        H_proj_new.bottomRightCorner(m, m) = A_diag;
        H_proj = (H_proj_new + H_proj_new.transpose())/2.0;
        
        SelfAdjointEigenSolver<MatrixXd> es2(H_proj);
        MatrixXd pre_V_EC2 = es2.eigenvectors();
        MatrixXd L_EC2 = es2.eigenvalues().asDiagonal();
        // Assume eigenvalues sorted ascending.
        MatrixXd V_EC2 = N.block(0, 0, n, p*m) * pre_V_EC2;
        
        MatrixXd RES_new = HN.block(0, 0, n, p*m) * pre_V_EC2.topRows(m) - V_EC2.block(0, 0, n, m) * L_EC2.topLeftCorner(m, m);
        for(int i = 0; i < m; i++){
            EIGS(p-1,i) = L_EC2(i,i);
            RESID(p-1,i) = RES_new.col(i).norm() / std::abs(L_EC2(i,i));
            if(RESID(p-1,i) >= TOL)
                CONVERGED = false;
        }
        std::cout << "Perturbation/Iteration " << p-1 << ": Relative residual";
        for(int i = 0; i < m; i++){
            std::cout << "\t" << RESID(p-1,i);
        }
        std::cout << std::endl;
        
        if(CONVERGED){
            X = V_EC2.block(0, 0, n, m);
            break;
        } else {
            CONVERGED = true;
        }
        // Update current approximation.
        X = V_EC2.block(0, 0, n, m);
    } // end perturbation loop
    
    // If no convergence within P iterations, X holds last V_EC2.
}

