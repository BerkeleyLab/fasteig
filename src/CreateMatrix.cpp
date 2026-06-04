#include "fasteig.hpp"
#include "mmio.hpp"


FASTEIG_INT CreateMatrix(SpMat& A, fasteig_params *params) {

    FASTEIG_INT n, nnz;

    n   = params->n;
    nnz = params->annz;

    if (n <=0 || nnz <=0) {
       std::cerr << " Matrix dimension n = " << n << " Number of nonzeros = " << nnz << std::endl;
       return 1;
    }
    
    // Create sparse matrix in Eigen format
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    for ( int i = 0; i < nnz; ++i ) {
       tripletList.push_back( T(params->arowind[i], params->acolind[i], params->anzvals[i]) );
    }
    
    A.resize(n,n); 
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    return 0;
}
