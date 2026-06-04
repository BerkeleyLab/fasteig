#include "fasteig.hpp"
#include "mmio.hpp"

bool read_matrix_market_file( const char* filename, SpMat& H, int&M, int& N ) {

  // Read the input file
  char fname[50];
  FILE *fp = NULL;

  sscanf( filename, "%s", fname );
  std::cout << std::endl;
  fp = fopen( fname, "rb" );
  if ( fp ) {
    std::cout << "Reading the input file name: " << fname << std::endl;
  }
  else {
    std::cerr << "No such file name!" << std::endl;
    return false;
  }
  std::cout << std::endl;

  // Read the file
  int ret_code;
  MM_typecode matcode;
  int nz;
  int *I, *J;
  double *val;

  if ( mm_read_banner(fp, &matcode) != 0 ) {
    std::cerr << "Could not process Matrix Market banner." << std::endl;
    return false;
  }

  if ( (ret_code = mm_read_mtx_crd_size(fp, &M, &N, &nz)) != 0 ) return 1;

  I = (int *) malloc(nz * sizeof(int));
  J = (int *) malloc(nz * sizeof(int));
  val = (double *) malloc(nz * sizeof(double));

  for ( int i = 0; i < nz; ++i ) {
    fscanf(fp, "%d %d %lg\n", &I[i], &J[i], &val[i]);
    I[i]--;  /* adjust from 1-based to 0-based */
    J[i]--;
  }

  // Create sparse matrix in Eigen format
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  for ( int i = 0; i < nz; ++i ) {
    tripletList.push_back( T(I[i], J[i], val[i]) );
  }
  H.resize(N, N);
  H.setFromTriplets(tripletList.begin(), tripletList.end());

  free(I);
  free(J);
  free(val);

  return true;
}
