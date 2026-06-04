#ifndef PRINT_UTIL_HPP
#define PRINT_UTIL_HPP


#include <iostream> // cout
#include <fstream>  // ofstream
#include <sstream>  // stringstream
#include <iomanip>  // setprecision
#include <stdexcept>
#include <ostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

ofstream open_log(const string path);
ofstream open_log_append(const string path);

void print_progress_note(   int verbose,
                            std::ostream& os,
                            const string& message,
                            int iter = -1);

void print_locking_note(    int verbose,
                            std::ostream& os,
                            int eig_idx,
                            int iter);


void print_convergence_data(    int verbose,
                                std::ostream& os,
                                double ritz_val,
                                double residual,
                                int iter);

void print_convergence_data(    int verbose,
                                std::ostream& os,
                                const Ref<const VectorXd> ritz_vals,
                                const Ref<const VectorXd> residuals,
                                int iter);

#endif // PRINT_UTIL_HPP

