#include "print_util.hpp"

using namespace std;
using namespace Eigen;

#include <filesystem>

/*
 Returns a ofstream object for writing to the file specified by 'path'
 Throws a runtime error if unable to open the file. 
 */
ofstream open_log(const string path) {
    if (!path.empty()) {
        std::filesystem::path p(path);
        const auto parent = p.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }
    }
    ofstream logfile(path);
    if (path.length() > 0 && !logfile.is_open()) {
        throw runtime_error("open_log(): Unable to open logfile");
    }
    return logfile;
}

ofstream open_log_append(const string path) {
    if (!path.empty()) {
        std::filesystem::path p(path);
        const auto parent = p.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }
    }
    ofstream logfile(path, std::ios::out | std::ios::app);
    if (path.length() > 0 && !logfile.is_open()) {
        throw runtime_error("open_log_append(): Unable to open logfile");
    }
    return logfile;
}


/*
 Prints a formatted message to stdout and/or the provided ostream
 depending on the level of verbosity.

 If iter < 0, simply prints the message on a new line
 If iter >= 0, appends "Iteration {iter} | " to the front of
               the message.

 Verbosity Levels   0   ---     No output
                    1   ---     Print to stdout only
                    2   ---     Print to fp only
                    3   ---     Write to stdout and fp
*/
void print_progress_note(
        int verbose,
        std::ostream& os,
        const string& message,
        int iter) {

    // Return immediately for verbosity level 0
    if (verbose == 0)  return;

    std::string out = message;
    // Add iteration label to the message
    if (iter >= 0) out = "Iteration " + to_string(iter) + " | " + out;

    // Printing progress updates ...
    // ... to cout
    if (verbose == 1 || verbose == 3){
        cout << out << endl;
    } 
    // ... to os
    if (verbose == 2 || verbose == 3) {
        os << out << endl;
    }
}



/*
 Prints a message to indicate an eigenvector has been locked. 
*/
void print_locking_note(
        int verbose,
        std::ostream& os,
        int eig_idx,
        int iter
    ) {
    string message = "Locking Eigenvector " + to_string(eig_idx);
    print_progress_note(verbose, os, message, iter);
}



/*
 Prints the eigenvalues and residuals to stdout and/or the provided
 file depending on the level of verbosity selected.

  Verbosity Levels  0   ---     No output
                    1   ---     Print to stdout only
                    2   ---     Print to fp only
                    3   ---     Write to stdout and fp

*/
void print_convergence_data(  
        int verbose,
        std::ostream& os,
        const Ref<const VectorXd> ritz_vals,
        const Ref<const VectorXd> residuals,
        int iter) {
    
    // Return immediately for verbosity level 0
    if (verbose == 0)  return;

    // Verify the ritz values and residuals have the same length
    int n = ritz_vals.size();
    if (n != residuals.size()) {
        throw invalid_argument("Error in print_convergence_data(): Length of Ritz values and residuals do no match!" );
    }

    // Printing progress update
    string header_message = "Ritz Values & Residuals";
    print_progress_note(verbose, os, header_message, iter);
    for (int i = 0; i < n; i++) {
        stringstream ss;
        ss  << "Eig " << (i + 1) << " | "
            << setw(22) << scientific << setprecision(15) << ritz_vals(i) 
            << "\t"     
            << setw(10) << setprecision(3) << residuals(i);
        string line = ss.str();
        // Do not label eigenpair rows as iterations.
        print_progress_note(verbose, os, line, -1);
    }
}



/*
 Prints a single pair of eigenvalues and residuals the eigenvalues and 
 residuals to stdout and/or the provided file depending on the level of verbosity selected.

  Verbosity Levels  0   ---     No output
                    1   ---     Print to stdout only
                    2   ---     Print to fp only
                    3   ---     Write to stdout and fp

*/
void print_convergence_data(  
        int verbose,
        std::ostream& os,
        double ritz_val,
        double residual,
        int iter) {
    
    // Return immediately for verbosity level 0
    if (verbose == 0)  return;

    // Printing progress update
    stringstream ss;
    ss  << "Ritz Value = " 
        << setw(22) << scientific << setprecision(15) << ritz_val 
        << "\tResidual = "     
        << setw(10) << setprecision(3) << residual;
    string line = ss.str();
    print_progress_note(verbose, os, line, iter);
}

