#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

void Display2Vecs(const Eigen::VectorXd& vec1, 
                  const Eigen::VectorXd& vec2, 
                  const std::string& label1,
                  const std::string& label2,
                  int precision) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors must be of the same size.\n";
        std::cout << vec1.size() << "  |   " << vec2.size() << std::endl;
        return;
    }

    std::cout << std::scientific << std::setprecision(precision);

    // Determine column width based on expected number size and precision
    const int col_width = precision + 10;

    // Print column headers
    std::cout << std::setw(col_width) << label1
              << std::setw(col_width) << label2 << "\n";

    for (int i = 0; i < vec1.size(); ++i) {
        std::cout << std::setw(col_width) << vec1[i]
                  << std::setw(col_width) << vec2[i] << "\n";
    }

    // Reset formatting if needed
    std::cout.unsetf(std::ios::scientific);
}
