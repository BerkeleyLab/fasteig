#include "fasteig.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

bool readRSA(const std::string& filename, 
             FASTEIG_INT &nRows,
             FASTEIG_INT &nCols,
             FASTEIG_INT &nNonZeros,
             std::vector<FASTEIG_INT> &colPtr,
             std::vector<FASTEIG_INT> &rowInd,
             std::vector<FASTEIG_DOUBLE> &values 
     )
{
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line1, line2, line3, line4, line5;
    // 1. Read Header (4 lines)
    std::string title, key;
    std::getline(file, line1); // Title
    std::getline(file, line2); // Total lines, ptr lines, ind lines, val lines, type
    
    int nPtrLines, nIndLines, nValLines;
    std::string formatPtr, formatInd, formatVal;
    
    // Parse header 2: totln, ptrln, indln, valln
    std::stringstream ss2(line3);
    ss2 >> nPtrLines >> nPtrLines >> nIndLines >> nValLines;

    std::getline(file, line4); // Read Ptr/Ind/Val format strings
    std::getline(file, line5); // Matrix dimensions
    std::stringstream ss4(line4);
    std::string type;
    int dummy;
    ss4 >> type >> nRows >> nCols >> nNonZeros >> dummy;

    // 2. Read Column Pointers (CSC format)
    colPtr.resize(nCols + 1);
    for (int i = 0; i <= nCols; ++i) {
        file >> colPtr[i];
        // change to 0-based
        colPtr[i]--;
    }

    // 3. Read Row Indices
    rowInd.resize(nNonZeros);
    for (int i = 0; i < nNonZeros; ++i) {
        file >> rowInd[i];
        // change to 0-based
        rowInd[i]--;
    }

    // 4. Read Values
    values.resize(nNonZeros);
    for (int i = 0; i < nNonZeros; ++i) {
        file >> values[i];
    }

    file.close();
    return true;
}

bool readFEI(const std::string& filename, 
             FASTEIG_INT &nRows,
             FASTEIG_INT &nCols,
             FASTEIG_INT &nNonZeros,
             std::vector<FASTEIG_INT> &colPtr,
             std::vector<FASTEIG_INT> &rowInd,
             std::vector<FASTEIG_DOUBLE> &values 
     )
{
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    // Read dimensions
    file.read(reinterpret_cast<char*>(&nRows), sizeof(FASTEIG_INT));
    file.read(reinterpret_cast<char*>(&nCols), sizeof(FASTEIG_INT));
    file.read(reinterpret_cast<char*>(&nNonZeros), sizeof(FASTEIG_INT));

    if (!file) {
        throw std::runtime_error("Failed reading matrix header.");
    }

    if (nRows <= 0 || nCols <= 0 || nNonZeros < 0) {
        throw std::runtime_error("Invalid matrix sizes in binary file.");
    }

    // Allocate
    colPtr.resize(static_cast<size_t>(nCols) + 1);
    rowInd.resize(static_cast<size_t>(nNonZeros));
    values.resize(static_cast<size_t>(nNonZeros));

    // Read arrays
    file.read(reinterpret_cast<char*>(colPtr.data()),
            sizeof(FASTEIG_INT) * colPtr.size());

    file.read(reinterpret_cast<char*>(rowInd.data()),
            sizeof(FASTEIG_INT) * rowInd.size());

    file.read(reinterpret_cast<char*>(values.data()),
            sizeof(double) * values.size());

    if (!file) {
        throw std::runtime_error("Failed reading CSC arrays.");
    }

    // Basic consistency checks
    if (colPtr.front() != 0 || colPtr.back() != nNonZeros) {
        throw std::runtime_error("Invalid column pointer array.");
    }

    file.close();
    return true;
}
