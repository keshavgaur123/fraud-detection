
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

void loadData(const std::string& inputFile) {
    std::ifstream file(inputFile);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << inputFile << std::endl;
        return;
    }

    std::string line;
    std::vector<std::vector<std::string>> dataset;

    // Read CSV file
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        dataset.push_back(row);
    }

    file.close();

    // Debug: print loaded data
    for (const auto& row : dataset) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }

    // Call Python script with input file as argument
    std::cout << "Calling Python script for further processing...\n";
    std::string cmd = "python ./python/model.py " + inputFile;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Python script failed with exit code " << ret << std::endl;
    }
}
