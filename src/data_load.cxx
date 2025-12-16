
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
/*
// src/data_load.cxx

#include <cstdlib>  // For system() to call Python script

// Function to load the dataset and print it (for debugging)
void loadData(const std::string& inputFile) {
    std::ifstream file(inputFile);
    std::string line;
    std::vector<std::vector<std::string>> dataset;

    // Read CSV file and store it in a vector of vectors
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        dataset.push_back(row);
    }

    // Print the loaded data (for debugging)
    for (const auto& row : dataset) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }

    // Call the Python script to process the data
    std::cout << "Calling Python script for further processing...\n";
    system("python ./python/model.py");
}

int main() {
    // Load the data from 'creditcard.csv'
    loadData("data/creditcard.csv");
    return 0;
}

#include <Python.h>  // Include Python API
#include <iostream>
#include <string>

void loadData(const std::string& inputFile) {
    std::cout << "Loading data from: " << inputFile << "\n";

    // Initialize Python interpreter
    Py_Initialize();

    // Build Python command string
    std::string cmd = "import sys\n"
    "sys.path.append('./python')\n"  // path to your Python scripts
    "from model import run_pipeline\n"
    "run_pipeline('" + inputFile + "')\n";

    // Run Python code
    PyRun_SimpleString(cmd.c_str());

    // Finalize Python interpreter
    Py_Finalize();
}
*/
