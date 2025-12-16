#include <iostream>

// Forward declaration of your function from data_load.cxx
void loadData(const std::string& inputFile);

int main() {
    std::cout << "Starting Fraud Detection Pipeline...\n";
    
    // Call existing data loading + python execution pipeline
    loadData("data/creditcard.csv");

    std::cout << "Finished.\n";
    return 0;
}

