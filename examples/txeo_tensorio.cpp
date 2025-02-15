#include "txeo/TensorIO.h"
#include <iostream>
#include <numbers>

int main() {
  // ======================================================================================
  // 1. Basic File Reading (Static Method)
  // ======================================================================================
  {
    std::cout << "\n=== 1. Reading CSV with Header ===\n";
    try {
      auto data = txeo::TensorIO::read_textfile<float>("data.csv", // Assume this file exists
                                                       ',',
                                                       true // Skip header row
      );
      std::cout << "CSV Tensor shape: " << data.shape() << "\n"
                << "First 3 elements: " << data(0, 0) << ", " << data(0, 1) << ", " << data(0, 2)
                << "\n";
    } catch (const txeo::TensorIOError &e) {
      std::cerr << "Read error: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 2. Reading TSV Without Header (Instance Method)
  // ======================================================================================
  {
    std::cout << "\n=== 2. Reading TSV Without Header ===\n";
    txeo::TensorIO reader("data.tsv", '\t'); // Tab-separated

    try {
      auto tensor = reader.read_text_file<int>();
      std::cout << "TSV Tensor:\n" << tensor << "\n";
    } catch (const txeo::TensorIOError &e) {
      std::cerr << "Read error: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 3. Basic File Writing (Static Method)
  // ======================================================================================
  {
    std::cout << "\n=== 3. Writing CSV ===\n";
    txeo::Tensor<int> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    txeo::TensorIO::write_textfile(matrix, "output_matrix.csv",
                                   ',' // CSV format
    );
    std::cout << "Wrote 3x3 matrix to output_matrix.csv\n";
  }

  // ======================================================================================
  // 4. Precision Writing for Floating Points
  // ======================================================================================
  {
    std::cout << "\n=== 4. Precision Writing ===\n";
    txeo::Tensor<double> measurements{{std::numbers::pi, std::numbers::e},
                                      {std::numbers::phi, std::numbers::egamma}};

    txeo::TensorIO::write_textfile(measurements,
                                   4, // 4 decimal places
                                   "precision_data.csv");
    std::cout << "Wrote double tensor with 4-digit precision\n";
  }

  // ======================================================================================
  // 5. Instance-Based Writing with Custom Format
  // ======================================================================================
  {
    std::cout << "\n=== 5. Custom Format Writing ===\n";
    txeo::Tensor<float> sensor_data{{25.4f, 18.9f, 30.1f}, {22.5f, 19.8f, 28.7f}};

    txeo::TensorIO writer("sensor_data.psv", '|');
    try {
      writer.write_text_file(sensor_data, 1); // 1 decimal place
      std::cout << "Created pipe-separated file with 1 decimal precision\n";
    } catch (const txeo::TensorIOError &e) {
      std::cerr << "Write error: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 6. Error Handling Demonstration
  // ======================================================================================
  {
    std::cout << "\n=== 6. Error Handling ===\n";
    try {
      auto invalid = txeo::TensorIO::read_textfile<int>("non_existent_file.dat",
                                                        '$', // Invalid separator
                                                        true);
    } catch (const txeo::TensorIOError &e) {
      std::cerr << "Caught expected error: " << e.what() << "\n";
    }
  }

  return 0;
}