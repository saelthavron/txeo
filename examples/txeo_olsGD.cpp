#include "txeo/MatrixIO.h"
#include "txeo/OlsGDTrainer.h"
#include "txeo/Tensor.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorPart.h"

#include <iostream>

int main() {
  // ======================================================================================
  // 1. Basic Training
  // ======================================================================================

  // Data downloaded at
  // https://media.geeksforgeeks.org/wp-content/uploads/20240319120216/housing.csv
  auto train_data =
      txeo::MatrixIO::one_hot_encode_text_file("housing.csv", ',', true, "housing_one_hot.csv");

  auto x_train = txeo::TensorPart<double>::sub_matrix_cols_exclude(
      train_data.read_text_file<double>(true), {8});
  auto y_train =
      txeo::TensorPart<double>::sub_matrix_cols(train_data.read_text_file<double>(true), {8});

  x_train.normalize_columns(txeo::NormalizationType::MIN_MAX);

  std::cout << "X: " << x_train.shape() << std::endl;
  std::cout << "Y: " << y_train.shape() << std::endl;

  txeo::OlsGDTrainer<double> ols{x_train, y_train};

  ols.enable_variable_lr();
  ols.fit(100, txeo::LossFunc::MAE);

  std::cout << "Weight-Bias: " << ols.weight_bias() << std::endl;
  std::cout << "Minimum loss: " << ols.min_loss() << std::endl;
  std::cout << "min Y:" << txeo::TensorAgg<double>::reduce_min(y_train, {0})() << std::endl;

  return 0;
}