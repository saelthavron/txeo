#include "txeo/Loss.h"
#include "txeo/TensorShape.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(LossTest, MeanSquaredError) {
  txeo::Tensor<float> valid({3}, {1.0f, 2.0f, 3.0f});
  txeo::Loss<float> loss(valid);

  txeo::Tensor<float> pred({3}, {1.5f, 1.8f, 2.9f});
  float expected = ((0.5f * 0.5f) + (0.2f * 0.2f) + (0.1f * 0.1f)) / 3.0f;

  EXPECT_FLOAT_EQ(loss.mean_squared_error(pred), expected);
  EXPECT_FLOAT_EQ(loss.mse(pred), expected);
}

TEST(LossTest, MeanAbsoluteError) {
  txeo::Tensor<double> valid({4}, {5.0, 3.0, 2.0, 7.0});
  txeo::Loss<double> loss(valid);

  txeo::Tensor<double> pred({4}, {4.5, 3.2, 2.5, 6.8});
  double expected = (0.5 + 0.2 + 0.5 + 0.2) / 4.0;

  EXPECT_DOUBLE_EQ(loss.mean_absolute_error(pred), expected);
  EXPECT_DOUBLE_EQ(loss.mae(pred), expected);
}

TEST(LossTest, MeanSquaredLogarithmicError) {
  txeo::Tensor<float> valid({2}, {1.0f, 10.0f});
  txeo::Loss<float> loss(valid);

  txeo::Tensor<float> pred({2}, {1.2f, 9.5f});
  float log_valid1 = std::log1p(1.0f);
  float log_valid2 = std::log1p(10.0f);
  float log_pred1 = std::log1p(1.2f);
  float log_pred2 = std::log1p(9.5f);
  float expected = (pow(log_pred1 - log_valid1, 2) + pow(log_pred2 - log_valid2, 2)) / 2.0f;

  EXPECT_FLOAT_EQ(loss.mean_squared_logarithmic_error(pred), expected);
  EXPECT_FLOAT_EQ(loss.msle(pred), expected);
}

TEST(LossTest, LogCoshError) {
  txeo::Tensor<double> valid({3}, {2.0, 3.0, 5.0});
  txeo::Loss<double> loss(valid);

  txeo::Tensor<double> pred({3}, {1.8, 3.2, 4.9});
  double error1 = 1.8 - 2.0;
  double error2 = 3.2 - 3.0;
  double error3 = 4.9 - 5.0;
  double expected =
      (std::log(std::cosh(error1)) + std::log(std::cosh(error2)) + std::log(std::cosh(error3))) /
      3.0;

  EXPECT_DOUBLE_EQ(loss.log_cosh_error(pred), expected);
  EXPECT_DOUBLE_EQ(loss.lche(pred), expected);
}

TEST(LossTest, ShapeMismatchError) {
  txeo::Tensor<float> valid({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  txeo::Loss<float> loss(valid);

  txeo::Tensor<float> invalid_pred({4});
  EXPECT_THROW(loss.mse(invalid_pred), txeo::LossError);
}

TEST(LossTest, EmptyTensorHandling) {
  txeo::Tensor<double> valid(txeo::TensorShape({0}));

  EXPECT_THROW({ txeo::Loss<double> loss(valid); }, txeo::LossError);
}

TEST(LossTest, InvalidLogarithmicInput) {
  txeo::Tensor<float> valid({2}, {1.0f, -2.0f});
  txeo::Loss<float> loss(valid);

  txeo::Tensor<float> pred({2}, {0.5f, -1.5f});
  EXPECT_THROW(loss.msle(pred), txeo::LossError);
}

TEST(LossTest, ShorthandEquivalence) {
  txeo::Tensor<double> valid({3}, {2.0, 3.0, 4.0});
  txeo::Loss<double> loss(valid);

  txeo::Tensor<double> pred({3}, {1.9, 3.1, 4.0});
  EXPECT_EQ(loss.mse(pred), loss.mean_squared_error(pred));
  EXPECT_EQ(loss.mae(pred), loss.mean_absolute_error(pred));
  EXPECT_EQ(loss.msle(pred), loss.mean_squared_logarithmic_error(pred));
  EXPECT_EQ(loss.lche(pred), loss.log_cosh_error(pred));
}

TEST(LossTest, DefaultLossFunction) {
  txeo::Tensor<float> valid({3}, {1.0f, 2.0f, 3.0f});
  txeo::Loss<float> loss(valid); // Default to MSE

  txeo::Tensor<float> pred({3}, {1.1f, 1.9f, 3.05f});
  EXPECT_FLOAT_EQ(loss.get_loss(pred), loss.mse(pred));
}

TEST(LossTest, SetAndGetMAE) {
  txeo::Tensor<double> valid({4}, {5.0, 3.0, 2.0, 7.0});
  txeo::Loss<double> loss(valid, txeo::LossFunc::MSE);

  loss.set_loss(txeo::LossFunc::MAE);
  txeo::Tensor<double> pred({4}, {4.5, 3.2, 2.5, 6.8});
  EXPECT_DOUBLE_EQ(loss.get_loss(pred), loss.mae(pred));
}

TEST(LossTest, SetAndGetMSLE) {
  txeo::Tensor<float> valid({2}, {1.0f, 10.0f});
  txeo::Loss<float> loss(valid, txeo::LossFunc::MAE);

  loss.set_loss(txeo::LossFunc::MSLE);
  txeo::Tensor<float> pred({2}, {1.2f, 9.5f});
  EXPECT_FLOAT_EQ(loss.get_loss(pred), loss.msle(pred));
}

TEST(LossTest, SetAndGetLCHE) {
  txeo::Tensor<double> valid({3}, {2.0, 3.0, 5.0});
  txeo::Loss<double> loss(valid);

  loss.set_loss(txeo::LossFunc::LCHE);
  txeo::Tensor<double> pred({3}, {1.8, 3.2, 4.9});
  EXPECT_DOUBLE_EQ(loss.get_loss(pred), loss.lche(pred));
}

TEST(LossTest, MultipleFunctionChanges) {
  txeo::Tensor<float> valid({2}, {4.0f, 6.0f});
  txeo::Tensor<float> pred({2}, {3.8f, 6.2f});
  txeo::Loss<float> loss(valid);

  loss.set_loss(txeo::LossFunc::MAE);
  EXPECT_FLOAT_EQ(loss.get_loss(pred), loss.mae(pred));

  loss.set_loss(txeo::LossFunc::MSE);
  EXPECT_FLOAT_EQ(loss.get_loss(pred), loss.mse(pred));

  loss.set_loss(txeo::LossFunc::LCHE);
  EXPECT_NEAR(loss.get_loss(pred), loss.lche(pred), 1e-6);
}