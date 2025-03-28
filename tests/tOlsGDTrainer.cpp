#include <gtest/gtest.h>

#include "txeo/Matrix.h"
#include "txeo/OlsGDTrainer.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/types.h"

namespace txeo {

TEST(OlsGDTrainerTest, PredictOutputDimensions) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);
  trainer.enable_variable_lr();
  trainer.fit(10, LossFunc::MSE);

  Matrix<double> input(2, 1, {4.0, 5.0});
  auto result = trainer.predict(input);

  ASSERT_EQ(result.shape().axis_dim(0), 2);
  ASSERT_EQ(result.shape().axis_dim(1), 1);
  ASSERT_EQ(result.order(), 2);
}

TEST(OlsGDTrainerTest, LearningRateConfiguration) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.set_learning_rate(0.123);
  EXPECT_DOUBLE_EQ(trainer.learning_rate(), 0.123);
}

TEST(OlsGDTrainerTest, VariableLearningRateSwitches) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.enable_variable_lr();
  trainer.fit(10, LossFunc::MSE);
  trainer.disable_variable_lr();
  trainer.fit(10, LossFunc::MSE);
  SUCCEED();
}

TEST(OlsGDTrainerTest, WeightBiasMatrixDimensions) {
  Matrix<double> x_train(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Matrix<double> y_train(3, 1, {3.0, 7.0, 11.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.fit(100, LossFunc::MSE);
  const auto &wb = trainer.weight_bias();

  ASSERT_EQ(wb.row_size(), 3);
  ASSERT_EQ(wb.col_size(), 1);
}

TEST(OlsGDTrainerTest, ToleranceConfiguration) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.set_tolerance(1e-5);
  EXPECT_DOUBLE_EQ(trainer.tolerance(), 1e-5);
}

TEST(OlsGDTrainerTest, ConvergenceDetection) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.fit(1000, LossFunc::MSE);
  EXPECT_TRUE(trainer.is_converged());
  EXPECT_TRUE(trainer.is_trained());
}

TEST(OlsGDTrainerTest, WeightUpdateDuringTraining) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.fit(10, LossFunc::MSE);
  const auto trained_weights = trainer.weight_bias();

  EXPECT_GT(trained_weights(0, 0), 0.0);
}

TEST(OlsGDTrainerTest, EarlyConvergenceWithHighTolerance) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(x_train, y_train);

  trainer.set_tolerance(0.1);
  trainer.enable_variable_lr();
  trainer.fit(30, LossFunc::MSE);
  EXPECT_TRUE(trainer.is_converged());
}

} // namespace txeo