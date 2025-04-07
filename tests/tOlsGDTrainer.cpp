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
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));
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
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.set_learning_rate(0.123);
  EXPECT_DOUBLE_EQ(trainer.learning_rate(), 0.123);
}

TEST(OlsGDTrainerTest, VariableLearningRateSwitches) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.enable_variable_lr();
  trainer.fit(10, LossFunc::MSE);
  trainer.disable_variable_lr();
  trainer.fit(10, LossFunc::MSE);
  SUCCEED();
}

TEST(OlsGDTrainerTest, WeightBiasMatrixDimensions) {
  Matrix<double> x_train(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Matrix<double> y_train(3, 1, {3.0, 7.0, 11.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.fit(100, LossFunc::MSE);
  const auto &wb = trainer.weight_bias();

  ASSERT_EQ(wb.row_size(), 3);
  ASSERT_EQ(wb.col_size(), 1);
}

TEST(OlsGDTrainerTest, ToleranceConfiguration) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.set_tolerance(1e-5);
  EXPECT_DOUBLE_EQ(trainer.tolerance(), 1e-5);
}

TEST(OlsGDTrainerTest, ConvergenceDetection) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.fit(1000, LossFunc::MSE);
  EXPECT_TRUE(trainer.is_converged());
  EXPECT_TRUE(trainer.is_trained());
}

TEST(OlsGDTrainerTest, WeightUpdateDuringTraining) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.fit(10, LossFunc::MSE);
  const auto trained_weights = trainer.weight_bias();

  EXPECT_GT(trained_weights(0, 0), 0.0);
}

TEST(OlsGDTrainerTest, EarlyConvergenceWithHighTolerance) {
  Matrix<double> x_train(3, 1, {1.0, 2.0, 3.0});
  Matrix<double> y_train(3, 1, {2.0, 4.0, 6.0});
  OlsGDTrainer<double> trainer(DataTable<double>(x_train, y_train));

  trainer.set_tolerance(0.1);
  trainer.enable_variable_lr();
  trainer.fit(30, LossFunc::MSE);
  EXPECT_TRUE(trainer.is_converged());
}

TEST(OlsGDTrainerTest, DataTableAccess) {

  Matrix<double> data(100, 4);
  DataTable<double> dt(data, {0, 1}, {2, 3}, 20, 10);

  OlsGDTrainer<double> trainer(dt);

  const auto &table = trainer.data_table();
  EXPECT_EQ(table.x_train().row_size(), 70);
  EXPECT_EQ(table.x_test()->col_size(), 2);
  EXPECT_EQ(table.y_dim(), 2);
}

TEST(OlsGDTrainerTest, EvaluateTestWithValidData) {

  Matrix<double> X_train{{1.0}, {2.0}, {3.0}};
  Matrix<double> y_train{{3.1}, {5.2}, {7.3}};
  Matrix<double> X_test{{4.0}, {5.0}};
  Matrix<double> y_test{{9.4}, {11.5}};

  DataTable<double> dt(X_train, y_train, X_train, y_train, X_test, y_test);
  OlsGDTrainer<double> trainer(dt);

  trainer.fit(100, LossFunc::MSE);

  double loss = trainer.compute_test_loss(LossFunc::MSE);

  EXPECT_GT(loss, 0.0);
  EXPECT_LT(loss, 2.0);
}

TEST(OlsGDTrainerTest, EvaluatePrediction) {

  txeo::Matrix<double> data(4, 2, {1, 3, 2, 6, 3, 9, 5, 15});
  txeo::OlsGDTrainer<double> trainer{txeo::DataTable<double>{std::move(data), {1}}};
  trainer.enable_feature_norm(txeo::NormalizationType::MIN_MAX);
  trainer.enable_variable_lr();
  trainer.fit(100, txeo::LossFunc::MAE, 5);

  txeo::Matrix<double> x(1, 1, {4});
  auto &&res1 = trainer.predict(x);

  EXPECT_NEAR(res1(), 12.0, 1e-4);

  trainer.disable_feature_norm();
  trainer.fit(100, txeo::LossFunc::MAE, 5);

  auto &&res2 = trainer.predict(x);
  EXPECT_NEAR(res2(), 12.0, 1e-3);
}

TEST(OlsGDTrainerTest, EvaluateTestWithoutTestSplit) {

  Matrix<double> X{{1.0}, {2.0}};
  Matrix<double> y{{3.0}, {5.0}};
  DataTable<double> dt(X, y);

  OlsGDTrainer<double> trainer(dt);
  trainer.fit(10, LossFunc::MSE);

  EXPECT_THROW(trainer.compute_test_loss(LossFunc::MSE), TrainerError);
}

TEST(OlsGDTrainerTest, EvaluateTestEdgeCases) {

  Matrix<double> data(10, 2);
  DataTable<double> dt(data, {0}, std::vector<size_t>{1}, 20);

  OlsGDTrainer<double> trainer(dt);
  trainer.fit(10, LossFunc::MSE);

  EXPECT_THROW(trainer.compute_test_loss(LossFunc::MSE), TrainerError);
}

} // namespace txeo