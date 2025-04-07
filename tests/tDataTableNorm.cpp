#include "txeo/DataTable.h"
#include "txeo/DataTableNorm.h"
#include <gtest/gtest.h>

using namespace txeo;

// Helper function to create test data
template <typename T>
DataTable<T> create_sample_datatable() {
  Matrix<T> data({{1.0, 2.0, 0.0}, {3.0, 4.0, 1.0}, {5.0, 6.0, 2.0}});
  return DataTable<T>(std::move(data), {2});
}

TEST(DataTableNormTest, ConstructorInitialization) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt, NormalizationType::MIN_MAX);

  EXPECT_EQ(&normalizer.data_table(), &dt);
  EXPECT_EQ(normalizer.type(), NormalizationType::MIN_MAX);
}

TEST(DataTableNormTest, NormalizeMinMax) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt, NormalizationType::MIN_MAX);

  Matrix<double> input({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  auto result = normalizer.normalize(input);

  // Expected min-max normalized values for columns 0 and 1
  EXPECT_DOUBLE_EQ(result(0, 0), 0.0); // (1-1)/(5-1)
  EXPECT_DOUBLE_EQ(result(1, 0), 0.5); // (3-1)/(5-1)
  EXPECT_DOUBLE_EQ(result(0, 1), 0.0); // (2-2)/(6-2)
  EXPECT_DOUBLE_EQ(result(1, 1), 0.5); // (4-2)/(6-2)
}

TEST(DataTableNormTest, NormalizeZScore) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt, NormalizationType::Z_SCORE);

  Matrix<double> input({{3.0, 0.0}, {3.0, 1.0}, {3.0, 2.0}}); // Column 0 values
  auto result = normalizer.normalize(input);

  // μ=3.0, σ=1.63299 for column 0
  const double eps = 1e-5;
  EXPECT_NEAR(result(0, 0), (3.0 - 3.0) / 1.63299, eps);
  EXPECT_NEAR(result(1, 0), (3.0 - 3.0) / 1.63299, eps);
}

TEST(DataTableNormTest, NormalizeThrowsWhenUninitialized) {
  DataTableNorm<double> normalizer;
  Matrix<double> input({{1.0}});

  EXPECT_THROW(normalizer.normalize(input), DataTableNormError);
}

TEST(DataTableNormTest, SetDataTableUpdatesParameters) {
  auto dt1 = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt1, NormalizationType::MIN_MAX);

  Matrix<double> new_data({{10.0, 20.0}});
  DataTable<double> dt2(std::move(new_data), {1});
  normalizer.set_data_table(dt2);

  Matrix<double> input({{10.0}, {20.0}});
  auto result = normalizer.normalize(input);
  EXPECT_DOUBLE_EQ(result(0, 0), 0.0); // New min=10
  EXPECT_DOUBLE_EQ(result(0, 0), 0.0); // New max=20
}

TEST(DataTableNormTest, ConstantFeatureHandling) {
  Matrix<double> data({{5.0, 0.0}, {5.0, 1.0}, {5.0, 2.0}});
  DataTable<double> dt(std::move(data), {1});
  DataTableNorm<double> normalizer(dt, NormalizationType::MIN_MAX);

  Matrix<double> input({{5.0}, {5.0}});
  auto result = normalizer.normalize(input);

  // Constant feature should be zeroed
  EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
}

TEST(DataTableNormTest, XTrainNormalized) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt, NormalizationType::MIN_MAX);

  auto normalized_train = normalizer.x_train_normalized();
  const auto &original_train = dt.x_train();

  EXPECT_TRUE(original_train.shape() == txeo::TensorShape({3, 2}));

  // Verify first column normalization
  EXPECT_DOUBLE_EQ(normalized_train(0, 0), 0.0); // min=1
  EXPECT_DOUBLE_EQ(normalized_train(2, 0), 1.0); // max=5
}

TEST(DataTableNormTest, NormalizeRvalueSemantics) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> normalizer(dt);

  Matrix<double> input(1000, 2); // Large matrix
  auto original_ptr = input.data();
  auto result = normalizer.normalize(std::move(input));

  // Verify move semantics preserved storage
  EXPECT_EQ(result.data(), original_ptr);
}

TEST(DataTableNormTest, TypeGetterReturnsCorrectType) {
  auto dt = create_sample_datatable<double>();
  DataTableNorm<double> minmax_norm(dt, NormalizationType::MIN_MAX);
  DataTableNorm<double> zscore_norm(dt, NormalizationType::Z_SCORE);

  EXPECT_EQ(minmax_norm.type(), NormalizationType::MIN_MAX);
  EXPECT_EQ(zscore_norm.type(), NormalizationType::Z_SCORE);
}