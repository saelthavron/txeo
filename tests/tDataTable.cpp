#include "txeo/DataTable.h"
#include "txeo/Matrix.h"
#include <gtest/gtest.h>

TEST(DataTableTest, ConstructWithSpecifiedFeatureAndLabelColumns) {
  txeo::Matrix<double> data({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  txeo::DataTable<double> dt(data, {0, 1}, std::vector<size_t>({2}));

  EXPECT_EQ(dt.x_train().row_size(), 3);
  EXPECT_EQ(dt.x_train().col_size(), 2);
  EXPECT_EQ(dt.y_train().col_size(), 1);
}

TEST(DataTableTest, ConstructWithLabelColumnsOnly) {
  txeo::Matrix<float> data({{1.1f, 2.2f, 3.3f, 4.4f}, {5.5f, 6.6f, 7.7f, 8.8f}});
  txeo::DataTable<float> dt(data, {3});

  EXPECT_EQ(dt.x_dim(), 3);
  EXPECT_EQ(dt.y_dim(), 1);
}

TEST(DataTableTest, ConstructWithEvalSplit) {
  txeo::Matrix<double> data(100, 5);
  txeo::DataTable<double> dt(data, {0, 1, 2}, {3, 4}, 30);

  EXPECT_EQ(dt.x_train().row_size(), 70);
  EXPECT_EQ(dt.x_eval()->row_size(), 30);
}

TEST(DataTableTest, ConstructWithEvalSplitLabelOnly) {
  txeo::Matrix<int> data({{1, 2, 3, 4}, {5, 6, 7, 8}});
  txeo::DataTable<int> dt(data, {3}, 50);

  EXPECT_EQ(dt.row_size(), 1);
  EXPECT_EQ(dt.x_eval()->row_size(), 1);
}

TEST(DataTableTest, ConstructWithEvalAndTestSplit) {
  txeo::Matrix<double> data(1000, 10);
  txeo::DataTable<double> dt(data, {0, 1, 2, 3}, {4, 5}, 20, 10);

  EXPECT_EQ(dt.x_train().row_size(), 700);
  EXPECT_EQ(dt.x_eval()->row_size(), 200);
  EXPECT_EQ(dt.x_test()->row_size(), 100);
}

TEST(DataTableTest, ConstructExplicitTrainingEvalTestSplits) {
  txeo::Matrix<double> X_train({{1.0, 2.0}, {3.0, 4.0}});
  txeo::Matrix<double> y_train({{0.5}, {1.2}});
  txeo::Matrix<double> X_eval({{5.0, 6.0}});
  txeo::Matrix<double> y_eval({{2.1}});
  txeo::Matrix<double> X_test({{7.0, 8.0}});
  txeo::Matrix<double> y_test({{3.0}});

  txeo::DataTable<double> dt_full(X_train, y_train, X_eval, y_eval, X_test, y_test);

  EXPECT_EQ(dt_full.x_train().row_size(), 2);
  EXPECT_EQ(dt_full.x_eval()->row_size(), 1);
  EXPECT_EQ(dt_full.x_test()->row_size(), 1);
}

TEST(DataTableTest, ConstructExplicitTrainingAndEvalSplits) {
  txeo::Matrix<double> X_train({{1.0, 2.0}, {3.0, 4.0}});
  txeo::Matrix<double> y_train({{0.5}, {1.2}});
  txeo::Matrix<double> X_eval({{5.0, 6.0}});
  txeo::Matrix<double> y_eval({{2.1}});

  txeo::DataTable<double> dt_eval(X_train, y_train, X_eval, y_eval);

  EXPECT_EQ(dt_eval.x_train().row_size(), 2);
  EXPECT_EQ(dt_eval.x_eval()->row_size(), 1);
}

TEST(DataTableTest, ConstructTrainingDataOnly) {
  txeo::Matrix<double> X_train({{1.0, 2.0}, {3.0, 4.0}});
  txeo::Matrix<double> y_train({{0.5}, {1.2}});

  txeo::DataTable<double> dt_simple(X_train, y_train);

  EXPECT_EQ(dt_simple.x_train().row_size(), 2);
}