#include "txeo/Tensor.h"
#include "txeo/TensorPart.h"
#include <gtest/gtest.h>

namespace txeo {

TEST(TensorPartTest, UnstackAxis0) {
  Tensor<int> tensor_3d({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});

  auto unstacked_tensors = TensorPart<int>::unstack(tensor_3d, 0);

  EXPECT_EQ(unstacked_tensors.size(), 2);

  EXPECT_EQ(unstacked_tensors[0].shape(), TensorShape({2, 3}));
  EXPECT_EQ(unstacked_tensors[1].shape(), TensorShape({2, 3}));

  EXPECT_EQ(unstacked_tensors[0](0, 0), 1);
  EXPECT_EQ(unstacked_tensors[0](0, 1), 2);
  EXPECT_EQ(unstacked_tensors[0](0, 2), 3);
  EXPECT_EQ(unstacked_tensors[0](1, 0), 4);
  EXPECT_EQ(unstacked_tensors[0](1, 1), 5);
  EXPECT_EQ(unstacked_tensors[0](1, 2), 6);

  EXPECT_EQ(unstacked_tensors[1](0, 0), 7);
  EXPECT_EQ(unstacked_tensors[1](0, 1), 8);
  EXPECT_EQ(unstacked_tensors[1](0, 2), 9);
  EXPECT_EQ(unstacked_tensors[1](1, 0), 10);
  EXPECT_EQ(unstacked_tensors[1](1, 1), 11);
  EXPECT_EQ(unstacked_tensors[1](1, 2), 12);
}

TEST(TensorPartTest, UnstackAxis1) {

  Tensor<int> tensor_3d({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});

  auto unstacked_tensors = TensorPart<int>::unstack(tensor_3d, 1);

  EXPECT_EQ(unstacked_tensors.size(), 2);

  EXPECT_EQ(unstacked_tensors[0].shape(), TensorShape({2, 3}));
  EXPECT_EQ(unstacked_tensors[1].shape(), TensorShape({2, 3}));

  EXPECT_EQ(unstacked_tensors[0](0, 0), 1);
  EXPECT_EQ(unstacked_tensors[0](0, 1), 2);
  EXPECT_EQ(unstacked_tensors[0](0, 2), 3);
  EXPECT_EQ(unstacked_tensors[0](1, 0), 7);
  EXPECT_EQ(unstacked_tensors[0](1, 1), 8);
  EXPECT_EQ(unstacked_tensors[0](1, 2), 9);

  EXPECT_EQ(unstacked_tensors[1](0, 0), 4);
  EXPECT_EQ(unstacked_tensors[1](0, 1), 5);
  EXPECT_EQ(unstacked_tensors[1](0, 2), 6);
  EXPECT_EQ(unstacked_tensors[1](1, 0), 10);
  EXPECT_EQ(unstacked_tensors[1](1, 1), 11);
  EXPECT_EQ(unstacked_tensors[1](1, 2), 12);
}

TEST(TensorPartTest, UnstackAxis2) {
  Tensor<int> tensor_3d({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});

  auto unstacked_tensors = TensorPart<int>::unstack(tensor_3d, 2);

  EXPECT_EQ(unstacked_tensors.size(), 3);

  EXPECT_EQ(unstacked_tensors[0].shape(), TensorShape({2, 2}));
  EXPECT_EQ(unstacked_tensors[1].shape(), TensorShape({2, 2}));
  EXPECT_EQ(unstacked_tensors[2].shape(), TensorShape({2, 2}));

  EXPECT_EQ(unstacked_tensors[0](0, 0), 1);
  EXPECT_EQ(unstacked_tensors[0](0, 1), 4);
  EXPECT_EQ(unstacked_tensors[0](1, 0), 7);
  EXPECT_EQ(unstacked_tensors[0](1, 1), 10);

  EXPECT_EQ(unstacked_tensors[1](0, 0), 2);
  EXPECT_EQ(unstacked_tensors[1](0, 1), 5);
  EXPECT_EQ(unstacked_tensors[1](1, 0), 8);
  EXPECT_EQ(unstacked_tensors[1](1, 1), 11);

  EXPECT_EQ(unstacked_tensors[2](0, 0), 3);
  EXPECT_EQ(unstacked_tensors[2](0, 1), 6);
  EXPECT_EQ(unstacked_tensors[2](1, 0), 9);
  EXPECT_EQ(unstacked_tensors[2](1, 1), 12);
}

TEST(TensorPartTest, SliceFirstAxis) {

  txeo::Tensor<int> tensor({2, 3});
  tensor(0, 0) = 1;
  tensor(0, 1) = 2;
  tensor(0, 2) = 3;
  tensor(1, 0) = 4;
  tensor(1, 1) = 5;
  tensor(1, 2) = 6;

  auto sliced_tensor = TensorPart<int>::slice(tensor, 0, 1);

  EXPECT_EQ(sliced_tensor.shape().axis_dim(0), 1);
  EXPECT_EQ(sliced_tensor.shape().axis_dim(1), 3);

  EXPECT_EQ(sliced_tensor(0, 0), 1);
  EXPECT_EQ(sliced_tensor(0, 1), 2);
  EXPECT_EQ(sliced_tensor(0, 2), 3);
}

TEST(TensorPartTest, SliceSecondRow) {

  txeo::Tensor<int> tensor({2, 3});
  tensor(0, 0) = 1;
  tensor(0, 1) = 2;
  tensor(0, 2) = 3;
  tensor(1, 0) = 4;
  tensor(1, 1) = 5;
  tensor(1, 2) = 6;

  auto sliced_tensor = TensorPart<int>::slice(tensor, 1, 2);

  EXPECT_EQ(sliced_tensor.shape().axis_dim(0), 1);
  EXPECT_EQ(sliced_tensor.shape().axis_dim(1), 3);

  EXPECT_EQ(sliced_tensor(0, 0), 4);
  EXPECT_EQ(sliced_tensor(0, 1), 5);
  EXPECT_EQ(sliced_tensor(0, 2), 6);
}

TEST(TensorPartTest, SliceMultipleRows) {

  txeo::Tensor<int> tensor({3, 3});
  tensor(0, 0) = 1;
  tensor(0, 1) = 2;
  tensor(0, 2) = 3;
  tensor(1, 0) = 4;
  tensor(1, 1) = 5;
  tensor(1, 2) = 6;
  tensor(2, 0) = 7;
  tensor(2, 1) = 8;
  tensor(2, 2) = 9;

  auto sliced_tensor = TensorPart<int>::slice(tensor, 0, 2);

  EXPECT_EQ(sliced_tensor.shape().axis_dim(0), 2);
  EXPECT_EQ(sliced_tensor.shape().axis_dim(1), 3);

  EXPECT_EQ(sliced_tensor(0, 0), 1);
  EXPECT_EQ(sliced_tensor(0, 1), 2);
  EXPECT_EQ(sliced_tensor(0, 2), 3);
  EXPECT_EQ(sliced_tensor(1, 0), 4);
  EXPECT_EQ(sliced_tensor(1, 1), 5);
  EXPECT_EQ(sliced_tensor(1, 2), 6);
}

TEST(MatrixTest, SubMatrixColsSelectsCorrectColumns) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  auto sub = TensorPart<int>::sub_matrix_cols(mat, {0, 2});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 2);
  EXPECT_EQ(sub(0, 0), 1);
  EXPECT_EQ(sub(0, 1), 3);
  EXPECT_EQ(sub(1, 0), 4);
  EXPECT_EQ(sub(1, 1), 6);
}

TEST(MatrixTest, SubMatrixColsReversedColumns) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  auto sub = TensorPart<int>::sub_matrix_cols(mat, {2, 1, 0});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 3);
  EXPECT_EQ(sub(0, 0), 3);
  EXPECT_EQ(sub(0, 1), 2);
  EXPECT_EQ(sub(0, 2), 1);
  EXPECT_EQ(sub(1, 0), 6);
  EXPECT_EQ(sub(1, 1), 5);
  EXPECT_EQ(sub(1, 2), 4);
}

TEST(MatrixTest, SubMatrixColsSingleColumn) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  auto sub = TensorPart<int>::sub_matrix_cols(mat, {1});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 1);
  EXPECT_EQ(sub(0, 0), 2);
  EXPECT_EQ(sub(1, 0), 5);
}

TEST(MatrixTest, SubMatrixColsDuplicateColumns) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  auto sub = TensorPart<int>::sub_matrix_cols(mat, {0, 0});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 2);
  EXPECT_EQ(sub(0, 0), 1);
  EXPECT_EQ(sub(0, 1), 1);
  EXPECT_EQ(sub(1, 0), 4);
  EXPECT_EQ(sub(1, 1), 4);
}

TEST(MatrixTest, SubMatrixRowsSelectsCorrectRows) {
  txeo::Matrix<int> mat({{1, 2}, {3, 4}, {5, 6}});
  auto sub = TensorPart<int>::sub_matrix_rows(mat, {2, 0});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 2);
  EXPECT_EQ(sub(0, 0), 5);
  EXPECT_EQ(sub(0, 1), 6);
  EXPECT_EQ(sub(1, 0), 1);
  EXPECT_EQ(sub(1, 1), 2);
}

TEST(MatrixTest, SubMatrixRowsSingleRow) {
  txeo::Matrix<int> mat({{1, 2, 3}, {4, 5, 6}});
  auto sub = TensorPart<int>::sub_matrix_rows(mat, {0});

  ASSERT_EQ(sub.row_size(), 1);
  ASSERT_EQ(sub.col_size(), 3);
  EXPECT_EQ(sub(0, 0), 1);
  EXPECT_EQ(sub(0, 1), 2);
  EXPECT_EQ(sub(0, 2), 3);
}

TEST(MatrixTest, SubMatrixRowsDuplicateRows) {
  txeo::Matrix<int> mat({{1, 2}, {3, 4}, {5, 6}});
  auto sub = TensorPart<int>::sub_matrix_rows(mat, {1, 1});

  ASSERT_EQ(sub.row_size(), 2);
  ASSERT_EQ(sub.col_size(), 2);
  EXPECT_EQ(sub(0, 0), 3);
  EXPECT_EQ(sub(0, 1), 4);
  EXPECT_EQ(sub(1, 0), 3);
  EXPECT_EQ(sub(1, 1), 4);
}

} // namespace txeo