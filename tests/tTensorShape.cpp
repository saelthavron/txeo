// #include "txeo/TensorShape.h"
// #include <gtest/gtest.h>
// #include <sstream>

// TEST(TensorShapeTest, ConstructFromVector) {
//   std::vector<int64_t> shape = {2, 3, 4};
//   txeo::TensorShape ts(shape);
//   EXPECT_EQ(ts.number_of_axes(), 3);
//   EXPECT_EQ(ts.axes_dims(), shape);
//   EXPECT_TRUE(ts.is_fully_defined());
// }

// TEST(TensorShapeTest, ConstructFromnumber_of_axesAndDim) {
//   txeo::TensorShape ts(3, 5);
//   EXPECT_EQ(ts.number_of_axes(), 3);
//   EXPECT_EQ(ts.axes_dims(), std::vector<int64_t>({5, 5, 5}));
// }

// TEST(TensorShapeTest, CopyConstructor) {
//   txeo::TensorShape ts1({2, 3});
//   txeo::TensorShape ts2(ts1);
//   EXPECT_EQ(ts1, ts2);
// }

// TEST(TensorShapeTest, MoveConstructor) {
//   txeo::TensorShape ts1({2, 3});
//   txeo::TensorShape ts2(std::move(ts1));
//   EXPECT_EQ(ts2.number_of_axes(), 2);
//   EXPECT_EQ(ts2.axes_dims(), std::vector<int64_t>({2, 3}));
// }

// TEST(TensorShapeTest, AxisDim) {
//   txeo::TensorShape ts({4, 5, 6});
//   EXPECT_EQ(ts.axis_dim(1), 5);
//   EXPECT_THROW(
//       {
//         auto result = ts.axis_dim(3);
//         (void)result; // Avoiding incompatibility of gtest with [[nodiscard]]
//       },
//       txeo::TensorShapeError);
//   EXPECT_THROW(
//       {
//         auto result = ts.axis_dim(-1);
//         (void)result; // Avoiding incompatibility of gtest with [[nodiscard]]
//       },
//       txeo::TensorShapeError);
// }

// TEST(TensorShapeTest, InsertAxis) {
//   txeo::TensorShape ts({2, 3});
//   ts.insert_axis(1, 4);
//   EXPECT_EQ(ts.axes_dims(), std::vector<int64_t>({2, 4, 3}));
//   EXPECT_THROW(ts.insert_axis(3, 5), txeo::TensorShapeError);
//   EXPECT_THROW(ts.insert_axis(-1, 5), txeo::TensorShapeError);
//   EXPECT_THROW(ts.insert_axis(1, -5), txeo::TensorShapeError);
// }

// TEST(TensorShapeTest, RemoveAxis) {
//   txeo::TensorShape ts({2, 3, 4});
//   ts.remove_axis(1);
//   EXPECT_EQ(ts.axes_dims(), std::vector<int64_t>({2, 4}));
//   EXPECT_THROW(ts.remove_axis(2), txeo::TensorShapeError);
//   EXPECT_THROW(ts.remove_axis(-1), txeo::TensorShapeError);
// }

// TEST(TensorShapeTest, SetAxisDim) {
//   txeo::TensorShape ts({2, 3, 4});
//   ts.set_dim(1, 7);
//   EXPECT_EQ(ts.axes_dims(), std::vector<int64_t>({2, 7, 4}));
//   EXPECT_THROW(ts.set_dim(3, 5), txeo::TensorShapeError);
//   EXPECT_THROW(ts.set_dim(-1, 5), txeo::TensorShapeError);
// }

// TEST(TensorShapeTest, ComparisonOperators) {
//   txeo::TensorShape ts1({2, 3, 4});
//   txeo::TensorShape ts2({2, 3, 4});
//   txeo::TensorShape ts3({2, 3, 5});
//   EXPECT_TRUE(ts1 == ts2);
//   EXPECT_FALSE(ts1 == ts3);
//   EXPECT_TRUE(ts1 != ts3);
// }

// TEST(TensorShapeTest, StreamOperator) {
//   txeo::TensorShape ts({2, 3, 4});
//   std::stringstream ss;
//   ss << ts;
//   EXPECT_EQ(ss.str(), "[2,3,4]");
// }

#include "txeo/TensorShape.h"
#include <gtest/gtest.h>
#include <sstream>

namespace txeo {

TEST(TensorShapeTest, ConstructorNumberOfAxesDim) {
  TensorShape shape(3, 4);
  EXPECT_EQ(shape.number_of_axes(), 3);
  EXPECT_EQ(shape.axis_dim(0), 4);
  EXPECT_EQ(shape.axis_dim(1), 4);
  EXPECT_EQ(shape.axis_dim(2), 4);

  EXPECT_THROW(TensorShape(-1, 5), TensorShapeError);
  EXPECT_THROW(TensorShape(2, -2), TensorShapeError);
}

TEST(TensorShapeTest, ConstructorVector) {
  TensorShape shape({1, 3, 5});
  EXPECT_EQ(shape.number_of_axes(), 3);
  EXPECT_EQ(shape.axis_dim(0), 1);
  EXPECT_EQ(shape.axis_dim(1), 3);
  EXPECT_EQ(shape.axis_dim(2), 5);

  EXPECT_THROW(TensorShape({2, -3}), TensorShapeError);
}

TEST(TensorShapeTest, CopySemantics) {
  TensorShape original({2, 3, 5});
  TensorShape copy(original);
  EXPECT_EQ(copy, original);

  TensorShape copy_assigned = original;
  EXPECT_EQ(copy_assigned, original);
}

TEST(TensorShapeTest, MoveSemantics) {
  TensorShape original({2, 4, 6});
  TensorShape moved(std::move(original));
  EXPECT_EQ(moved.number_of_axes(), 3);
  EXPECT_EQ(moved.axis_dim(1), 4);

  TensorShape move_assigned = std::move(moved);
  EXPECT_EQ(move_assigned.number_of_axes(), 3);
}

TEST(TensorShapeTest, AxisAccess) {
  TensorShape shape({2, 3, 5});
  EXPECT_EQ(shape.axis_dim(1), 3);
  EXPECT_THROW(
      {
        auto result = shape.axis_dim(3);
        (void)result; // Avoiding incompatibility of gtest with [[nodiscard]]
      },
      TensorShapeError);
  EXPECT_THROW(
      {
        auto result = shape.axis_dim(-1);
        (void)result; // Avoiding incompatibility of gtest with [[nodiscard]]
      },
      TensorShapeError);
}

TEST(TensorShapeTest, AxesDims) {
  TensorShape shape({2, 3, 5});
  auto dims = shape.axes_dims();
  ASSERT_EQ(dims.size(), 3);
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(dims[2], 5);
}

TEST(TensorShapeTest, ShapeModifications) {
  TensorShape shape({1, 2});

  shape.push_axis_back(3);
  EXPECT_EQ(shape.number_of_axes(), 3);
  EXPECT_EQ(shape.axis_dim(2), 3);

  shape.insert_axis(1, 4);
  EXPECT_EQ(shape.axes_dims(), std::vector<int64_t>({1, 4, 2, 3}));

  shape.remove_axis(2);
  EXPECT_EQ(shape.axes_dims(), std::vector<int64_t>({1, 4, 3}));

  shape.set_dim(1, 5);
  EXPECT_EQ(shape.axis_dim(1), 5);

  TensorShape empty_shape(0, 0);
  EXPECT_THROW(empty_shape.push_axis_back(-1), TensorShapeError);
  EXPECT_THROW(shape.insert_axis(5, 2), TensorShapeError);
  EXPECT_THROW(shape.remove_axis(5), TensorShapeError);
  EXPECT_THROW(shape.set_dim(5, 2), TensorShapeError);
}

TEST(TensorShapeTest, ComparisonOperators) {
  TensorShape shape1(3, 4);
  TensorShape shape2({4, 4, 4});
  TensorShape shape3({2, 3, 5});

  EXPECT_TRUE(shape1 == shape2);
  EXPECT_FALSE(shape1 != shape2);
  EXPECT_TRUE(shape1 != shape3);
  EXPECT_FALSE(shape1 == shape3);
}

TEST(TensorShapeTest, FullyDefinedCheck) {
  TensorShape defined_shape({2, 3, 5});

  EXPECT_TRUE(defined_shape.is_fully_defined());
}

TEST(TensorShapeTest, StreamOperator) {
  TensorShape shape({2, 3, 5});
  std::stringstream ss;
  ss << shape;
  EXPECT_EQ(ss.str(), "[2,3,5]");
}

TEST(TensorShapeTest, NumberOfElements) {
  TensorShape shape({2, 3, 5});

  EXPECT_EQ(shape.number_of_elements(), 2 * 3 * 5);
}

TEST(TensorShapeTest, EmptyShape) {
  TensorShape empty_shape(0, 0);
  EXPECT_EQ(empty_shape.number_of_axes(), 0);
  EXPECT_EQ(empty_shape.number_of_elements(), 1); // Scalar
  EXPECT_TRUE(empty_shape.is_fully_defined());
}

} // namespace txeo