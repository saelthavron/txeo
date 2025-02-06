#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include <cstddef>
#include <cstdint>
#include <ios>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#include <type_traits>
#include <vector>

namespace tf = tensorflow;

// void create_tensor() {
//   tf::Tensor S(tf::DT_DOUBLE, tf::TensorShape({}));
//   auto scal = S.scalar<double>();
//   scal() = 10;
//   std::cout << "Scalar: " << scal << std::endl;
//   std::cout << "Scalar dimensions: " << scal.NumDimensions << std::endl;

//   tf::Tensor V(tf::DT_DOUBLE, tf::TensorShape({3}));
//   auto vec = V.vec<double>();
//   vec(0) = 1;
//   vec(1) = 2;
//   vec(2) = 3;
//   std::cout << "Vector: " << vec << std::endl;
//   std::cout << "Vector dimensions: " << vec.NumDimensions << std::endl;

//   tf::Tensor M(tf::DT_DOUBLE, tf::TensorShape({3, 3}));
//   auto mat = M.matrix<double>();
//   mat(0, 0) = 1.0;
//   mat(1, 0) = 2.0;
//   mat(2, 0) = 3.0;
//   mat(0, 1) = 4.0;
//   mat(1, 1) = 5.0;
//   mat(2, 1) = 6.0;
//   mat(0, 2) = 7.0;
//   mat(1, 2) = 8.0;
//   mat(2, 2) = 9.0;
//   std::cout << "Matrix: \n" << mat << std::endl;
//   std::cout << "Matrix dimensions: " << mat.NumDimensions << std::endl;

//   tf::Tensor M1(tf::DT_DOUBLE, tf::TensorShape({2, 3}));
//   auto mat1 = M1.matrix<double>();
//   mat1.setZero();
//   std::cout << "Zero Matrix: \n" << mat1 << std::endl;

//   tf::Tensor M2(tf::DT_DOUBLE, tf::TensorShape({4, 2}));
//   auto mat2 = M2.matrix<double>();
//   mat2.setConstant(2.6);
//   std::cout << "Value Matrix: \n" << mat2 << std::endl;
// }

// int multby2(const int &a) {
//   return 2 * a;
// }

// class Foo {
//   public:
//     tf::Tensor M = tf::Tensor(tf::DT_DOUBLE, tf::TensorShape({3, 3}));

//     template <typename... Args>
//     double &operator()(Args... args) {
//       static_assert(((std::is_integral_v<Args>) && ...), "All arguments must be integers!");
//       auto aux = M.tensor<double, 2>();
//       //      aux.setConstant(2576.23);
//       return aux(args...);
//     }
// };

size_t calc_flat_index(const std::vector<size_t> &indexes, const tf::TensorShape &sizes) {
  size_t accum_sizes{1};
  size_t resp{indexes.back()};

  const size_t *idx_ptr = indexes.data();

  for (size_t i = indexes.size() - 1; i > 0; --i) {
    accum_sizes *= sizes.dim_size(i);
    resp += idx_ptr[i - 1] * accum_sizes;
  }

  return resp;
}

int main() {

  tf::Tensor V(tensorflow::DT_DOUBLE, {125});
  auto aux = V.vec<double>();
  aux.setConstant(25.6);

  aux(124) = 10;

  // tf::Tensor M(tf::DT_DOUBLE, tf::TensorShape({5, 5, 5}));

  // if (!M.CopyFrom(V, M.shape()))
  //   std::cout << "Deu pau!\n";

  // auto aux2 = M.tensor<double, 3>();
  // //  M.Slice(int64_t dim0_start, int64_t dim0_limit)

  std::cout << aux(-1) << std::endl;

  return 0;
}