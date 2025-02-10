#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/utils.h"
#include <cstddef>
#include <initializer_list>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
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

template <typename T>
void fill_data_shape(const std::initializer_list<std::initializer_list<T>> &list,
                     std::vector<T> &flat_data, std::vector<size_t> &shape) {

  shape.emplace_back(list.size());
  std::vector<std::initializer_list<T>> v_list(list);
  for (size_t i{1}; i < v_list.size(); ++i)
    if (v_list[i].size() != v_list[i - 1].size())
      throw txeo::TensorError("Tensor initialization is inconsistent!");

  shape.emplace_back(v_list[0].size());
  for (auto &item : v_list)
    for (auto &subitem : item)
      flat_data.emplace_back(subitem);
}

template <typename T>
void fill_data_shape(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &list,
    std::vector<T> &flat_data, std::vector<size_t> &shape) {

  shape.emplace_back(list.size());
  std::vector<std::initializer_list<std::initializer_list<T>>> v_list(list);
  for (size_t i{1}; i < v_list.size(); ++i)
    if (v_list[i].size() != v_list[i - 1].size())
      throw txeo::TensorError("Tensor initialization is inconsistent!");

  shape.emplace_back(v_list[0].size());
  bool emplaced{false};
  for (size_t i{0}; i < v_list.size(); ++i) {
    std::vector<std::initializer_list<T>> v_sublist(v_list[i]);
    for (size_t i{1}; i < v_sublist.size(); ++i)
      if (v_sublist[i].size() != v_sublist[i - 1].size())
        throw txeo::TensorError("Tensor initialization is inconsistent!");

    if (!emplaced) {
      shape.emplace_back(v_sublist[0].size());
      emplaced = true;
    }
    for (auto &item : v_sublist)
      for (auto &subitem : item)
        flat_data.emplace_back(subitem);
  }
}

int main() {

  // std::vector<double> flat_data;
  // std::vector<size_t> shape;

  // fill_data_shape<double>({{{1, 2, 3}, {4, 5, 6}},
  //                          {{7, 8, 9}, {10, 11, 12}},
  //                          {{13, 14, 15}, {16, 17, 18}},
  //                          {{19, 20, 21}, {22, 23, 24}}},
  //                         flat_data, shape);

  // for (auto &item : flat_data) {
  //   std::cout << item << " ";
  // }
  // std::cout << std::endl;

  // for (auto &item : shape) {
  //   std::cout << item << " ";
  // }
  // std::cout << std::endl;

  txeo::Tensor<double> a(
      txeo::TensorShape({4, 2, 3}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 444, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  std::cout << a << std::endl;

  return 0;
}