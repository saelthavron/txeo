#include "txeo/Predictor.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/TensorPrivate.h"

#include "txeo/detail/PredictorPrivate.h"
#include "txeo/detail/utils.h"
#include <memory>
#include <tensorflow/cc/saved_model/tag_constants.h>

namespace tf = tensorflow;

namespace txeo {

template <typename T>
Predictor<T>::Predictor(std::string model_path) : _impl{std::make_unique<Impl>()} {
  std::unordered_set<std::string> tags{static_cast<const char *>(tf::kSavedModelTagServe)};

  tf::Status status = tf::LoadSavedModel(_impl->session_options, _impl->run_options, model_path,
                                         tags, &_impl->model);
  if (!status.ok())
    throw txeo::PredictorError("Error loading model: " + status.ToString());

  auto meta_graph_def = _impl->model.meta_graph_def;
  auto signature_map = meta_graph_def.signature_def().at("serving_default");

  for (const auto &input : signature_map.inputs()) {
    auto info = input.second;
    if (info.has_name() && info.has_tensor_shape())
      _impl->input_name_shape_map.emplace_back(
          info.name(), txeo::detail::to_txeo_tensor_shape(info.tensor_shape()));
    else if (info.has_name())
      _impl->input_name_shape_map.emplace_back(info.name(), txeo::TensorShape({0}));
  }

  for (const auto &output : signature_map.outputs()) {
    auto info = output.second;
    if (info.has_name() && info.has_tensor_shape())
      _impl->output_name_shape_map.emplace_back(
          info.name(), txeo::detail::to_txeo_tensor_shape(info.tensor_shape()));
    else if (info.has_name())
      _impl->output_name_shape_map.emplace_back(info.name(), txeo::TensorShape({0}));
  }
}

template <typename T>
inline Predictor<T>::~Predictor() {
  auto aux = _impl->model.session->Close();
}

template <typename T>
inline const Predictor<T>::TensorInfo &Predictor<T>::get_input_metadata() const {
  return _impl->input_name_shape_map;
}

template <typename T>
inline const Predictor<T>::TensorInfo &Predictor<T>::get_output_metadata() const {
  return _impl->output_name_shape_map;
}

template <typename T>
inline txeo::Tensor<T> Predictor<T>::predict(const txeo::Tensor<T> input) const {
  if (_impl->input_name_shape_map.size() == 0)
    throw txeo::PredictorError("The loaded model has no input metadata!");
  if (_impl->input_name_shape_map[0].second != input.shape())
    throw txeo::PredictorError("The shapes of the input tensor and the model input do not match!");
  if (_impl->output_name_shape_map.size() == 0)
    throw txeo::PredictorError("The loaded model has no output metadata!");

  auto input_name = _impl->input_name_shape_map[0].first;
  auto output_name = _impl->output_name_shape_map[0].first;
  auto tf_tensor = *input._impl->tf_tensor;
  std::vector<tf::Tensor> outputs;

  auto status = _impl->model.session->Run({{input_name, tf_tensor}}, {output_name}, {}, &outputs);
  if (!status.ok())
    txeo::PredictorError("Error running model: " + status.ToString());

  txeo::Tensor<T> resp{txeo::detail::to_txeo_tensor<T>(outputs[0])};

  return resp;
}

// template <typename T>
// inline std::vector<txeo::Tensor<T>>
// Predictor<T>::predict(const std::vector<txeo::Tensor<T>> inputs) const {

// }

template class Predictor<short>;
template class Predictor<int>;
template class Predictor<long>;
template class Predictor<long long>;
template class Predictor<float>;
template class Predictor<double>;

} // namespace txeo