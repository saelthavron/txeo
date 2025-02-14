#include "txeo/TensorIO.h"
#include <iostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>

namespace tf = tensorflow;

#include <iostream>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

#include "txeo/Predictor.h"

int main() {

  txeo::Predictor pred{"/home/roberto/my_works/personal/hello_python/model_regression"};

  const auto &in_meta = pred.get_input_metadata();
  const auto &out_meta = pred.get_output_metadata();

  std::cout << "INPUTS:" << std::endl;
  for (auto &item : in_meta) {
    std::cout << item.first << ":" << std::endl;
    std::cout << item.second << ":" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "OUTPUTS:" << std::endl;
  for (auto &item : out_meta) {
    std::cout << item.first << ":" << std::endl;
    std::cout << item.second << ":" << std::endl;
  }

  txeo::TensorIO io{"/home/roberto/my_works/personal/txeo-tf/tests/teste.txt"};
  txeo::Tensor<float> input = io.read_text_file<float>(true);

  auto resp = pred.predict(input);

  txeo::TensorIO::write_textfile(resp, "/home/roberto/my_works/personal/txeo-tf/tests/w_teste.txt");

  // tf::SavedModelBundle model;
  // tf::SessionOptions session_options;
  // tf::RunOptions run_options;
  // std::unordered_set<std::string> tags{static_cast<const char *>(tf::kSavedModelTagServe)};

  // std::string model_dir = "/home/roberto/my_works/personal/hello_python/model_regression";
  // tf::Status status = tf::LoadSavedModel(session_options, run_options, model_dir, tags,
  // &model); if (!status.ok()) {
  //   std::cerr << "Error loading model: " << status.ToString() << std::endl;
  //   return 1;
  // }
  // std::cout << "Model loaded successfully!" << std::endl;

  // tensorflow::MetaGraphDef meta_graph_def = model.meta_graph_def;
  // tensorflow::SignatureDef signature_map =
  // meta_graph_def.signature_def().at("serving_default");

  // // const auto &input_name = (*std::begin(signature_map.inputs())).second.name();

  // std::cout << "Inputs:\n";
  // for (const auto &input : signature_map.inputs()) {
  //   std::cout << input.second.has_name() << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << "Outputs:\n";
  // for (const auto &output : signature_map.outputs()) {
  //   std::cout << output.second.name() << std::endl;
  // }

  // // Prepare input tensor
  // std::vector<float> input_data = {0.5869565217391305f,
  //                                  0.2479149852031207f,
  //                                  0.4f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  0.0f,
  //                                  0.0f,
  //                                  0.3913043478260869f,
  //                                  0.3782620392789884f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.2939198278181328f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  1.0f,
  //                                  0.0f,
  //                                  0.0f};

  // //

  // tf::Tensor input_tensor(tf::DT_FLOAT, tf::TensorShape({3, 11}));
  // auto input_map = input_tensor.flat<float>();
  // for (size_t i = 0; i < input_data.size(); ++i) {
  //   input_map(i) = input_data[i];
  // }

  // // Run model inference
  // std::vector<std::pair<std::string, tf::Tensor>> inputs = {{input_name, input_tensor}};
  // std::vector<tf::Tensor> outputs;

  // status = model.session->Run(inputs, {output_name}, {}, &outputs);
  // if (!status.ok()) {
  //   std::cerr << "Error running model: " << status.ToString() << std::endl;
  //   return 1;
  // }

  // // Extract and print predictions
  // auto output_map = outputs[0].flat<float>();
  // std::cout << "Model Prediction: ";
  // for (int i = 0; i < output_map.size(); i++) {
  //   std::cout << output_map(i) << " ";
  // }
  // std::cout << std::endl;

  // if (!model.session->Close().ok()) {
  //   std::cerr << "Error closing session: " << status.ToString() << std::endl;
  //   return 1;
  // }

  return 0;
}
