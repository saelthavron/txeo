#include "txeo/Predictor.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace txeo {

const std::filesystem::path TEST_MODEL_PATH = "../../../../tests/test_data/model_regression";
const std::filesystem::path INVALID_MODEL_PATH = "invalid/path";

class PredictorTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PredictorTest, ModelLoading) {
  ASSERT_NO_THROW({ Predictor<float> predictor(TEST_MODEL_PATH); });
}

TEST_F(PredictorTest, InvalidModelPath) {
  ASSERT_THROW({ Predictor<float> predictor(INVALID_MODEL_PATH); }, PredictorError);
}

TEST_F(PredictorTest, MetadataHandling) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  const auto &inputs = predictor.get_input_metadata();
  ASSERT_FALSE(inputs.empty());
  EXPECT_EQ(inputs[0].first, "serving_default_dense_8_input:0");
  EXPECT_EQ(inputs[0].second, TensorShape({0, 11}));

  const auto &outputs = predictor.get_output_metadata();
  ASSERT_FALSE(outputs.empty());
  EXPECT_EQ(outputs[0].first, "StatefulPartitionedCall:0");
  EXPECT_EQ(outputs[0].second, TensorShape({0, 1}));
}

TEST_F(PredictorTest, SinglePrediction) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  Tensor<float> input({1, 11}, {0.5869565217391305, 0.24791498520312072, 0.4, 1.0, 0.0, 1.0, 0.0,
                                1.0, 0.0, 0.0, 0.0});

  auto output = predictor.predict(input);

  ASSERT_EQ(output.shape(), TensorShape({1, 1}));
  EXPECT_FLOAT_EQ(trunc(output(0, 0)), 9418.0f);
}

TEST_F(PredictorTest, BatchPrediction) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  std::vector<std::pair<std::string, Tensor<float>>> inputs = {
      {"serving_default_dense_8_input:0",
       Tensor<float>({1, 11}, {0.5869565217391305, 0.24791498520312072, 0.4, 1.0, 0.0, 1.0, 0.0,
                               1.0, 0.0, 0.0, 0.0})}};

  auto outputs = predictor.predict_batch(inputs);

  ASSERT_FALSE(outputs.empty());
  EXPECT_FLOAT_EQ(trunc(outputs[0](0, 0)), 9418.0f);
}

TEST_F(PredictorTest, EnableXLA) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  ASSERT_NO_THROW(predictor.enable_xla(true));

  ASSERT_NO_THROW(predictor.enable_xla(false));
}

TEST_F(PredictorTest, DeviceListing) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  auto devices = predictor.get_devices();
  ASSERT_FALSE(devices.empty());

  bool has_cpu = std::ranges::any_of(devices.begin(), devices.end(),
                                     [](const auto &d) { return d.device_type == "CPU"; });
  EXPECT_TRUE(has_cpu);
}

TEST_F(PredictorTest, ShapeMismatch) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  Tensor<float> input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  ASSERT_THROW({ auto output = predictor.predict(input); }, PredictorError);
}

TEST_F(PredictorTest, InvalidBatchInput) {
  Predictor<float> predictor(TEST_MODEL_PATH);

  std::vector<std::pair<std::string, Tensor<float>>> inputs = {
      {"invalid_input", Tensor<float>({1, 1}, {1.0f})}};

  ASSERT_THROW({ auto outputs = predictor.predict_batch(inputs); }, PredictorError);
}

} // namespace txeo
