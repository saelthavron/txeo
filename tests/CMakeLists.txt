
#################################################### Test Executable ####################################################

find_package(GTest REQUIRED)

add_executable(
  txeo_tests
  txeo_tests.cpp
  tUtils.cpp
  tTensorShape.cpp
  tTensor.cpp
  tMatrixIO.cpp
  tPredictor.cpp
  tTensorOp.cpp
  tTensorAgg.cpp
  tTensorPart.cpp
  tTensorFunc.cpp
  tMatrix.cpp
  tVector.cpp
  tLoss.cpp
  tOlsGDTrainer.cpp
  tDataTable.cpp
)  

target_link_libraries(
  txeo_tests  
  GTest::gtest GTest::gtest_main txeo_shared 
)

include(GoogleTest)
gtest_discover_tests(txeo_tests)



