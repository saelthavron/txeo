
#################################################### Test Executable ####################################################

find_package(GTest REQUIRED)

add_executable(
  txeo_tests
  txeo_tests.cpp
  tUtils.cpp
  tTensorShape.cpp
  tTensor.cpp
  tTensorIO.cpp
  tPredictor.cpp
  tTensorOp.cpp
  tTensorAgg.cpp
)  

target_link_libraries(
  txeo_tests  
  GTest::gtest GTest::gtest_main txeo_shared 
)

include(GoogleTest)
gtest_discover_tests(txeo_tests)



