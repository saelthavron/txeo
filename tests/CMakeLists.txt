
#################################################### Test Executable ####################################################

find_package(GTest REQUIRED)

add_executable(
  txeo_tests
  txeo_tests.cpp
  tUtils.cpp
  tTensorShape.cpp
  tTensor.cpp
)  

target_link_libraries(
  txeo_tests  
  GTest::GTest GTest::Main txeo_static 
)

include(GoogleTest)
gtest_discover_tests(txeo_tests)

