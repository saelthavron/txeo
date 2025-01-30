# cmake/FindTensorFlow.cmake
include(FindPackageHandleStandardArgs)

# Find TensorFlow
find_path(TensorFlow_INCLUDE_DIR
  NAMES tensorflow/core/public/session.h
  HINTS
    ${TensorFlow_ROOT_DIR}/include
    $ENV{TensorFlow_ROOT_DIR}/include
    /usr/local/include/tensorflow
    /opt/tensorflow/include
)

find_library(TensorFlow_CC_LIBRARY
  NAMES tensorflow_cc
  HINTS
    ${TensorFlow_ROOT_DIR}/lib
    $ENV{TensorFlow_ROOT_DIR}/lib
    /usr/local/lib
    /opt/tensorflow/lib
)

find_library(TensorFlow_FRAMEWORK_LIBRARY
  NAMES tensorflow_framework
  HINTS
    ${TensorFlow_ROOT_DIR}/lib
    $ENV{TensorFlow_ROOT_DIR}/lib
    /usr/local/lib
    /opt/tensorflow/lib
)

# Find Protobuf 
find_path(Protobuf_INCLUDE_DIR
  NAMES google/protobuf/message.h
  HINTS
    ${Protobuf_ROOT_DIR}/include
    $ENV{Protobuf_ROOT_DIR}/include
    /usr/local/include
    /opt/protobuf/include
)

find_library(Protobuf_LIBRARY
  NAMES protobuf
  HINTS
    ${Protobuf_ROOT_DIR}/lib
    $ENV{Protobuf_ROOT_DIR}/lib
    /usr/local/lib
    /opt/protobuf/lib
)

# Combine Results
set(TensorFlow_INCLUDE_DIRS
  ${TensorFlow_INCLUDE_DIR}
  ${Protobuf_INCLUDE_DIR}  # Add Protobuf headers
)

set(TensorFlow_LIBRARIES
  ${TensorFlow_CC_LIBRARY}
  ${TensorFlow_FRAMEWORK_LIBRARY}
  ${Protobuf_LIBRARY}      # Link Protobuf
)

# Validate Found Dependencies
find_package_handle_standard_args(TensorFlow
  REQUIRED_VARS
    TensorFlow_INCLUDE_DIR
    TensorFlow_CC_LIBRARY
    TensorFlow_FRAMEWORK_LIBRARY
    Protobuf_INCLUDE_DIR
    Protobuf_LIBRARY
)

# Create Imported Targets
if(TensorFlow_FOUND AND NOT TARGET TensorFlow::TensorFlow)
  # TensorFlow target
  add_library(TensorFlow::TensorFlow SHARED IMPORTED)
  set_target_properties(TensorFlow::TensorFlow PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TensorFlow_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${TensorFlow_CC_LIBRARY}"
    INTERFACE_LINK_LIBRARIES 
      "${TensorFlow_FRAMEWORK_LIBRARY};${Protobuf_LIBRARY}"
  )
endif()

# Clean up internal variables
mark_as_advanced(
  TensorFlow_INCLUDE_DIR
  TensorFlow_CC_LIBRARY
  TensorFlow_FRAMEWORK_LIBRARY
  Protobuf_INCLUDE_DIR
  Protobuf_LIBRARY
)