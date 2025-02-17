# cmake/FindTensorFlow.cmake
include(FindPackageHandleStandardArgs)

# Find TensorFlow
find_path(TensorFlow_INCLUDE_DIR
  NAMES tensorflow/core/public/session.h
  HINTS
    ${TensorFlow_ROOT_DIR}/include  
    ${TensorFlow_HOME}/include      
    $ENV{TensorFlow_ROOT_DIR}/include
    $ENV{TensorFlow_HOME}/include
)

message(STATUS "Found TensorFlow Includes at: ${TensorFlow_INCLUDE_DIR}")

find_library(TensorFlow_CC_LIBRARY
  NAMES tensorflow_cc
  HINTS
    ${TensorFlow_ROOT_DIR}/lib  
    ${TensorFlow_HOME}/lib      
    $ENV{TensorFlow_ROOT_DIR}/lib
    $ENV{TensorFlow_HOME}/lib
)

message(STATUS "Found TensorFlow C Library at: ${TensorFlow_CC_LIBRARY}")


find_library(TensorFlow_FRAMEWORK_LIBRARY
  NAMES tensorflow_framework
  HINTS
    ${TensorFlow_ROOT_DIR}/lib  
    ${TensorFlow_HOME}/lib      
    $ENV{TensorFlow_ROOT_DIR}/lib
    $ENV{TensorFlow_HOME}/lib
)

message(STATUS "Found TensorFlow Framework Library at: ${TensorFlow_FRAMEWORK_LIBRARY}")


# Find Protobuf 
find_path(Protobuf_INCLUDE_DIR
  NAMES google/protobuf/message.h
  HINTS
    ${Protobuf_ROOT_DIR}/include    
    ${Protobuf_HOME}/include
    $ENV{Protobuf_ROOT_DIR}/include
    $ENV{Protobuf_HOME}/include
)

message(STATUS "Found Protobuf Includes at: ${Protobuf_INCLUDE_DIR}")


find_library(Protobuf_LIBRARY
  NAMES protobuf
  HINTS
    ${Protobuf_ROOT_DIR}/lib    
    ${Protobuf_HOME}/lib
    $ENV{Protobuf_ROOT_DIR}/lib
    $ENV{Protobuf_HOME}/lib
)

message(STATUS "Found Protobuf Library at: ${Protobuf_LIBRARY}")

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