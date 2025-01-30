cmake_minimum_required (VERSION 3.20)


project(rhoe LANGUAGES CXX VERSION 0.1 DESCRIPTION "RHOE - A DEVELOPER-FRIENDLY TENSORFLOW C++ WRAPPER")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF) #Turns off compiler non-standard features  

add_library(rhoe)
target_include_directories(rhoe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_subdirectory (src)
add_subdirectory (examples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorFlow REQUIRED COMPONENTS CXX)
target_link_libraries(rhoe PUBLIC TensorFlow::TensorFlow)

# Testing
enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)  
endif()

# Instalation
include(GNUInstallDirs)
install(TARGETS rhoe
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



