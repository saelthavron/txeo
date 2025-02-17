cmake_minimum_required (VERSION 3.25)


project(txeo LANGUAGES CXX VERSION 0.1 DESCRIPTION "TXEO - A DEVELOPER-FRIENDLY TENSORFLOW C++ WRAPPER")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) #Turns off compiler non-standard features  


#################################################### Source Directories ####################################################



include_directories(include) 
add_subdirectory(src)
add_subdirectory(examples)


#################################################### Testing ####################################################

enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)  
endif()


#################################################### Txeo Installation ####################################################

# After defining your library targets (add_library) but before install()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

write_basic_package_version_file(
  TxeoConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion  
)

configure_package_config_file(
  cmake/TxeoConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TxeoConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Txeo
)

install(
  TARGETS txeo_static txeo_shared
  EXPORT TxeoTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  
)

install(
  EXPORT TxeoTargets
  FILE TxeoTargets.cmake
  NAMESPACE Txeo::  # Add namespace here
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Txeo
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TxeoConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TxeoConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Txeo
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "*.h"  
)

#################################################### Automating Doxygen ####################################################

find_package(Doxygen QUIET COMPONENTS dot)

if(DOXYGEN_FOUND)
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

  add_custom_target(doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating Doxygen documentation..."
      VERBATIM
  )
  install(DIRECTORY ${DOXYGEN_OUT} DESTINATION share/docs)
else()
  message(WARNING "Doxygen not found. Documentation target will not be available.")
endif()


#################################################### Automating gcovr ####################################################

option(TXEO_ENABLE_COVERAGE "Enable code coverage reporting" OFF)

if(TXEO_ENABLE_COVERAGE)
  find_program(GCOVR_PATH gcovr PATHS ENV PATH)
  if(NOT GCOVR_PATH)
    message(FATAL_ERROR "gcovr not found.")
  endif()

  add_custom_target(coverage
  COMMAND ${GCOVR_PATH}
      --gcov-executable "llvm-cov gcov"
      --root ${CMAKE_SOURCE_DIR}
      --exclude-unreachable-branches
      --exclude-throw-branches
      --print-summary
      --html-details ${CMAKE_BINARY_DIR}/coverage_report.html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating coverage report..."
    VERBATIM
  )

  add_custom_target(coverage-clean
    COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -type f -delete
    COMMENT "Cleaning coverage data..."
  )

  add_dependencies(coverage txeo_tests)
endif()
