cmake_minimum_required (VERSION 3.20)


project(txeo LANGUAGES CXX VERSION 0.1 DESCRIPTION "TXEO - A DEVELOPER-FRIENDLY TENSORFLOW C++ WRAPPER")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) #Turns off compiler non-standard features  

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorFlow REQUIRED COMPONENTS CXX)

add_subdirectory(src)
add_subdirectory(examples)

# Testing
enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)  
endif()

# Txeo Instalation
include(GNUInstallDirs)
install(TARGETS txeo_static txeo_shared EXPORT txeoTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export configuration to allow `find_package(txeo REQUIRED)`
install(EXPORT txeoTargets
  FILE txeoConfig.cmake
  NAMESPACE txeo::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/txeo
)

# Provide package config files for CMake package discovery
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/txeoConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/txeoConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/txeo
)


# Automating Doxygen
find_package(Doxygen REQUIRED)

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

add_custom_target(doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating Doxygen documentation..."
    VERBATIM
)

# Install documentation (optional)
install(DIRECTORY ${DOXYGEN_OUT} DESTINATION share/docs)
