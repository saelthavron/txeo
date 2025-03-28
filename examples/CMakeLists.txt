add_executable(txeo_example txeo_example.cpp)
add_executable(txeo_shapes txeo_shapes.cpp)
add_executable(txeo_tensors txeo_tensors.cpp)
add_executable(txeo_tensorio txeo_tensorio.cpp)
add_executable(txeo_olsGD txeo_olsGD.cpp)

target_precompile_headers(txeo_example PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/txeo/detail/pch.h>"
)
target_precompile_headers(txeo_shapes PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/txeo/detail/pch.h>"
)
target_precompile_headers(txeo_tensors PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/txeo/detail/pch.h>"
)
target_precompile_headers(txeo_tensorio PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/txeo/detail/pch.h>"
)
target_precompile_headers(txeo_olsGD PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/include/txeo/detail/pch.h>"
)

target_link_libraries(txeo_example txeo_shared)
target_link_libraries(txeo_shapes txeo_shared)
target_link_libraries(txeo_tensors txeo_shared)
target_link_libraries(txeo_tensorio txeo_shared)
target_link_libraries(txeo_olsGD txeo_shared)
