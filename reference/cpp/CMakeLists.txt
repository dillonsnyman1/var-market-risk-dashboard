cmake_minimum_required(VERSION 3.16)
project(var_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(var var.cpp)
target_include_directories(var PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_executable(test_var test_var.cpp)
target_link_libraries(test_var PRIVATE var)
target_compile_definitions(test_var PRIVATE FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")

add_test(NAME test_var COMMAND test_var)
