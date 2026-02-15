cmake_minimum_required(VERSION 3.16)
project(neural_net_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nn_lib
    src/matrix.cpp
    src/optimizer.cpp
    src/losses.cpp
    src/activations.cpp
    src/dense.cpp
    src/neural_network.cpp
)

target_include_directories(nn_lib PUBLIC include)

add_executable(neural_net_demo main.cpp)
target_link_libraries(neural_net_demo PRIVATE nn_lib)
