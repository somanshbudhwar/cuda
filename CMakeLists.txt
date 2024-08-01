cmake_minimum_required(VERSION 3.28)
project(test_cu CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(test_cu
#        rgb.cu
#        main.cu
        device_properties.cu
#        matmul.cu
)

set_target_properties(test_cu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
