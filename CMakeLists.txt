cmake_minimum_required(VERSION 3.28)
project(test_cu CUDA)

set(CMAKE_CUDA_STANDARD 17)

#find_package(CURL REQUIRED)

add_executable(test_cu
#        rgb.cu
#        main.cu
#        device_properties.cu
#        matmul.cu
#        by_myself.cu
#        blurring.cu
        square.cu
)

set_target_properties(test_cu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test_cu PRIVATE CURL::libcurl)

#target_include_directories(test_cu PRIVATE ${CURL_INCLUDE_DIRS})
#target_link_libraries(test_cu PRIVATE ${CURL_LIBRARIES})