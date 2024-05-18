cmake_minimum_required(VERSION 3.10)
project(chat)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register -O3")

find_package(ncnn REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(chat chat.cpp)
target_link_libraries(chat ncnn)