cmake_minimum_required(VERSION 3.5)
project(chat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register -O3")

# 设置openmp
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP Not Found!")
endif()

# 这是ncnn
find_package(ncnn REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(chat chat.cpp)
target_link_libraries(chat ncnn)