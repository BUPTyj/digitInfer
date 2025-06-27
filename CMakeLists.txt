cmake_minimum_required(VERSION 3.31)

project(test LANGUAGES C CXX)         # ⭐ 启用 C

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- zlib ----------
file(GLOB ZLIB_CORE_SRC external/zlib/*.c)
add_library(zlib STATIC ${ZLIB_CORE_SRC})
target_include_directories(zlib PUBLIC external/zlib)

# ---------- cnpy ----------
add_library(cnpy STATIC cnpy/cnpy.cpp)
target_include_directories(cnpy PUBLIC cnpy)
target_link_libraries(cnpy PUBLIC zlib)

if(MSVC)
    target_compile_definitions(cnpy PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# ---------- exe ----------
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")

find_package(OpenCL REQUIRED)          # 让 CMake 帮你找头文件 + 库

add_executable(test
        nets.hpp
#        main.cpp)
        NeuralNetworkOpenCL.cpp)

target_link_libraries(test PRIVATE     # 原有 cnpy 保留
        cnpy
        OpenCL::OpenCL)                # 新增：官方导入目标
