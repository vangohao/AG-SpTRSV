cmake_minimum_required(VERSION 3.22)

if(EXISTS "/workspace")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8/")
endif()

set(CMAKE_CUDA_FLAGS "-O3 -g -w -m64 -Xptxas -dlcm=cg")

project(SpTRSV LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_definitions(-DUNI_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SRC_FILE ${SRC_DIR}/common.cu
${SRC_DIR}/GPU_setup.cpp
${SRC_DIR}/schedule.cu
${SRC_DIR}/scheduling_new.cu
${SRC_DIR}/preprocessing.cu
${SRC_DIR}/partition_row.cu
${SRC_DIR}/partition_level.cu
${SRC_DIR}/preprocessing_new.cu
${SRC_DIR}/finalize.cu
${SRC_DIR}/SpTRSV_executor.cu
${SRC_DIR}/SpTRSV_executor_variant.cu
${SRC_DIR}/specialization.cpp
${SRC_DIR}/format_def.cpp
${SRC_DIR}/transformation.cu
${SRC_DIR}/elimination.cpp
${SRC_DIR}/search.cu
)

set(UTILS_DIR ${PROJECT_SOURCE_DIR}/utils)
set(UTILS_FILE ${UTILS_DIR}/utils.cpp)

add_library(SpTRSV ${SRC_FILE})
add_library(utils ${UTILS_FILE})
