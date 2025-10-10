cmake_minimum_required(VERSION 3.10)
project(MMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---- Build options ----
option(BUILD_PYBIND "Build pybind11 Python module" OFF)
option(BUILD_BENCHMARK "Build benchmarking executable" ON)

# Ensure that at least one target is built
if(NOT BUILD_PYBIND AND NOT BUILD_BENCHMARK)
    message(FATAL_ERROR "At least one of BUILD_PYBIND or BUILD_BENCHMARK must be ON.")
endif()

# ---- LibTok ----
set(LIBTOK_PATH "/Users/paultriana/libtok")
include(external/Dependencies.cmake)
FetchContent_Declare(
    LibTok
    SOURCE_DIR ${LIBTOK_PATH}
)
FetchContent_MakeAvailable(LibTok)

# ---- ONNX Runtime ----
include(libs/onnxruntime.cmake)

# ---- LibTorch ----
set(DEFAULT_LIBTORCH_ROOT "$ENV{HOME}/libs/libtorch")
set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")
set(CMAKE_PREFIX_PATH ${LIBTORCH_ROOTDIR})
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

if (BUILD_BENCHMARK)
    # ---- Sources ----
    set(SOURCES
        src/benchmark.cpp
        src/model.cpp
        src/sampler.cpp
        src/engine.cpp
        src/inference/inference.cpp
        src/inference/bar_infill.cpp
        src/inference/track_sample.cpp
    )

    # ---- Benchmark executable ----
    add_executable(mmm_benchmark ${SOURCES})
    target_include_directories(mmm_benchmark PRIVATE include)
    target_link_libraries(mmm_benchmark
        "${ONNXRUNTIME_LIBRARIES}"
        LibTok
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
    )
    set_target_properties(mmm_benchmark PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOTDIR_PIP}"
    )
endif()

# ---- Pybind11 Python module ----
if (BUILD_PYBIND)
    # Add pybind11
    add_subdirectory(external/pybind11)

    # All sources for pybind module (include inference bindings)
    set(PYBIND_SOURCES
        pylib/bindings.cpp
        src/model.cpp
        src/engine.cpp
        src/sampler.cpp
        src/inference/inference.cpp
        src/inference/bar_infill.cpp
        src/inference/track_sample.cpp
    )

    pybind11_add_module(mmm ${PYBIND_SOURCES})
    target_include_directories(mmm PRIVATE include)
    target_link_libraries(mmm PRIVATE
        LibTok
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
        "${ONNXRUNTIME_LIBRARIES}"
    )
    set_target_properties(mmm PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOTDIR_PIP}"
    )
endif()
