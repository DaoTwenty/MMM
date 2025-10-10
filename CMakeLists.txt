cmake_minimum_required(VERSION 3.10)
project(MMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# -------------------------------------------------------------------
# Build Options
# -------------------------------------------------------------------
option(BUILD_PYBIND     "Build pybind11 Python module"       OFF)
option(BUILD_BENCHMARK  "Build benchmarking executable"      ON)
option(USE_TORCH        "Use Torch model backend"            OFF)
option(USE_ONNX         "Use ONNX model backend"             ON)

# Sanity checks
if (NOT USE_ONNX AND NOT USE_TORCH)
    message(FATAL_ERROR "You must enable either USE_ONNX or USE_TORCH.")
elseif (USE_ONNX AND USE_TORCH)
    message(FATAL_ERROR "Only one inference backend can be enabled: ONNX or TORCH.")
endif()

if (NOT BUILD_PYBIND AND NOT BUILD_BENCHMARK)
    message(FATAL_ERROR "At least one of BUILD_PYBIND or BUILD_BENCHMARK must be ON.")
endif()

# -------------------------------------------------------------------
# External dependencies
# -------------------------------------------------------------------

# LibTok
include(libs/libtok.cmake)

# -------------------------------------------------------------------
# Backend selection
# -------------------------------------------------------------------

if (USE_TORCH)
    add_compile_definitions(USE_TORCH)
    message(STATUS "Building with Torch backend")

    # LibTorch
    set(DEFAULT_LIBTORCH_ROOT "$ENV{HOME}/libs/libtorch")
    set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOTDIR})
    find_package(Torch REQUIRED)

    message(STATUS "Torch_DIR: ${Torch_DIR}")
    message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

elseif (USE_ONNX)
    add_compile_definitions(USE_ONNX)
    message(STATUS "Building with ONNX Runtime backend")

    include(libs/onnxruntime.cmake)

    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
    set(ONNXRUNTIME_LIBRARIES onnxruntime::onnxruntime)
endif()

# -------------------------------------------------------------------
# Common sources
# -------------------------------------------------------------------

set(COMMON_SOURCES
    src/model.cpp
    src/sampler.cpp
    src/engine.cpp
    src/inference/inference.cpp
    src/inference/bar_infill.cpp
    src/inference/track_sample.cpp
)

set(COMMON_INCLUDES include)
set(COMMON_LIBS LibTok nlohmann_json::nlohmann_json)

if (USE_TORCH)
    list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")
elseif (USE_ONNX)
    list(APPEND COMMON_LIBS "${ONNXRUNTIME_LIBRARIES}")
endif()

# -------------------------------------------------------------------
# Benchmark executable
# -------------------------------------------------------------------

if (BUILD_BENCHMARK)
    add_executable(mmm_benchmark src/benchmark.cpp ${COMMON_SOURCES})
    target_include_directories(mmm_benchmark PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(mmm_benchmark PRIVATE ${COMMON_LIBS})
    set_target_properties(mmm_benchmark PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOTDIR_PIP}"
    )
endif()

# -------------------------------------------------------------------
# Pybind11 Python module
# -------------------------------------------------------------------

if (BUILD_PYBIND)
    add_subdirectory(external/pybind11)

    set(PYBIND_SOURCES
        pylib/bindings.cpp
        ${COMMON_SOURCES}
    )

    pybind11_add_module(mmm ${PYBIND_SOURCES})
    target_include_directories(mmm PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(mmm PRIVATE ${COMMON_LIBS})
    set_target_properties(mmm PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOTDIR_PIP}"
    )
endif()

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
message(STATUS "------------------------------------------------------------")
message(STATUS "Project configuration summary:")
message(STATUS "  Backend:       ${USE_TORCH} (Torch) / ${USE_ONNX} (ONNX)")
message(STATUS "  Pybind:        ${BUILD_PYBIND}")
message(STATUS "  Benchmark:     ${BUILD_BENCHMARK}")
message(STATUS "  LibTok path:   ${LIBTOK_PATH}")
message(STATUS "------------------------------------------------------------")
