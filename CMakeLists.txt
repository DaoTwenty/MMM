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
include(cmake/libtok.cmake)

# -------------------------------------------------------------------
# Backend selection
# -------------------------------------------------------------------
if (USE_TORCH)
    add_compile_definitions(USE_TORCH)
    message(STATUS "Building with Torch backend")

    # Include our LibTorch cmake
    include(cmake/libtorch.cmake)

elseif (USE_ONNX)
    add_compile_definitions(USE_ONNX)
    message(STATUS "Building with ONNX Runtime backend")

    # Include our ONNX Runtime cmake
    include(cmake/onnxruntime.cmake)
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
    list(APPEND COMMON_LIBS onnxruntime::onnxruntime)
endif()

# -------------------------------------------------------------------
# Benchmark executable
# -------------------------------------------------------------------
if (BUILD_BENCHMARK)
    add_executable(mmm_benchmark src/benchmark.cpp ${COMMON_SOURCES})
    target_include_directories(mmm_benchmark PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(mmm_benchmark PRIVATE ${COMMON_LIBS})
endif()

# -------------------------------------------------------------------
# Pybind11 Python module
# -------------------------------------------------------------------
if (BUILD_PYBIND)

    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(mmm pylib/bindings.cpp ${COMMON_SOURCES})
    target_include_directories(mmm PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(mmm PRIVATE ${COMMON_LIBS})

    if(USE_ONNX)
        if(APPLE)
            set(RPATH_LIB "@loader_path/onnxruntime/lib")
        elseif(UNIX)
            set(RPATH_LIB "$ORIGIN/onnxruntime/lib")
        endif()
    elseif(USE_TORCH)
        if(APPLE)
            set(RPATH_LIB "@loader_path/torch/lib")
        elseif(UNIX)
            set(RPATH_LIB "$ORIGIN/torch/lib")
        endif()
    endif()

    # Apply RPATH if set
    if(RPATH_LIB)
        set_target_properties(mmm PROPERTIES
            BUILD_RPATH "${RPATH_LIB}"
            INSTALL_RPATH "${RPATH_LIB}"
        )
    endif()

    # -------------------------------------------------------------------
    # Wheel building
    # -------------------------------------------------------------------
    # Copy module into the wheel
    install(TARGETS mmm
            DESTINATION "mmm")

    # Copy engine lib
    if(USE_ONNX)
        install(DIRECTORY "${ONNXRUNTIME_ROOTDIR}/lib"
                DESTINATION "mmm/onnxruntime")
    elseif(USE_TORCH)
        install(DIRECTORY "${LIBTORCH_ROOTDIR}/lib"
                DESTINATION "mmm/torch")
    endif()
endif()

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
message(STATUS "------------------------------------------------------------")
message(STATUS "Project configuration summary:")
message(STATUS "  Backend:       ${USE_TORCH} (Torch) / ${USE_ONNX} (ONNX)")
message(STATUS "  Pybind:        ${BUILD_PYBIND}")
message(STATUS "  Benchmark:     ${BUILD_BENCHMARK}")
message(STATUS "  LibTok path:   ${LIBTOK_ROOTDIR}")
if(USE_ONNX)
    message(STATUS "  ONNX path:     ${ONNXRUNTIME_ROOTDIR}")
elseif(USE_TORCH)
    message(STATUS "  TORCH path:    ${LIBTORCH_ROOTDIR}")
endif()
message(STATUS "------------------------------------------------------------")
