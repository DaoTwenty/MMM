cmake_minimum_required(VERSION 3.10)
project(MMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---- LibTok ----
set(LIBTOK_PATH "/Users/paultriana/libtok")

include(external/Dependencies.cmake)
FetchContent_Declare(
    LibTok
    SOURCE_DIR ${LIBTOK_PATH}
)
FetchContent_MakeAvailable(LibTok)

# ---- ONNX Runtime ----
if (APPLE)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-osx-x86_64-1.22.0")
elseif (UNIX)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.22.1")
endif()
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")
set(ONNXRUNTIME_ROOTDIR_PIP "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.11/site-packages/onnxruntime/capi")

if (NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNXRuntime not found at ${ONNXRUNTIME_ROOTDIR}. 
    Please set ONNXRUNTIME_ROOTDIR to your installation path.")
endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR_PIP}")

# Find the actual .dylib in the pip folder
file(GLOB ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOTDIR_PIP}/libonnxruntime*.dylib")

if(NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "Could not find libonnxruntime*.dylib in ${ONNXRUNTIME_ROOTDIR_PIP}")
endif()

# ---- LibTorch ----
set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch" CACHE PATH "Path to LibTorch cmake folder")
find_package(Torch REQUIRED)

# ---- JSON Library ----
find_package(nlohmann_json 3.2.0 REQUIRED)

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
