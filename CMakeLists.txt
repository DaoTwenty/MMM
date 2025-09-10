cmake_minimum_required(VERSION 3.10)
project(CausalLMOnnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- LibTok ----
set(LIBTOK_ROOTDIR "$ENV{HOME}/libtok" CACHE PATH "Path to LibTok root")
set(LIBTOK_EXTERNAL "${LIBTOK_ROOTDIR}/external")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external)
file(COPY ${LIBTOK_EXTERNAL} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
include(FetchContent)
include(external/Dependencies.cmake)
FetchContent_Declare(LibTok SOURCE_DIR ${LIBTOK_ROOTDIR})
FetchContent_MakeAvailable(LibTok)

# ---- ONNX Runtime ----
if (APPLE)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-osx-x86_64-1.22.0")
elseif (UNIX)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.22.1")
endif()
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")

if (NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNXRuntime not found at ${ONNXRUNTIME_ROOTDIR}. 
    Please set ONNXRUNTIME_ROOTDIR to your installation path.")
endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# ---- LibTorch ----
set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch" CACHE PATH "Path to LibTorch cmake folder")
find_package(Torch REQUIRED)

# ---- JSON Library ----
find_package(nlohmann_json 3.2.0 REQUIRED)

# ---- Sources ----
set(SOURCES
    src/test.cpp
    src/model.cpp
    src/sampler.cpp
    src/engine.cpp
)

add_executable(causallm ${SOURCES})
target_include_directories(causallm PRIVATE include)

target_link_libraries(causallm
    onnxruntime
    LibTok
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)
