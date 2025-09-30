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
    onnxruntime::onnxruntime
    LibTok
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)