cmake_minimum_required(VERSION 3.10)
project(CausalLMOnnx LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- ONNX Runtime setup ----
set(ONNXRUNTIME_ROOTDIR "/Users/paultriana/onnxruntime")

include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

link_directories("${ONNXRUNTIME_ROOTDIR}/build/MacOS/RelWithDebInfo")

# ---- LibTorch ----
set(Torch_DIR "/Users/paultriana/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

# ---- Source files ----
set(SOURCES
    scripts/utils/test.cpp
    scripts/utils/model.cpp
)

set(HEADERS
    scripts/utils/model.h
)

# ---- Build executable ----
add_executable(causallm ${SOURCES} ${HEADERS})

# ---- Link ONNX Runtime ----
target_link_libraries(causallm
    onnxruntime
    "${TORCH_LIBRARIES}"
)

set_property(TARGET causallm PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -O3")