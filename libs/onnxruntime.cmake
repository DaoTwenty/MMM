# Default path (override via environment or CMake)
if (APPLE)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-osx-x86_64-1.22.0")
elseif (UNIX)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.22.1")
endif()
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")

if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNXRuntime not found at ${ONNXRUNTIME_ROOTDIR}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so"  # use .dylib on macOS
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
)
