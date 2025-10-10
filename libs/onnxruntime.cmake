# Default path (override via environment or CMake)
if (APPLE)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-osx-x86_64-1.22.0")
elseif (UNIX)
    set(DEFAULT_ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.22.1")
endif()

set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")

if (APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib")
elseif (UNIX)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNXRuntime include not found at ${ONNXRUNTIME_ROOTDIR}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "ONNXRuntime binary not found at ${ONNXRUNTIME_ROOTDIR}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
)
