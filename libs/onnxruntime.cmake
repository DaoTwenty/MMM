# -------------------------------------------------------------------
# ONNX Runtime auto-setup
# -------------------------------------------------------------------

set(ONNX_VERSION "1.22.0")

# Default path (within project)
set(DEFAULT_ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

# Allow user override
set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime installation")

# Detect platform
if (APPLE)
    set(ONNX_ARCHIVE "onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_ARCHIVE}")
    set(ONNX_LIB_NAME "libonnxruntime.dylib")
elseif (UNIX)
    set(ONNX_ARCHIVE "onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_ARCHIVE}")
    set(ONNX_LIB_NAME "libonnxruntime.so")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime auto-download")
endif()

# If the user provided an existing path, copy it
if(NOT "${ONNXRUNTIME_ROOTDIR}" STREQUAL "${DEFAULT_ONNXRUNTIME_ROOT}" AND EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Using user-specified ONNX Runtime from ${ONNXRUNTIME_ROOTDIR}")
    file(COPY "${ONNXRUNTIME_ROOTDIR}/" DESTINATION "${DEFAULT_ONNXRUNTIME_ROOT}/" FILES_MATCHING PATTERN "*")
    set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}")
endif()

# If not found, download
if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Downloading ONNX Runtime from ${ONNX_URL}")
    file(DOWNLOAD "${ONNX_URL}" "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}" SHOW_PROGRESS)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/${ONNX_LIB_NAME}")

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "Failed to locate ${ONNXRUNTIME_LIB}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
)

message(STATUS "ONNXRuntime root: ${ONNXRUNTIME_ROOTDIR}")
message(STATUS "ONNXRuntime includes: ${ONNXRUNTIME_ROOTDIR}/include")
message(STATUS "ONNXRuntime libs: ${ONNXRUNTIME_LIB}")
