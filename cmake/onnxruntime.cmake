# -------------------------------------------------------------------
# ONNX Runtime auto-setup
# -------------------------------------------------------------------
set(ONNX_VERSION "1.22.0")

# Default path within project
set(DEFAULT_ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/onnxruntime")
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")

# Determine platform & binary archive
if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64;x86_64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64;arm64")
        set(ONNX_ARCHIVE "onnxruntime-osx-universal2-${ONNX_VERSION}.tgz")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(ONNX_ARCHIVE "onnxruntime-osx-arm64-${ONNX_VERSION}.tgz")
    else()
        set(ONNX_ARCHIVE "onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz")
    endif()
    set(ONNX_LIB_NAME "libonnxruntime.dylib")
elseif(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ONNX_ARCHIVE "onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
    else()
        set(ONNX_ARCHIVE "onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
    endif()
    set(ONNX_LIB_NAME "libonnxruntime.so")
elseif(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ONNX_ARCHIVE "onnxruntime-win-x64-${ONNX_VERSION}.zip")
        set(ONNX_LIB_NAME "onnxruntime.dll")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(ONNX_ARCHIVE "onnxruntime-win-x86-${ONNX_VERSION}.zip")
        set(ONNX_LIB_NAME "onnxruntime.dll")
    else()
        message(FATAL_ERROR "Unsupported Windows architecture")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime auto-download")
endif()

# Check user-provided path
if(NOT "${ONNXRUNTIME_ROOTDIR}" STREQUAL "${DEFAULT_ONNXRUNTIME_ROOT}" 
   AND EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Using user-specified ONNX Runtime from ${ONNXRUNTIME_ROOTDIR}")
    file(COPY "${ONNXRUNTIME_ROOTDIR}/" DESTINATION "${DEFAULT_ONNXRUNTIME_ROOT}/" FILES_MATCHING PATTERN "*")
    set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}")
endif()

# Download and extract if missing
if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Downloading ONNX Runtime from ${ONNX_URL}...")
    file(DOWNLOAD "${ONNX_URL}" "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}" SHOW_PROGRESS)

    # Extract
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/onnx_tmp")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}" 
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/onnx_tmp"
    )

    # Move extracted contents to libraries/onnxruntime
    file(GLOB ONNX_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/onnx_tmp/onnxruntime-*")
    list(GET ONNX_EXTRACTED_DIR 0 ONNX_RUNTIME_DIR)
    file(MAKE_DIRECTORY "${DEFAULT_ONNXRUNTIME_ROOT}")
    file(COPY "${ONNX_RUNTIME_DIR}/" DESTINATION "${DEFAULT_ONNXRUNTIME_ROOT}")

    # Cleanup
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/onnx_tmp")
    file(REMOVE "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}")

    set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}")
endif()

# Set library path
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/${ONNX_LIB_NAME}")
if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "Failed to locate ${ONNXRUNTIME_LIB}")
endif()

# Create imported target
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
)

# Expose variables for parent CMakeLists
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOTDIR}/include" PARENT_SCOPE)
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_LIB}" PARENT_SCOPE)

message(STATUS "ONNXRuntime root: ${ONNXRUNTIME_ROOTDIR}")
message(STATUS "ONNXRuntime includes: ${ONNXRUNTIME_ROOTDIR}/include")
message(STATUS "ONNXRuntime libs: ${ONNXRUNTIME_LIB}")
