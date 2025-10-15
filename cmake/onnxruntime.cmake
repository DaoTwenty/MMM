# -------------------------------------------------------------------
# ONNX Runtime auto-setup
# -------------------------------------------------------------------
if(NOT DEFINED ONNX_VERSION)
    set(ONNX_VERSION "1.22.0")
endif()

# Default path within project
set(DEFAULT_ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/onnxruntime")
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")
set(ONNX_BASE_URL "https://github.com/microsoft/onnxruntime/releases/download")

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

set(ONNX_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/${ONNX_ARCHIVE}")
message(STATUS "ONNX Runtime version: ${ONNX_VERSION}")
message(STATUS "ONNX archive: ${ONNX_ARCHIVE}")
message(STATUS "ONNX download URL: ${ONNX_URL}")

# Download and extract if missing
if(NOT EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Downloading ONNX Runtime from ${ONNX_URL}...")

    # Ensure destination directory exists
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/libraries")

    # Download the archive
    file(DOWNLOAD "${ONNX_URL}" "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}" SHOW_PROGRESS STATUS DL_STATUS)
    list(GET DL_STATUS 0 DL_CODE)
    if(NOT DL_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download ONNX Runtime: ${DL_STATUS}")
    endif()

    # Extract into libraries/
    message(STATUS "Extracting TAR archive...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/libraries"
        RESULT_VARIABLE EXTRACT_RESULT
    )

    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNX Runtime archive.")
    endif()

    # Find extracted folder
    file(GLOB ONNX_EXTRACTED_DIRS "${CMAKE_SOURCE_DIR}/libraries/onnxruntime-*")
    list(GET ONNX_EXTRACTED_DIRS 0 ONNX_EXTRACTED_DIR)
    if(NOT EXISTS "${ONNX_EXTRACTED_DIR}")
        message(FATAL_ERROR "Could not find extracted ONNX Runtime directory.")
    endif()

    # Rename it to stable name
    file(RENAME "${ONNX_EXTRACTED_DIR}" "${ONNXRUNTIME_ROOTDIR}")

    message(STATUS "ONNX Runtime ready at ${ONNXRUNTIME_ROOTDIR}")
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
