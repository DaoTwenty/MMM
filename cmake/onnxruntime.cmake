# -------------------------------------------------------------------
# ONNX Runtime auto-setup
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# ONNX Runtime version selection
# -------------------------------------------------------------------
if(NOT DEFINED ONNX_VERSION)
    if(APPLE)
        # Prefer explicit MACOSX_DEPLOYMENT_TARGET if defined
        if(DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
            set(_MACOS_VER_STR $ENV{MACOSX_DEPLOYMENT_TARGET})
        else()
            set(_MACOS_VER_STR ${CMAKE_SYSTEM_VERSION})
        endif()

        string(REPLACE "." ";" VERSION_LIST ${_MACOS_VER_STR})
        list(GET VERSION_LIST 0 MACOS_MAJOR)
        list(LENGTH VERSION_LIST LIST_LEN)
        if(LIST_LEN GREATER 1)
            list(GET VERSION_LIST 1 MACOS_MINOR)
        else()
            set(MACOS_MINOR 0)
        endif()

        math(EXPR MACOS_COMBINED "${MACOS_MAJOR} * 100 + ${MACOS_MINOR}")

        message(STATUS "Detected (target) macOS version: ${_MACOS_VER_STR}")

        if(MACOS_COMBINED LESS 1100)
            message(FATAL_ERROR "Unsupported macOS version (< 11.0). Minimum supported macOS is 11 (Big Sur).")
        elseif(MACOS_COMBINED LESS 1200)
            set(ONNX_VERSION "1.18.1")  # macOS < 12.0
        elseif(MACOS_COMBINED LESS 1330)
            set(ONNX_VERSION "1.19.0")  # macOS < 13.3
        else()
            set(ONNX_VERSION "1.22.0")  # macOS 13.3+
        endif()
    elseif(UNIX AND NOT APPLE)
        set(ONNX_VERSION "1.22.0") # Linux default
    elseif(WIN32)
        set(ONNX_VERSION "1.22.0") # Windows default
    endif()
endif()

message(STATUS "Selected ONNX Runtime version: ${ONNX_VERSION}")

# -------------------------------------------------------------------
# Base paths and URLs
# -------------------------------------------------------------------
set(ONNX_BASE_URL "https://github.com/microsoft/onnxruntime/releases/download")
set(ONNXRUNTIME_DIR_NAME "onnxruntime-${ONNX_VERSION}")
set(DEFAULT_ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/${ONNXRUNTIME_DIR_NAME}")
# cache variable allows user override (they may point to a versioned folder or another path)
set(ONNXRUNTIME_ROOTDIR ${DEFAULT_ONNXRUNTIME_ROOT} CACHE PATH "Path to ONNX Runtime installation")

# Determine platform & archive name
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
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(ONNX_ARCHIVE "onnxruntime-win-x86-${ONNX_VERSION}.zip")
    endif()
    set(ONNX_LIB_NAME "onnxruntime.dll")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime auto-download")
endif()

set(ONNX_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/${ONNX_ARCHIVE}")
message(STATUS "ONNX Runtime archive: ${ONNX_ARCHIVE}")
message(STATUS "ONNX download URL: ${ONNX_URL}")

# -------------------------------------------------------------------
# Helper: canonicalize a found extracted folder to DEFAULT_ONNXRUNTIME_ROOT
# -------------------------------------------------------------------
function(_canonicalize_extracted_dir extracted_dir dest_dir)
    # extracted_dir: full path to the extracted folder (e.g. libraries/onnxruntime-osx-x86_64-1.22.0)
    # dest_dir: desired canonical path (e.g. libraries/onnxruntime-1.22.0)
    if(NOT EXISTS "${extracted_dir}")
        message(FATAL_ERROR "Internal error: extracted dir does not exist: ${extracted_dir}")
    endif()

    if("${extracted_dir}" STREQUAL "${dest_dir}")
        # already canonical
        set(_canon_result "${dest_dir}" PARENT_SCOPE)
        return()
    endif()

    # Remove any stale destination first
    if(EXISTS "${dest_dir}")
        message(STATUS "Removing existing destination ${dest_dir} before rename")
        file(REMOVE_RECURSE "${dest_dir}")
    endif()

    # Attempt rename; file(RENAME) works cross-platform for simple cases
    file(RENAME "${extracted_dir}" "${dest_dir}")
    if(NOT EXISTS "${dest_dir}")
        message(FATAL_ERROR "Failed to rename ${extracted_dir} -> ${dest_dir}")
    endif()
    set(_canon_result "${dest_dir}" PARENT_SCOPE)
endfunction()

# -------------------------------------------------------------------
# Resolve ONNX runtime location:
# - Accept user override if it points to an include file
# - Otherwise, look for a versioned folder in libraries/
# - If not found, download archive, extract, then canonicalize the extracted folder
# -------------------------------------------------------------------

# If the user explicitly provided a path that already contains the include, use it.
if(EXISTS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "Using user-specified ONNX Runtime from ${ONNXRUNTIME_ROOTDIR}")
else()
    # Look for any existing extracted folder that contains the requested version
    file(GLOB ALL_ONNX_CANDIDATES RELATIVE "${CMAKE_SOURCE_DIR}/libraries" "${CMAKE_SOURCE_DIR}/libraries/onnxruntime*")
    set(FOUND_CANDIDATE "")
    foreach(CAND IN LISTS ALL_ONNX_CANDIDATES)
        # CAND is relative path under libraries/
        string(FIND "${CAND}" "${ONNX_VERSION}" POS)
        if(NOT POS EQUAL -1)
            # candidate contains the version string
            set(FOUND_CANDIDATE "${CMAKE_SOURCE_DIR}/libraries/${CAND}")
            break()
        endif()
    endforeach()

    if(FOUND_CANDIDATE)
        message(STATUS "Found existing extracted ONNX runtime candidate: ${FOUND_CANDIDATE}")
        # canonicalize the found dir to DEFAULT_ONNXRUNTIME_ROOT (libraries/onnxruntime-<version>)
        _canonicalize_extracted_dir("${FOUND_CANDIDATE}" "${DEFAULT_ONNXRUNTIME_ROOT}")
        set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}")
    else()
        # Not found locally -> download and extract
        message(STATUS "ONNX Runtime ${ONNX_VERSION} not found locally, will download from ${ONNX_URL}...")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/libraries")

        file(DOWNLOAD "${ONNX_URL}" "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}" SHOW_PROGRESS STATUS DL_STATUS)
        list(GET DL_STATUS 0 DL_CODE)
        if(NOT DL_CODE EQUAL 0)
            message(FATAL_ERROR "Failed to download ONNX Runtime: ${DL_STATUS}")
        endif()

        # Extract
        if(WIN32 AND ONNX_ARCHIVE MATCHES "\\.zip$")
            execute_process(
                COMMAND powershell -Command "Expand-Archive -Path '${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}' -DestinationPath '${CMAKE_SOURCE_DIR}/libraries'"
                RESULT_VARIABLE EXTRACT_RESULT
            )
        else()
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/${ONNX_ARCHIVE}"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/libraries"
                RESULT_VARIABLE EXTRACT_RESULT
            )
        endif()

        if(NOT EXTRACT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to extract ONNX Runtime archive.")
        endif()

        # Find the actual extracted directory (it may have arch prefixed name)
        file(GLOB POST_EXTRACT_DIRS RELATIVE "${CMAKE_SOURCE_DIR}/libraries" "${CMAKE_SOURCE_DIR}/libraries/onnxruntime*")
        set(EXTRACTED_FOUND "")
        foreach(DIRNAME IN LISTS POST_EXTRACT_DIRS)
            if(DIRNAME MATCHES ".*${ONNX_VERSION}.*")
                set(EXTRACTED_FOUND "${CMAKE_SOURCE_DIR}/libraries/${DIRNAME}")
                break()
            endif()
        endforeach()

        if(NOT EXTRACTED_FOUND)
            # As a fallback, see if a directory exactly named onnxruntime-<version> exists
            if(EXISTS "${DEFAULT_ONNXRUNTIME_ROOT}")
                set(EXTRACTED_FOUND "${DEFAULT_ONNXRUNTIME_ROOT}")
            else()
                message(FATAL_ERROR "Could not find extracted ONNX Runtime directory for version ${ONNX_VERSION}. Looking for patterns under ${CMAKE_SOURCE_DIR}/libraries")
            endif()
        endif()

        message(STATUS "Canonicalizing extracted folder: ${EXTRACTED_FOUND} -> ${DEFAULT_ONNXRUNTIME_ROOT}")
        _canonicalize_extracted_dir("${EXTRACTED_FOUND}" "${DEFAULT_ONNXRUNTIME_ROOT}")
        set(ONNXRUNTIME_ROOTDIR "${DEFAULT_ONNXRUNTIME_ROOT}")

        message(STATUS "ONNX Runtime ${ONNX_VERSION} ready at ${ONNXRUNTIME_ROOTDIR}")
    endif()
endif()

# -------------------------------------------------------------------
# Library path setup (correct Windows linkage)
# -------------------------------------------------------------------
if(WIN32)
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll")
    set(ONNXRUNTIME_IMPLIB "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
    if(NOT EXISTS "${ONNXRUNTIME_IMPLIB}")
        message(FATAL_ERROR "Missing ONNX Runtime import library: ${ONNXRUNTIME_IMPLIB}")
    endif()
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
        IMPORTED_IMPLIB "${ONNXRUNTIME_IMPLIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
    )
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOTDIR}/lib/${ONNX_LIB_NAME}")
    if(NOT EXISTS "${ONNXRUNTIME_LIB}")
        message(FATAL_ERROR "Missing ONNX Runtime library: ${ONNXRUNTIME_LIB}")
    endif()
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOTDIR}/include"
    )
endif()

# Expose variables to parent
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOTDIR}/include" PARENT_SCOPE)
if(WIN32)
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_IMPLIB}" PARENT_SCOPE)
else()
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_IMPLIB}" PARENT_SCOPE)
endif()

message(STATUS "ONNXRuntime root: ${ONNXRUNTIME_ROOTDIR}")
message(STATUS "ONNXRuntime includes: ${ONNXRUNTIME_ROOTDIR}/include")
if(WIN32)
    message(STATUS "ONNXRuntime libs: ${ONNXRUNTIME_IMPLIB}")
else()
    message(STATUS "ONNXRuntime libs: ${ONNXRUNTIME_LIB}")
endif()
