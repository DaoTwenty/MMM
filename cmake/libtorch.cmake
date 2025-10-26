# -------------------------------------------------------------------
# LibTorch auto-setup (mirroring ONNX Runtime style)
# -------------------------------------------------------------------

include(FetchContent)
include(ExternalProject)

# -------------------------------------------------------------------
# LibTorch version selection
# -------------------------------------------------------------------
if(NOT DEFINED LIBTORCH_VERSION)
    set(LIBTORCH_VERSION "2.2.2")  # Default fallback
endif()

message(STATUS "Selected LibTorch version: ${LIBTORCH_VERSION}")

# -------------------------------------------------------------------
# Base paths and URLs
# -------------------------------------------------------------------
set(LIBTORCH_BASE_URL "https://download.pytorch.org/libtorch")
set(LIBTORCH_DIR_NAME "libtorch-${LIBTORCH_VERSION}")
set(DEFAULT_LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/${LIBTORCH_DIR_NAME}")
set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")

# -------------------------------------------------------------------
# Determine platform & archive name
# -------------------------------------------------------------------
if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64;x86_64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64;arm64")
        set(LIBTORCH_ARCHIVE "libtorch-macos-universal2-${LIBTORCH_VERSION}.zip")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(LIBTORCH_ARCHIVE "libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
    else()
        set(LIBTORCH_ARCHIVE "libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip")
    endif()
    set(LIBTORCH_URL "${LIBTORCH_BASE_URL}/cpu/${LIBTORCH_ARCHIVE}")
elseif(UNIX AND NOT APPLE)
    set(LIBTORCH_ARCHIVE "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip")
    set(LIBTORCH_URL "${LIBTORCH_BASE_URL}/cpu/${LIBTORCH_ARCHIVE}")
elseif(WIN32)
    if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
        set(LIBTORCH_ARCHIVE "libtorch-win-arm64-${LIBTORCH_VERSION}.zip")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LIBTORCH_ARCHIVE "libtorch-win-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip")
    else()
        set(LIBTORCH_ARCHIVE "libtorch-win-x86-${LIBTORCH_VERSION}.zip")
    endif()
    set(LIBTORCH_URL "${LIBTORCH_BASE_URL}/cpu/${LIBTORCH_ARCHIVE}")
else()
    message(FATAL_ERROR "Unsupported platform for LibTorch auto-download")
endif()

message(STATUS "LibTorch archive: ${LIBTORCH_ARCHIVE}")
message(STATUS "LibTorch download URL: ${LIBTORCH_URL}")

# -------------------------------------------------------------------
# Helper: canonicalize extracted folder name
# -------------------------------------------------------------------
function(_canonicalize_extracted_dir extracted_dir dest_dir)
    if(NOT EXISTS "${extracted_dir}")
        message(FATAL_ERROR "Internal error: extracted dir does not exist: ${extracted_dir}")
    endif()

    if(EXISTS "${dest_dir}")
        message(STATUS "Removing existing destination ${dest_dir} before rename")
        file(REMOVE_RECURSE "${dest_dir}")
    endif()

    file(RENAME "${extracted_dir}" "${dest_dir}")
    if(NOT EXISTS "${dest_dir}")
        message(FATAL_ERROR "Failed to rename ${extracted_dir} -> ${dest_dir}")
    endif()
    set(_canon_result "${dest_dir}" PARENT_SCOPE)
endfunction()

# -------------------------------------------------------------------
# Resolve LibTorch location
# -------------------------------------------------------------------
set(LIBTORCH_CONFIG_FILE "${LIBTORCH_ROOTDIR}/share/cmake/Torch/TorchConfig.cmake")

if(EXISTS "${LIBTORCH_CONFIG_FILE}")
    message(STATUS "Using existing LibTorch installation at ${LIBTORCH_ROOTDIR}")
else()
    # Look for local candidates
    file(GLOB ALL_LIBTORCH_CANDIDATES RELATIVE "${CMAKE_SOURCE_DIR}/libraries" "${CMAKE_SOURCE_DIR}/libraries/libtorch*")
    set(FOUND_CANDIDATE "")
    foreach(CAND IN LISTS ALL_LIBTORCH_CANDIDATES)
        if(CAND MATCHES ".*${LIBTORCH_VERSION}.*")
            set(FOUND_CANDIDATE "${CMAKE_SOURCE_DIR}/libraries/${CAND}")
            break()
        endif()
    endforeach()

    if(FOUND_CANDIDATE)
        message(STATUS "Found existing LibTorch candidate: ${FOUND_CANDIDATE}")
        _canonicalize_extracted_dir("${FOUND_CANDIDATE}" "${DEFAULT_LIBTORCH_ROOT}")
        set(LIBTORCH_ROOTDIR "${DEFAULT_LIBTORCH_ROOT}")
    else()
        # Download
        message(STATUS "LibTorch ${LIBTORCH_VERSION} not found locally. Downloading...")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/libraries")
        set(LIBTORCH_ARCHIVE_PATH "${CMAKE_BINARY_DIR}/${LIBTORCH_ARCHIVE}")
        file(DOWNLOAD "${LIBTORCH_URL}" "${LIBTORCH_ARCHIVE_PATH}" SHOW_PROGRESS STATUS DL_STATUS)
        list(GET DL_STATUS 0 DL_CODE)
        if(NOT DL_CODE EQUAL 0)
            message(FATAL_ERROR "Failed to download LibTorch: ${DL_STATUS}")
        endif()

        # Extract
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/libtorch_tmp")
        if("${LIBTORCH_ARCHIVE}" MATCHES "\\.zip$")
            execute_process(
                COMMAND unzip -q "${LIBTORCH_ARCHIVE_PATH}" -d "${CMAKE_BINARY_DIR}/libtorch_tmp"
                RESULT_VARIABLE EXTRACT_RESULT
            )
        else()
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${LIBTORCH_ARCHIVE_PATH}" -C "${CMAKE_BINARY_DIR}/libtorch_tmp"
                RESULT_VARIABLE EXTRACT_RESULT
            )
        endif()
        if(NOT EXTRACT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to extract LibTorch archive")
        endif()

        # Move extracted folder to versioned directory
        file(GLOB TMP_CONTENT "${CMAKE_BINARY_DIR}/libtorch_tmp/*")
        set(EXTRACTED_DIR "${TMP_CONTENT}")
        if(EXISTS "${EXTRACTED_DIR}")
            message(STATUS "Canonicalizing extracted folder: ${EXTRACTED_DIR} -> ${DEFAULT_LIBTORCH_ROOT}")
            _canonicalize_extracted_dir("${EXTRACTED_DIR}" "${DEFAULT_LIBTORCH_ROOT}")
        else()
            message(FATAL_ERROR "Could not locate extracted LibTorch folder")
        endif()
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/libtorch_tmp")
        set(LIBTORCH_ROOTDIR "${DEFAULT_LIBTORCH_ROOT}")
        message(STATUS "LibTorch ${LIBTORCH_VERSION} ready at ${LIBTORCH_ROOTDIR}")
    endif()
endif()

# -------------------------------------------------------------------
# Locate Torch package
# -------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}")
find_package(Torch REQUIRED)

# -------------------------------------------------------------------
# Expose to parent
# -------------------------------------------------------------------
set(LIBTORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" PARENT_SCOPE)
set(LIBTORCH_LIBRARIES "${TORCH_LIBRARIES}" PARENT_SCOPE)

message(STATUS "LibTorch root: ${LIBTORCH_ROOTDIR}")
message(STATUS "LibTorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch libs: ${TORCH_LIBRARIES}")
