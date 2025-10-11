# -------------------------------------------------------------------
# LibTorch configuration
# -------------------------------------------------------------------
include(FetchContent)
include(ExternalProject)

# Default install path (override via CMake cache or environment)
set(DEFAULT_LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/torch")
set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")

# -------------------------------------------------------------------
# Detect platform and version
# -------------------------------------------------------------------
set(LIBTORCH_VERSION "2.4.1")  # Change to your desired version

if (APPLE)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${LIBTORCH_VERSION}.zip")
    set(LIBTORCH_ARCHIVE_TYPE "zip")
elseif (UNIX)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip")
    set(LIBTORCH_ARCHIVE_TYPE "zip")  # Or "tgz" if using Linux tar.gz
else()
    message(FATAL_ERROR "Unsupported platform for automatic LibTorch download.")
endif()

# -------------------------------------------------------------------
# Download and extract LibTorch if not found
# -------------------------------------------------------------------
set(TORCH_CONFIG_FILE "${LIBTORCH_ROOTDIR}/share/cmake/Torch/TorchConfig.cmake")

if(NOT EXISTS "${TORCH_CONFIG_FILE}")
    message(STATUS "LibTorch not found at ${LIBTORCH_ROOTDIR}. Downloading from ${LIBTORCH_URL}...")

    file(MAKE_DIRECTORY ${LIBTORCH_ROOTDIR})
    set(LIBTORCH_ZIP "${CMAKE_BINARY_DIR}/libtorch.${LIBTORCH_ARCHIVE_TYPE}")

    # Download the archive
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)

    # Extract archive
    if("${LIBTORCH_ARCHIVE_TYPE}" STREQUAL "zip")
        execute_process(
            COMMAND unzip -q ${LIBTORCH_ZIP} -d ${CMAKE_BINARY_DIR}/libtorch_tmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    else()
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP} -C ${CMAKE_BINARY_DIR}/libtorch_tmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    # Move contents to final libraries/torch folder
    file(GLOB TMP_CONTENT "${CMAKE_BINARY_DIR}/libtorch_tmp/*")
    file(COPY ${TMP_CONTENT} DESTINATION ${LIBTORCH_ROOTDIR})

    # Cleanup
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/libtorch_tmp")
    file(REMOVE "${LIBTORCH_ZIP}")

    message(STATUS "LibTorch downloaded and installed at ${LIBTORCH_ROOTDIR}")
endif()

# -------------------------------------------------------------------
# Locate Torch
# -------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}")
find_package(Torch REQUIRED)

# -------------------------------------------------------------------
# Expose variables to parent CMakeLists
# -------------------------------------------------------------------
set(LIBTORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" PARENT_SCOPE)
set(LIBTORCH_LIBRARIES "${TORCH_LIBRARIES}" PARENT_SCOPE)

message(STATUS "LibTorch root: ${LIBTORCH_ROOTDIR}")
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
