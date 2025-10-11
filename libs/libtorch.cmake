# -------------------------------------------------------------------
# LibTorch configuration
# -------------------------------------------------------------------

include(FetchContent)
include(ExternalProject)

# Default install path (override via CMake cache or environment)
set(DEFAULT_LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/torch")

set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")

# -------------------------------------------------------------------
# Detect platform and version
# -------------------------------------------------------------------
set(LIBTORCH_VERSION "2.4.1")  # Example version

if (APPLE)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${LIBTORCH_VERSION}.zip")
elseif(UNIX)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip")
else()
    message(FATAL_ERROR "Unsupported platform for automatic LibTorch download.")
endif()

# -------------------------------------------------------------------
# Download LibTorch if not found
# -------------------------------------------------------------------
if (NOT EXISTS "${LIBTORCH_ROOTDIR}/share/cmake/Torch/TorchConfig.cmake")
    message(STATUS "LibTorch not found at ${LIBTORCH_ROOTDIR}. Downloading from ${LIBTORCH_URL}...")

    file(MAKE_DIRECTORY ${LIBTORCH_ROOTDIR})
    set(LIBTORCH_ZIP "${CMAKE_BINARY_DIR}/libtorch.zip")

    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    message(STATUS "LibTorch downloaded and extracted to ${LIBTORCH_ROOTDIR}")
endif()

# -------------------------------------------------------------------
# Locate Torch
# -------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}")
find_package(Torch REQUIRED)

# Expose variables so main CMakeLists can use them
set(LIBTORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" PARENT_SCOPE)
set(LIBTORCH_LIBRARIES "${TORCH_LIBRARIES}" PARENT_SCOPE)
set(LIBTORCH_FOUND TRUE PARENT_SCOPE)

message(STATUS "LibTorch root: ${LIBTORCH_ROOTDIR}")
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")
