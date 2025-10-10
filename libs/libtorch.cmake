# -------------------------------------------------------------------
# LibTorch configuration
# -------------------------------------------------------------------

# Default install path (override via CMake cache or environment)
set(DEFAULT_LIBTORCH_ROOT "$ENV{HOME}/libs/libtorch")

set(LIBTORCH_ROOTDIR ${DEFAULT_LIBTORCH_ROOT} CACHE PATH "Path to LibTorch installation")

# Append to existing prefix path so CMake can find TorchConfig.cmake
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}")

# Locate Torch
find_package(Torch REQUIRED)

# Expose variables so main CMakeLists can use them
set(LIBTORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}" PARENT_SCOPE)
set(LIBTORCH_LIBRARIES "${TORCH_LIBRARIES}" PARENT_SCOPE)
set(LIBTORCH_FOUND TRUE PARENT_SCOPE)

message(STATUS "LibTorch root: ${LIBTORCH_ROOTDIR}")
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")