#!/bin/bash
set -e

# Default values
FORCE_DELETE_BUILD=0
BUILD_DIR=build

# Parse options
while [[ "$1" == -* ]]; do
    case "$1" in
        -d|--delete)
            FORCE_DELETE_BUILD=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Delete build dir if requested
if [[ $FORCE_DELETE_BUILD -eq 1 ]]; then
    echo "Deleting build directory..."
    rm -rf "$BUILD_DIR"
fi

# Ensure build directory exists
mkdir -p "$BUILD_DIR"

# Configure CMake only if needed (if CMakeCache.txt does not exist)
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Running initial CMake configuration..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_PYBIND=ON \
        -DBUILD_BENCHMARK=ON \
        -DUSE_ONNX=ON \
        -DUSE_TORCH=OFF
fi

# Build incrementally
echo "Building..."
cmake --build "$BUILD_DIR" -j 16

# Run your program with remaining arguments
# Anything after -- is passed to your binary
if [[ $# -gt 0 ]]; then
    echo "Running benchmark..."
    "$BUILD_DIR/mmm_benchmark" "$@"
fi
