#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Usage:
#   ./build_wheels.sh --py 9 --py 11 --osx 11 --osx 13 [--arch x86_64|arm64] [--test]
# ------------------------------------------------------------

# Default values
PY_VERSIONS=()
OSX_VERSIONS=()
OSX_ARCH=x86_64
RUN_TESTS=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --py)
            shift; PY_VERSIONS+=("$1") ;;
        --osx)
            shift; OSX_VERSIONS+=("$1") ;;
        --arch)
            shift; OSX_ARCH="$1" ;;
        --test)
            RUN_TESTS=true ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--py 9 --py 10 ...] [--osx 11 --osx 12 ...] [--arch x86_64|arm64] [--test]"
            exit 1 ;;
    esac
    shift
done

if [[ ${#PY_VERSIONS[@]} -eq 0 ]]; then
    PY_VERSIONS=(8 9 10 11 12)
fi
if [[ ${#OSX_VERSIONS[@]} -eq 0 ]]; then
    OSX_VERSIONS=(11.0 12.0 13.3)
fi

# Activate environment
source .venv/bin/activate
python3 -m pip install --upgrade cibuildwheel==2.22.0 scikit-build-core pybind11 pytest

# Shared settings
export CIBW_PLATFORM=macos
export CIBW_ARCHS_MACOS="$OSX_ARCH"

# Enable cibuildwheel’s built-in testing if requested
if [[ "$RUN_TESTS" == true ]]; then
    export CIBW_TEST_REQUIRES="pytest"
    export CIBW_TEST_COMMAND="pytest {project}"
fi

# Loop over all combinations
for OSX_VER in "${OSX_VERSIONS[@]}"; do
    for PY_VER in "${PY_VERSIONS[@]}"; do
        echo "------------------------------------------------------------"
        echo "Building for macOS ${OSX_VER} with Python 3.${PY_VER}"
        echo "------------------------------------------------------------"

        export CIBW_MACOSX_DEPLOYMENT_TARGET="${OSX_VER}"
        export MACOSX_DEPLOYMENT_TARGET="${OSX_VER}"
        export CIBW_ENVIRONMENT_MACOS="MACOSX_DEPLOYMENT_TARGET=${OSX_VER}"

        export CIBW_BUILD="cp3${PY_VER}-*"

        # Skip other versions
        SKIPS=""
        for VER in 8 9 10 11 12 13; do
            if [[ "$VER" != "$PY_VER" ]]; then
                SKIPS+=" cp3${VER}-*"
            fi
        done
        export CIBW_SKIP="${SKIPS}"

        # Run build (and possibly tests)
        python3 -m cibuildwheel --output-dir "wheelhouse/macosx"
    done
done

echo "macOS wheel build complete. Wheels are in ./wheelhouse/macosx"

# Optional: post-build testing on host (if desired instead of cibuildwheel’s internal tests)
if [[ "$RUN_TESTS" == true ]]; then
    echo "------------------------------------------------------------"
    echo "Running post-build tests on host system"
    echo "------------------------------------------------------------"

    LATEST_WHEEL=$(ls -t wheelhouse/macosx/*.whl | head -n 1)
    if [[ -z "$LATEST_WHEEL" ]]; then
        echo "No wheel found to test!"
        exit 1
    fi

    echo "Installing $LATEST_WHEEL..."
    pip install --force-reinstall "$LATEST_WHEEL"

    echo "Running pytest..."
    pytest .
fi
