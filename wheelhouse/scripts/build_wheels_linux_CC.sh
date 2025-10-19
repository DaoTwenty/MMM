#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Usage:
#   ./build_wheels_linux.sh --py 8 --py 9 --py 10 --arch x86_64 --arch aarch64 [--test] [--no_build]
# ------------------------------------------------------------

PY_VERSIONS=()
ARCHS=("x86_64" "aarch64")
TEST_MODE=0
BUILD=1

declare -A MANYLINUX_IMAGES
MANYLINUX_IMAGES["x86_64"]="manylinux_2_28_x86_64"
MANYLINUX_IMAGES["aarch64"]="manylinux_2_28_aarch64"

# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --py) shift; PY_VERSIONS+=("$1") ;;
        --arch) shift; ARCHS=("$1") ;;
        --test) TEST_MODE=1 ;;
        --no_build) BUILD=0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# Defaults if not provided
if [[ ${#PY_VERSIONS[@]} -eq 0 ]]; then
    PY_VERSIONS=(8 9 10 11 12)
fi

OUTPUT_DIR="$(pwd)/wheelhouse/linux"
mkdir -p "$OUTPUT_DIR"

module load apptainer
source .venv/bin/activate

LIBTOK_DIR="./libraries/libtok"
PATCH_FILE="./cmake/libtok_patch.cmake"
BRANCH="main"

# ------------------------------------------------------------
# Clone LibTok repo if missing
# ------------------------------------------------------------
mkdir -p "$(dirname "$LIBTOK_DIR")"
if [[ ! -d "$LIBTOK_DIR" ]]; then
    echo "LibTok not found at $LIBTOK_DIR. Cloning branch '$BRANCH'..."
    git clone --branch "$BRANCH" git@github.com:steinbergmedia/libtok.git "$LIBTOK_DIR"
    echo "Running patch.cmake..."
    cmake -DLIBTOK_ROOTDIR="$LIBTOK_DIR" -P "$PATCH_FILE"
else
    echo "LibTok already exists at $LIBTOK_DIR. Skipping clone."
fi

python3 -m pip install --upgrade cibuildwheel==2.22.0 scikit-build-core pybind11

# ------------------------------------------------------------
# Build or test for each arch/python
# ------------------------------------------------------------
for ARCH in "${ARCHS[@]}"; do
    IMAGES="${MANYLINUX_IMAGES[$ARCH]}"
    for IMG_BASE in $IMAGES; do
        SANDBOX="./${IMG_BASE}.sandbox"

        # Create sandbox if missing
        if [[ ! -d "$SANDBOX" ]]; then
            echo "Creating writable sandbox $SANDBOX from $IMG_BASE..."
            apptainer build --sandbox "$SANDBOX" --fix-perms "docker://quay.io/pypa/${IMG_BASE}"
        else
            echo "Using existing sandbox $SANDBOX"
        fi

        for PY_VER in "${PY_VERSIONS[@]}"; do
            echo "------------------------------------------------------------"
            if [[ $TEST_MODE -eq 1 ]]; then
                echo "TESTING for $ARCH using $IMG_BASE with Python 3.${PY_VER}"
            fi
            if [[ $BUILD -eq 1 ]]; then
                echo "BUILDING for $ARCH using $IMG_BASE with Python 3.${PY_VER}"
            fi
            if [[ $TEST_MODE -eq 0 && $BUILD -eq 0 ]]; then
                echo "No action required, exiting..."
                exit 0
            fi
            echo "------------------------------------------------------------"

            apptainer exec \
                --bind "$(pwd):/io" \
                --bind "$SCRATCH:/scratch" \
                -W "$SLURM_TMPDIR" \
                "$SANDBOX" \
                bash -c '
                set -e

                python3.'"${PY_VER}"' -m venv /tmp/.venv_'"${PY_VER}"'
                source /tmp/.venv_'"${PY_VER}"'/bin/activate
                python -m ensurepip --upgrade
                echo "Python activated..."
                python --version

                if [[ '"$BUILD"' -eq 1 ]]; then
                    echo "=== BUILD MODE ==="
                    echo "Installing Rust..."
                    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    source $HOME/.cargo/env

                    source /tmp/.venv_'"${PY_VER}"'/bin/activate

                    python -m pip install --force-reinstall cmake
                    python -m pip install --upgrade build setuptools wheel

                    cd /io
                    python -m build --wheel --outdir /io/wheelhouse/linux
                fi

                if [[ '"$TEST_MODE"' -eq 1 ]]; then
                    echo "=== TEST MODE ENABLED ==="
                    python -m pip install -U pip pytest
                    WHEEL_TAG="cp3'"${PY_VER}"'-linux_'"${ARCH}"'"
                    echo "Looking for wheel tag: $WHEEL_TAG"
                    WHEEL_PATH=$(ls -t /io/wheelhouse/linux/*${WHEEL_TAG}*.whl | head -n 1 || true)
                    if [[ -z "$WHEEL_PATH" ]]; then
                        echo "ERROR: No matching wheel found for Python 3.'"${PY_VER}"' and arch '"${ARCH}"'"
                        exit 1
                    fi
                    echo "Installing wheel: $WHEEL_PATH"
                    python -m pip install "$WHEEL_PATH"

                    export OMP_NUM_THREADS=2
                    export ORT_NUM_THREADS=2
                    export ORT_DISABLE_CPU_MEM_ARENA=1

                    echo "Running pytest in /io/"
                    cd /io
                    pytest tests/ -v -s
                fi
            '
        done
    done
done

if [[ $TEST_MODE -eq 1 ]]; then
    echo "Linux wheel tests complete."
fi
if if [[ $BUILD -eq 1 ]]; then
    echo "Linux wheel build complete. Wheels are in ./wheelhouse/linux"
fi
