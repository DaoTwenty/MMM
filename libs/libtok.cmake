# ---- LibTok ----
# Default path (override via environment or CMake cache)
set(DEFAULT_LIBTOK_ROOT "$ENV{HOME}/libtok")
set(LIBTOK_ROOTDIR ${DEFAULT_LIBTOK_ROOT} CACHE PATH "Path to LibTok root directory")

# Ensure the LibTok source exists
if(NOT EXISTS "${LIBTOK_ROOTDIR}/CMakeLists.txt")
    message(FATAL_ERROR "LibTok not found at ${LIBTOK_ROOTDIR}")
endif()

# Include LibTok dependencies if present
if(EXISTS "${LIBTOK_ROOTDIR}/external/Dependencies.cmake")
    include("${LIBTOK_ROOTDIR}/external/Dependencies.cmake")
endif()

# Make LibTok available using FetchContent
include(FetchContent)
FetchContent_Declare(
    LibTok
    SOURCE_DIR ${LIBTOK_ROOTDIR}
)
FetchContent_MakeAvailable(LibTok)

# Create an imported target for easier linking
add_library(libtok::libtok INTERFACE IMPORTED)
set_target_properties(libtok::libtok PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LIBTOK_ROOTDIR}/include"
)

