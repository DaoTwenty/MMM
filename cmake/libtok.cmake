# ---- LibTok auto-setup ----
set(DEFAULT_LIBTOK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libraries/libtok")
set(LIBTOK_ROOTDIR "$ENV{LIBTOK_ROOT}" CACHE PATH "Path to LibTok root directory (user override)")

# Check if user path exists and is valid
if(NOT EXISTS "${LIBTOK_ROOTDIR}/CMakeLists.txt")
    if(DEFINED ENV{LIBTOK_ROOT} AND NOT "${LIBTOK_ROOTDIR}" STREQUAL "")
        message(WARNING "Provided LIBTOK_ROOT path is invalid or missing CMakeLists.txt: ${LIBTOK_ROOTDIR}")
    endif()
    set(LIBTOK_ROOTDIR "${DEFAULT_LIBTOK_ROOT}")
endif()

# Clone LibTok if it doesn't exist at default
if(NOT EXISTS "${LIBTOK_ROOTDIR}/CMakeLists.txt")
    message(STATUS "LibTok not found at ${LIBTOK_ROOTDIR}, cloning private repo via SSH...")
    set(LIBTOK_BRANCH "parameterised_test_suite")

    execute_process(
        COMMAND git clone --depth 1 --branch ${LIBTOK_BRANCH} git@github.com:steinbergmedia/libtok.git "${LIBTOK_ROOTDIR}"
        RESULT_VARIABLE git_result
        OUTPUT_VARIABLE git_output
        ERROR_VARIABLE git_error
    )

    if(NOT git_result EQUAL 0)
        message(FATAL_ERROR "Failed to clone LibTok branch '${LIBTOK_BRANCH}' via SSH:\n${git_error}")
    endif()

endif()

# Include dependencies if present
set(LIBTOK_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")

# Copy libtok's external folder if it doesn't exist
if(NOT EXISTS "${LIBTOK_EXTERNAL_DIR}")
    message(STATUS "Copying libtok external folder to ${LIBTOK_EXTERNAL_DIR}...")
    file(COPY "${LIBTOK_ROOTDIR}/external" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Disable installation of all dependencies before including them
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Disable msgpack installation (used by tokenizers_cpp)
set(MSGPACK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MSGPACK_INSTALL OFF CACHE BOOL "" FORCE)
set(MSGPACK_USE_BOOST OFF CACHE BOOL "" FORCE)

# Include Dependencies.cmake from the local external folder
set(LIBTOK_DEPENDENCIES "${LIBTOK_EXTERNAL_DIR}/Dependencies.cmake")
if(EXISTS "${LIBTOK_DEPENDENCIES}")
    include("${LIBTOK_DEPENDENCIES}")
else()
    message(FATAL_ERROR "LibTok Dependencies.cmake not found at ${LIBTOK_DEPENDENCIES}")
endif()

# Make LibTok available using FetchContent with EXCLUDE_FROM_ALL
include(FetchContent)
FetchContent_Declare(
    LibTok
    SOURCE_DIR "${LIBTOK_ROOTDIR}"
)

# Use manual population with EXCLUDE_FROM_ALL to prevent install rules
FetchContent_GetProperties(LibTok)
if(NOT libtok_POPULATED)
    FetchContent_Populate(LibTok)
    add_subdirectory(${libtok_SOURCE_DIR} ${libtok_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Create an imported target for easier linking
add_library(libtok::libtok INTERFACE IMPORTED)
set_target_properties(libtok::libtok PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LIBTOK_ROOTDIR}/include"
)

message(STATUS "LibTok root: ${LIBTOK_ROOTDIR}")