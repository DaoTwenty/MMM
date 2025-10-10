FetchContent_Declare(
    symusic
    GIT_REPOSITORY https://github.com/Yikai-Liao/symusic
    GIT_TAG 7f9a65f28d6919a387b4cfdc663f43aec3aeadae
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt <SOURCE_DIR>/CMakeLists.txt
)
FetchContent_MakeAvailable(symusic)
