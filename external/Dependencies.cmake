cmake_minimum_required(VERSION 3.24)
include(FetchContent)

# GOOGLE BENCHMARK #
set(BENCHMARK_ENABLE_TESTING off)
set(BENCHMARK_ENABLE_GTEST_TESTS off)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(benchmark)

# GOOGLE TEST #
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

# TOKENIZERS #
include(external/tokenizers_cpp/add_tokenizers_cpp.cmake)

# SYMUSIC #
include(external/symusic/add_symusic.cmake)

# JSON #
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
