cmake_minimum_required(VERSION 3.1)
project(pdist_example)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
# if xtensor was built with xsimd support:
# find_package(xsimd REQUIRED)

set(SOURCE_FILES pdist_example.cpp distance.cpp ncfs.cpp)
add_executable(pdist_example ${SOURCE_FILES})

if(MSVC)
    target_compile_options(pdist_example PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(pdist_example PRIVATE -march=native -std=c++14)
endif()

target_link_libraries(pdist_example xtensor)