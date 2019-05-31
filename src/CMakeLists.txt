cmake_minimum_required(VERSION 3.1)
project(pdist_example)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
# find_package(xtesnor-blas REQUIRED)
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

add_definitions(-DHAVE_CBLAS=1)

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else ()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

target_link_libraries(pdist_example xtensor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})