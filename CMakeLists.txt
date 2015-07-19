cmake_minimum_required (VERSION 2.8)
project (FastPCA) 

set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O2")

find_package (Boost COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

find_package (LAPACK REQUIRED)

add_subdirectory (src) 

