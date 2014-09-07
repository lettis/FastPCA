cmake_minimum_required (VERSION 2.8)
project (FastCA) 

#TODO copy fastca binary to build-dir
#TODO build doc
#TODO flags for NATIVE_ARCH, USE_PHI, SSE, AVX, etc.
set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O2 -ffast-math")

find_package (Boost COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

find_package (LAPACK REQUIRED)

add_subdirectory (src) 

