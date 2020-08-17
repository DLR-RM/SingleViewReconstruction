cmake_minimum_required(VERSION 3.13)
project(LossCalculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -lstdc++")

# HDF5
set(HDF5_DIR /home/max/3rdparty/hdf5-1.10.6/HDF_Group/HDF5/1.10.6/share/cmake/hdf5)
set (LIB_TYPE STATIC) # or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
find_package (HDF5 NAMES hdf5 COMPONENTS C ${SEARCH_TYPE})
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIR})
set (LINK_LIBS ${LINK_LIBS} ${HDF5_C_${LIB_TYPE}_LIBRARY})

# TCLAP
include_directories(/home/max/3rdparty/tclap-1.2.2/include)

add_compile_options(-pthread)
SET(CMAKE_EXE_LINKER_FLAGS "-pthread")

add_executable(LossCalculator main.cpp util/StopWatch.cc util/StopWatch.h Hdf5ReaderAndWriter.h Array3D.h Image.h)
target_compile_options(LossCalculator PRIVATE -march=native -ffast-math)

target_link_libraries (LossCalculator ${LINK_LIBS})


