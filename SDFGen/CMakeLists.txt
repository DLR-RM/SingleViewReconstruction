cmake_minimum_required(VERSION 3.10)
project(sdfgen)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -lstdc++ -lm")

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
add_executable(sdfgen main.cpp obj/ObjReader.cpp obj/ObjReader.h util/Utility.h util/StopWatch.h util/StopWatch.cc geom/Line.h geom/Line.cc geom/math/Point.h geom/math/Point_i.h geom/Polygon.cpp geom/Polygon.h container/Array3D.h container/Array3D_i.h geom/BoundingBox.cpp geom/BoundingBox.h container/Space.cpp container/Space.h test/PolygonTest.h test/PolygonTest.cc geom/math/AvgNumber.h container/Octree.cpp container/Octree.h geom/PolygonCubeIntersection.h geom/PolygonCubeIntersection.cpp util/Hdf5Writer.h)

target_compile_options(sdfgen PRIVATE -march=native -ffast-math)
target_link_libraries (sdfgen ${LINK_LIBS})

