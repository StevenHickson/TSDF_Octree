cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

IF(WIN32 OR win64)
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll)
ELSE()
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()


find_package(PCL 1.7 REQUIRED)
find_package( OpenCV REQUIRED )
 
project(TSDF_Octree)

add_subdirectory(src)

include_directories(C:/opencv/build)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(C:/blepo/external64/Microsoft/Kinect)
include_directories(${TSDF_Octree_SOURCE_DIR}/include)
include_directories(../PCL_WindowsKinectSDK/include)
include_directories(../cpu_tsdf/include)

link_directories(${PCL_LIBRARY_DIRS})
link_directories(C:/blepo/external64/Microsoft/Kinect)
link_directories(../../PCL_WindowsKinectSDK/build/)
link_directories(../../cpu_tsdf/build/)

add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE TSDF_Octree_HEADERS include/*.h)
file(GLOB_RECURSE TSDF_Octree_SOURCES src/*.cpp)
set (TSDF_Octree_INCLUDE_DIRS "")
foreach (_headerFile ${TSDF_Octree_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND TSDF_Octree_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES TSDF_Octree_INCLUDE_DIRS)

include_directories(${TSDF_Octree_INCLUDE_DIRS})

add_executable (TSDF_Octree ${TSDF_Octree_SOURCES} ${TSDF_Octree_HEADERS})
target_link_libraries (TSDF_Octree ${PCL_LIBRARIES} ${OpenCV_LIBS} Kinect10.lib debug Debug/PCL_WindowsKinectSDK optimized Release/PCL_WindowsKinectSDK debug Debug/cpu_tsdf optimized Release/cpu_tsdf)