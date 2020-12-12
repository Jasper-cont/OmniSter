cmake_minimum_required(VERSION 3.1.0)
set (CMAKE_CXX_STANDARD 14)
project (omnidir_stereo)

# Find installed Open3D, this imports
#   Open3D_INCLUDE_DIRS
#   Open3D_LIBRARIES
#   Open3D_LIBRARY_DIRS
#   Open3D_C_FLAGS
#   Open3D_CXX_FLAGS
#   Open3D_EXE_LINKER_FLAGS
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
    list(APPEND Open3D_LIBRARIES dl)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set (CMAKE_CXX_FLAGS "-march=native -fopenmp ${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
	add_definitions(-DUNIX)
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
    add_definitions(${PCL_DEFINITIONS})
endif(WIN32)

if(MSVC)
    set(OpenCV_DIR "C:/Program Files/opencv_3_4/opencv/build")
endif()

find_package (OpenCV REQUIRED)
find_package(Eigen3 3.3.4 REQUIRED NO_MODULE)
find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)

set (NAME_SRC
    src/main.cpp
    src/calib_img_gen.cpp
    src/globVector.cpp
    src/detectChessboardCorners.cpp
    src/calcChessboardCorners.cpp
    src/saveCalibration.cpp
    src/loadCalibration.cpp
    src/reprojectionError.cpp
    src/pointCloud.cpp
    src/SGMStereo.cpp
    src/SPSStereo.cpp
)

set (NAME_HEADERS
    include/calib_img_gen.h
    include/globVector.h
    include/stereocam.h
    include/detectChessboardCorners.h
    include/calcChessboardCorners.h
    include/omnidir.hpp
    include/saveCalibration.h
    include/loadCalibration.h
    include/reprojectionError.h
    include/pointCloud.h
    include/sgbm_param.h
    include/defParameter.h
    include/SPSStereo.h
    include/SGMStereo.h
)

if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

	INCLUDE_DIRECTORIES (${CMAKE_CURRENT_SOURCE_DIR}/include ${OPENCV_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} /home/jasper/Documents/Yoctopuce/yoctolib_cpp/Sources /home/jasper/Documents/Yoctopuce/yoctolib_cpp/Sources/yapi)
	link_directories (${CMAKE_BINARY_DIR}/bin ${Open3D_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS} /home/jasper/Documents/Yoctopuce/yoctolib_cpp/Binaries/linux/64bits)
	set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
	add_executable (omnidir_stereo ${NAME_SRC} ${NAME_HEADERS})

	target_link_libraries (omnidir_stereo ${OpenCV_LIBS} ${Open3D_LIBRARIES} ${PCL_LIBRARIES} /home/jasper/Documents/Yoctopuce/yoctolib_cpp/Binaries/linux/64bits/libyocto-static.a -lm -lpthread -lusb-1.0 -lstdc++ /home/jasper/Documents/Yoctopuce/yoctolib_cpp/Binaries/linux/64bits/libyocto.so.1.0.1 -fopenmp Eigen3::Eigen)

else ()
    message(SEND_ERROR "Open3D not found")
endif ()

