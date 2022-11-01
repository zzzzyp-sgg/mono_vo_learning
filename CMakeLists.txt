cmake_minimum_required( VERSION 3.0 )
project( mono_vo )

set( CMAKE_BUILD_TYPE "Debug" )

find_package( OpenCV 3 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file( GLOB lib_vo "*.h" "*.cpp" )

add_executable( mono_vo ${lib_vo} )
target_link_libraries( mono_vo ${OpenCV_LIBS} )