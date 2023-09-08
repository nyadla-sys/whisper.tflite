# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

project("tflitecpp")

# Specify where to find the header files for TF Lite C++
set( INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/tensorflow_src
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})

add_library( tflite SHARED IMPORTED )
set_target_properties( tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/${ANDROID_ABI}/libtensorflowlite.so )

# Build the main target `native-lib` that will use TF Lite
add_library( native-lib SHARED native-lib.cpp )

find_library( log-lib log ) # Library required by NDK.
find_library(android-lib android) # for AssetManager functionality

# Link the main target with two required libs: `log` and `libtensorflowlite.so`
target_link_libraries( native-lib ${log-lib} ${android-lib} tflite)