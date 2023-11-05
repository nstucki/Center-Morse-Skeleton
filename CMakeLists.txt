# CMakeLists.txt

cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(Python_VIRTUALENV FIRST)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

project(DMT)

add_executable(DMT
    src/main.cpp
	src/utils.cpp
    src/data_structures.cpp
    src/enumerators.cpp
)

set_target_properties(DMT PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)