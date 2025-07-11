# CMakeLists.txt

cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(Python_VIRTUALENV FIRST)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

project(DMT)

find_package(pybind11 REQUIRED)

find_package(Threads REQUIRED)
add_compile_options("-pthread")
link_libraries("-pthread")

#find_package(TBB REQUIRED)

pybind11_add_module(morse_complex
    src/dmt.cpp
    src/data_structures.cpp
    src/enumerators.cpp 
    src/utils.cpp
)

add_executable(CancelPairs
    src/cancel_pairs.cpp
    src/data_structures.cpp
    src/enumerators.cpp
    src/utils.cpp
)

add_executable(DMT
    src/main.cpp
    src/data_structures.cpp
    src/enumerators.cpp
    src/utils.cpp
)

set_target_properties(morse_complex PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

set_target_properties(CancelPairs PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

set_target_properties(DMT PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

#target_link_libraries(morse_complex PUBLIC TBB::tbb)
#target_link_libraries(DMT PUBLIC TBB::tbb)
#target_link_libraries(CancelPairs PUBLIC TBB::tbb)