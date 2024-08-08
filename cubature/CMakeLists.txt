cmake_minimum_required(VERSION 3.6)
project(cubature C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(cubature cubature.h hcubature.c
        numerics_cubature_defaults.cpp
        numerics_cubature_function_spherical_integral.hpp
        numerics_cubature_function_compactify.hpp
        numerics_cubature_function_nested_integral.hpp)
set_target_properties(cubature PROPERTIES SUFFIX -${CMAKE_BUILD_TYPE}.a)