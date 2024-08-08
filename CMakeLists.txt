cmake_minimum_required(VERSION 3.16)
project(numerics_new C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -save-temps")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wno-unknown-pragmas -Wextra")

set(CMAKE_VERBOSE_MAKEFILE off)

add_subdirectory(test) # Unit tests
add_subdirectory(cubature) # Numerical integration

## Computations
#add_subdirectory(GNM_MF/stability_analysis)
#add_subdirectory(GNM_MF/phase_diagram)
#add_subdirectory(QMM_rMF_CBSAAK/phase_diagram_homo)

## Notes for manual build/compilation
# CMake manual build (and options possible to set in CLion):
# cmake -DCMAKE_BUILD_TYPE=Debug -D OMP_THREADS="1" -D USE_MPI="FALSE" -D OUTPUT_DIR=$PWD ../
# make options:
# -j 2

## Git information for doc(xygen)
set(git_dirty_msg " (with with uncommitted changes)")
#execute_process(COMMAND git describe --always --long --tags --dirty=${git_dirty_msg} OUTPUT_VARIABLE GIT_REV)
set(PROJECT_VERSION "${PROJECT_NAME} ${GIT_REV}")
message("Building ${PROJECT_VERSION}")

## Default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

## OMP hyper threading ##
if (NOT DEFINED OMP_THREADS)
    set(OMP_THREADS "1")
    message("OMP_THREADS = ${OMP_THREADS} (DEFAULT=1 <=> no omp) ")
else ()
    message("OMP_THREADS = ${OMP_THREADS}")
endif ()

if (OMP_THREADS GREATER 1) # for running from console: export OMP_NUM_THREADS=12 && export OMP_DISPLAY_ENV=true && OMP_WAIT_POLICY=active && OMP_PROC_BIND=true && OMP_DYNAMIC=false
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif ()
endif ()

## MPI setup ##
if(NOT DEFINED USE_MPI)
    set(USE_MPI FALSE)
    message("USE_MPI = ${USE_MPI} (DEFAULT=FALSE <=> no mpi)")
else()
    message("USE_MPI = ${USE_MPI}")
endif()

if (USE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        include_directories(${MPI_INCLUDE_PATH})

        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
    else()
        set(USE_MPI TRUE)
    endif()
endif()

## Output directory ##
if (NOT DEFINED OUTPUT_DIR)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/debug)
else ()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
    message("CMAKE_RUNTIME_OUTPUT_DIRECTORY is ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif ()

## Project setup ##
add_executable(${PROJECT_NAME} main.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE OMP_THREADS=${OMP_THREADS}) # PROJECT_VERSION=${PROJECT_VERSION}

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cubature/libcubature-${CMAKE_BUILD_TYPE}.a)

# Matplotlib-cpp
include(../matplotlib-cpp/matplotlibcpp.cmake)

# GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES})

# Linear algebra with Eigen3 and lapack
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE -llapack)

