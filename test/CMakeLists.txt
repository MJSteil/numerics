# https://www.jetbrains.com/help/clion/boost-test-support.html#
# install on ubuntu with sudo apt-get install libboost-test-dev

set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Boost_Test_BSAM BSAM/BSAM.cpp)
target_link_libraries(Boost_Test_BSAM ${Boost_LIBRARIES})

add_executable(Boost_Test_cubature cubature_function/cubature_function.cpp)
target_link_libraries(Boost_Test_cubature ${Boost_LIBRARIES} ${CMAKE_CURRENT_SOURCE_DIR}/../cubature/libcubature-Debug.a)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

add_executable(Boost_Test_DLi DLi/DLi.cpp)
target_link_libraries(Boost_Test_DLi ${Boost_LIBRARIES} ${GSL_LIBRARIES} Python3::Python Python3::Module Python3::NumPy)

add_executable(Boost_Test_matplotlibcpp matplotlibcpp/matplotlibcpp.cpp)
target_link_libraries(Boost_Test_matplotlibcpp ${Boost_LIBRARIES} Python3::Python Python3::Module Python3::NumPy)

add_executable(Boost_Test_NM NM/NM.cpp)
target_link_libraries(Boost_Test_NM ${Boost_LIBRARIES} Python3::Python Python3::Module Python3::NumPy)