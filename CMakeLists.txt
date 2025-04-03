cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(HEADERS_DIR "${SRC_DIR}/headers")

file(GLOB HEADER_SOURCES "${HEADERS_DIR}/*.cpp")

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/main_check.cpp
    ${HEADER_SOURCES}
)

add_executable(main ${SRC_DIR}/main.cpp ${HEADER_SOURCES})
add_executable(main_check ${SRC_DIR}/main_check.cpp ${HEADER_SOURCES})

find_package(Eigen3 REQUIRED)

target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR} ${HEADERS_DIR})
target_include_directories(main_check PRIVATE ${EIGEN3_INCLUDE_DIR} ${HEADERS_DIR})
