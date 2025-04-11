# cmake_minimum_required(VERSION 3.10)

# project(NeuralNetworkProject)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
# set(HEADERS_DIR "${SRC_DIR}/headers")
# set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen")
# # set(EIGEN3_INCLUDE_DIR "${EIGEN_DIR}/Eigen")

# file(GLOB HEADER_SOURCES "${HEADERS_DIR}/*.cpp")

# set(SOURCES
#     ${SRC_DIR}/main.cpp
#     ${SRC_DIR}/main_check.cpp
#     ${HEADER_SOURCES}
# )

# # find_package(Eigen3 REQUIRED)
# find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

# add_executable(main ${SRC_DIR}/main.cpp ${HEADER_SOURCES})
# add_executable(main_check ${SRC_DIR}/main_check.cpp ${HEADER_SOURCES})
# add_executable(console_app ${SRC_DIR}/console_app.cpp ${HEADER_SOURCES})


# target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIR} ${HEADERS_DIR})
# target_include_directories(main_check PRIVATE ${EIGEN3_INCLUDE_DIR} ${HEADERS_DIR})
# target_include_directories(console_app PRIVATE ${EIGEN3_INCLUDE_DIR} ${HEADERS_DIR})

# include_directories(external/eigen)

# target_link_libraries(main PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets)
# target_link_libraries(main_check PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets)
# target_link_libraries(console_app PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets)

# set_target_properties(main PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
# set_target_properties(main_check PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
# set_target_properties(console_app PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)

# ------------------------------------------------

cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(HEADERS_DIR "${SRC_DIR}/headers")

add_subdirectory(external/eigen)

file(GLOB HEADER_SOURCES "${HEADERS_DIR}/*.cpp")

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/main_check.cpp
    ${HEADER_SOURCES}
)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

add_executable(main ${SRC_DIR}/main.cpp ${HEADER_SOURCES})
add_executable(main_check ${SRC_DIR}/main_check.cpp ${HEADER_SOURCES})
add_executable(console_app ${SRC_DIR}/console_app.cpp ${HEADER_SOURCES})

target_include_directories(main PRIVATE ${HEADERS_DIR})
target_include_directories(main_check PRIVATE ${HEADERS_DIR})
target_include_directories(console_app PRIVATE ${HEADERS_DIR})

target_link_libraries(main PRIVATE Eigen3::Eigen Qt6::Core Qt6::Gui Qt6::Widgets)
target_link_libraries(main_check PRIVATE Eigen3::Eigen Qt6::Core Qt6::Gui Qt6::Widgets)
target_link_libraries(console_app PRIVATE Eigen3::Eigen Qt6::Core Qt6::Gui Qt6::Widgets)

set_target_properties(main PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
set_target_properties(main_check PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
set_target_properties(console_app PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
