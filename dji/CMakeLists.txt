cmake_minimum_required(VERSION 3.16)
project(dji)

include_directories("include")

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(OpenCV_DIR "D:\\opencv-451-contrib-vs")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(Torch_DIR "D:\\libtorch\\share\\cmake\\Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

aux_source_directory(src SRC_LIST)
add_executable(dji ${SRC_LIST})
target_link_libraries(dji ${TORCH_LIBRARIES})
target_link_libraries(dji ${OpenCV_LIBS})

add_executable(test test/test.cpp)
target_link_libraries(test ${TORCH_LIBRARIES})
target_link_libraries(test ${OpenCV_LIBS})

add_executable(opencvTest test/opencvTest.cpp)
target_link_libraries(opencvTest ${OpenCV_LIBS})

add_executable(templateTest test/templateTest.cpp)
target_link_libraries(templateTest ${OpenCV_LIBS})

add_executable(torchTest test/torchTest.cpp)
target_link_libraries(torchTest ${TORCH_LIBRARIES})
target_link_libraries(torchTest ${OpenCV_LIBS})