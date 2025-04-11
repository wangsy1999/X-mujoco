cmake_minimum_required(VERSION 3.14)

project(bitbot)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BITBOT_DEPENDENCY_USE_PROXY TRUE)
set(BITBOT_DEPENDENCY_USE_LOCAL_FILE TRUE)
set(BITBOT_DEPENDENCY_LOCAL_FILE_PATH ${CMAKE_SOURCE_DIR}/dependencies)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/wsy/Library/libtorch/)
include_directories(${TORCH_PATH}/include)

include(bitbot.cmake)

include_directories("/usr/include/eigen3")

BitbotAddDependency(
  NAME           bitbot_mujoco
  FILE_NAME      "bitbot_mujoco-main.zip"
  FILE_PATH      ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
  USE_LOCAL_FILE TRUE
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package(Torch REQUIRED)
add_executable(main_app 
  main.cpp
  user_func.cpp
  leeMatrix.c
)

target_include_directories(main_app PUBLIC
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(main_app
  BitbotMujoco
  ${TORCH_LIBRARIES}
)
target_compile_options(main_app PUBLIC)
target_compile_features(main_app PUBLIC cxx_std_20)
