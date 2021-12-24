cmake_minimum_required(VERSION 3.22)
project(renderer CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_executable(interactive interactive.cu gl_manager.h)
target_link_libraries(interactive OpenGL::GL glfw GLEW::GLEW ${CUDA_CUDA_LIBRARY})
set_target_properties(interactive PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(interactive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
