cmake_minimum_required(VERSION 3.22)
project(renderer CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(CUDAToolkit REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_subdirectory(glm)

set(
        HEADERS
        gl_manager.h
        helper_math.h
        cuda_helpers.h
        helper_kernels.cuh
)

set(
        SOURCES
        gl_manager.cpp
        cuda_helpers.cpp
        helper_kernels.cu
)

add_library(renderlib ${HEADERS} ${SOURCES})
target_link_libraries(renderlib OpenGL::GL glfw GLEW::GLEW CUDA::cudart glm::glm)
set_target_properties(renderlib PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(interactive interactive.cu)
target_link_libraries(interactive renderlib)
set_target_properties(interactive PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(rt rt.cu)
target_link_libraries(rt renderlib)
set_target_properties(rt PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON)
