cmake_minimum_required(VERSION 3.22)
project(renderer CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(CUDAToolkit REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Boost 1.78 REQUIRED)

add_subdirectory(glm)

set(
        HEADERS
        gl_manager.h
        helper_math.h
        cuda_helpers.cuh
        helper_kernels.cuh
        primitives.cuh
        camera.h
        ray.h
        intersector.cuh
        random.cuh
)

set(
        SOURCES
        gl_manager.cpp
        cuda_helpers.cu
        helper_kernels.cu
        primitives.cu
        camera.cpp
        random.cu
)

add_library(renderlib ${HEADERS} ${SOURCES})
target_link_libraries(renderlib OpenGL::GL glfw GLEW::GLEW CUDA::cudart glm::glm Boost::boost)
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
