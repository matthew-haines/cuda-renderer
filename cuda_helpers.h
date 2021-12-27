#ifndef RENDERER_CUDA_HELPERS_H
#define RENDERER_CUDA_HELPERS_H

#include <string_view>

#include <cuda_runtime.h>
#include <GL/glew.h>

void cudaCheckImpl(cudaError err, int lineNumber, std::string_view fileName);

#define cudaCheck(err) cudaCheckImpl(err, __LINE__, __FILE__)

void fullscreenTexture(GLuint tex);

#endif//RENDERER_CUDA_HELPERS_H
