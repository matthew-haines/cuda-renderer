#include "gl_manager.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string_view>
#include <surface_functions.h>

#include "helper_math.h"
#include "cuda_helpers.h"
#include "helper_kernels.cuh"

static const int width = 1024;
static const int height = 768;

GLuint tex;
cudaGraphicsResource *cudaResource;

__global__ void kernel(cudaSurfaceObject_t surf, int width, int height, float shift) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uchar4 data =
      HSLToRGB({fmod(static_cast<float>(x) / static_cast<float>(width) + shift, 1.f) * 360,
                sqrt(fmod(static_cast<float>(y) / static_cast<float>(height) + shift, 1.f)), 0.50, 0});
    surf2Dwrite(data, surf, x * 4, y, cudaBoundaryModeTrap);
  }
}

void display(GLFWwindow*) {
  static float shift = 0;

  cudaCheck(cudaGraphicsMapResources(1, &cudaResource, nullptr));

  cudaArray_t arr;
  cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, cudaResource, 0, 0));
  cudaResourceDesc desc{};
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr;
  cudaSurfaceObject_t surfaceObject;
  cudaCheck(cudaCreateSurfaceObject(&surfaceObject, &desc));

  // Work
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernel<<<numBlocks, threadsPerBlock>>>(surfaceObject, width, height, shift);

  cudaCheck(cudaDestroySurfaceObject(surfaceObject));
  cudaCheck(cudaGraphicsUnmapResources(1, &cudaResource, nullptr));

  fullscreenTexture(tex);

  shift = fmod(shift + 0.01, 1);
}

int main() {
  GLManager glManager{width, height};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

  std::array<uchar4, width * height> local{};
  local.fill({255, 255, 255, 255});
  // Allocate render tex
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               local.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  cudaCheck(
    cudaGraphicsGLRegisterImage(&cudaResource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

  glManager.renderLoop(display, 1./60. * std::chrono::duration<float, std::chrono::seconds::period>{1.});

  return 0;
}