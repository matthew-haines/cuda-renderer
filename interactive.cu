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

void cudaCheck(cudaError err, int lineNumber, std::string_view fileName) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error " << err << " " << cudaGetErrorString(err)
              << "\nLocation: " << fileName << ":" << lineNumber;
    std::exit(err);
  }
}
#define cudaCheck(err) cudaCheck(err, __LINE__, __FILE__)

static const int width = 1024;
static const int height = 768;

GLuint tex;
cudaGraphicsResource *cudaResource;

// H: [0, 360), S: [0, 1], L: [0, 1]
__host__ __device__ uchar4 HSLToRGB(float4 pixel) {
  auto [h, s, l, _] = pixel;
  auto c = (1 - abs(2 * l - 1)) * s;
  auto x = c * (1 - abs(fmod(h / 60, (float) 2) - 1));
  auto m = l - c / 2;

  float4 rgbf;
  if (h < 60) rgbf = {c, x, 0, 0};
  else if (h < 120)
    rgbf = {x, c, 0, 0};
  else if (h < 180)
    rgbf = {0, c, x, 0};
  else if (h < 240)
    rgbf = {0, x, c, 0};
  else if (h < 300)
    rgbf = {x, 0, c, 0};
  else
    rgbf = {c, 0, x, 0};

  auto rgbff = (rgbf + float4{m, m, m, 0}) * 255;
  return make_uchar4(rgbff.x, rgbff.y, rgbff.z, 0);
}

__global__ void kernel(cudaSurfaceObject_t surf, int width, int height, float shift) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uchar4 data =
      HSLToRGB({fmod(static_cast<float>(x) / static_cast<float>(width) + shift, 1.f) * 360,
                fmod(static_cast<float>(y) / static_cast<float>(height) + shift, 1.f), 0.50, 0});
    surf2Dwrite(data, surf, x * 4, y, cudaBoundaryModeTrap);
  }
}

void fullscreenTexture(GLuint tex) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  {
    glTexCoord2f(0, 1);
    glVertex3f(0, 0, 0);

    glTexCoord2f(0, 0);
    glVertex3f(0, 1, 0);

    glTexCoord2f(1, 0);
    glVertex3f(1, 1, 0);

    glTexCoord2f(1, 1);
    glVertex3f(1, 0, 0);
  }
  glEnd();
}

void display() {
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

int main(int argc, char *argv[]) {
  if (!glfwInit()) {
    std::cerr << "GLFW init failed" << std::endl;
    return 1;
  }

  GLFWwindow *window;
  window = glfwCreateWindow(width, height, "Renderer", nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "GLFW window creation failed" << std::endl;
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed" << std::endl; }

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

  do {
    glClearColor(0.2, 0.2, 0.2, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // work
    display();

    glfwSwapBuffers(window);
    glfwPollEvents();
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

  return 0;
}