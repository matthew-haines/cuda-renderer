#include <iostream>
#include <utility>
#include <tuple>
#include <vector>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "gl_manager.h"
#include "cuda_helpers.h"
#include "helper_math.h"
#include "glm/ext/matrix_transform.hpp"

#include <cuda.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// #define GLM_FORCE_CUDA
#include <glm/glm.hpp>

constexpr float epsilon = 1e-6;

template <typename T>
struct FormatSymbol {};

template <>
struct FormatSymbol<int> {
  static constexpr char * const symbol = "%d";
};

template <>
struct FormatSymbol<float> {
  static constexpr char * const symbol = "%f";
};

template <int Length, typename T, glm::qualifier Q>
__host__ __device__ void printVector(const glm::vec<Length, T, Q>& v) {
  for (size_t i = 0; i < Length; ++i) {
    printf(FormatSymbol<T>::symbol, v[i]);
    printf(" ");
  }
  printf("\n");
}

struct Triangle {
  __host__ __device__ glm::vec3 p1() const { return p0 + e1; }
  __host__ __device__ glm::vec3 p2() const { return p0 + e2; }

  glm::vec3 p0;
  glm::vec3 e1;
  glm::vec3 e2;
};

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Sphere {
  glm::vec3 position;
  float radius;

  // (dist, intersect, normal)
  __host__ __device__ std::tuple<float, glm::vec3, glm::vec3> intersect(const Ray& ray) const {
    glm::vec3 dist = position - ray.origin;
    float dist2 = glm::dot(dist, dist);
    float lengthToClosest = glm::dot(dist, ray.direction);
    float radius2 = radius * radius;
    if (lengthToClosest < -epsilon)
      return {-1., {}, {}};
    float halfChordDistance2 = radius2 - dist2 + lengthToClosest * lengthToClosest;
    if (halfChordDistance2 < -epsilon)
      return {-1., {}, {}};
    float intersectDistance;
    if (dist2 > radius2) intersectDistance = lengthToClosest - sqrtf(halfChordDistance2);
    else
      intersectDistance = lengthToClosest + sqrtf(halfChordDistance2);
    glm::vec3 isect = ray.direction * intersectDistance + ray.origin;
    return {intersectDistance, isect, (isect - position) / radius};
  }
};

static const int width = 1920;
static const int height = 1080;

static const float fov = M_PI / 2;

__global__ void trace(cudaSurfaceObject_t outputSurface, int width, int height, Ray base,
                      glm::vec3 dx, glm::vec3 dy, Sphere* deviceSpheres, size_t n) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    glm::vec3 dir = glm::normalize(base.direction + static_cast<float>(static_cast<int>(x) - width / 2) * dx + static_cast<float>(static_cast<int>(y) - height / 2) * dy);
    Ray ray{base.origin, dir};

    float closest = 1e6;
    bool intersected = false;
    glm::vec3 closestIntersect;
    glm::vec3 closestNormal;

    for (size_t i = 0; i < n; ++i) {
      auto [dist, intersect, normal] = deviceSpheres[i].intersect(ray);
      if (dist >= 0 && dist < closest) {
        closest = dist;
        intersected = true;
        closestIntersect = intersect;
        closestNormal = normal;
      }
    }

    uchar4 colour;
    unsigned char brightness = glm::dot(-ray.direction, closestNormal) * 255;
    if (intersected) {
      colour = {brightness, 0, 0, 255};
    } else {
      colour = {0, 0, 0, 0};
    }
    surf2Dwrite(colour, outputSurface, x * 4, y, cudaBoundaryModeTrap);
  }
}

GLuint outputTexture;
cudaGraphicsResource* outputTextureResource;
Sphere* deviceSpheres;
size_t n;

template <int width, int height, typename T, typename... Args>
void runGridKernel(T kernel, Args&&... args) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernel<<<numBlocks, threadsPerBlock>>>(args...);
}

// (xrot, yrot)
glm::vec2 angles{0, 0};
glm::vec3 pos{0, 0, 0};

void display(GLFWwindow* window) {
  cudaCheck(cudaGraphicsMapResources(1, &outputTextureResource, nullptr));

  // update pos/dir
  if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
    angles.x -= 0.03;
  }
  if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
    angles.x += 0.03;
  }
  angles.x = clamp(angles.x, -M_PI_2, M_PI_2);
  if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
    angles.y -= 0.03;
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
    angles.y += 0.03;
  }
  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
    angles = {0, 0};
  }
  // Spherical to Cartesian
  auto dir = glm::vec3{sinf(angles.y) * cosf(angles.x), sinf(angles.x), cosf(angles.y) * cosf(angles.x)};
  auto posx = glm::normalize(glm::cross(dir, {0, 1, 0}));
  auto up = glm::cross(dir, posx);
  glm::mat3 mat{posx, up, dir};

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    pos += mat * glm::vec3{0, 0, 0.1};
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    pos -= mat * glm::vec3{0.1, 0, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    pos -= mat * glm::vec3{0, 0, 0.1};
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    pos += mat * glm::vec3{0.1, 0, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    pos += glm::vec3{0, 0.1, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    pos -= glm::vec3{0, 0.1, 0};
  }

  cudaArray_t arr;
  cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, outputTextureResource, 0, 0));
  cudaResourceDesc desc{};
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr;
  cudaSurfaceObject_t outputSurface;
  cudaCheck(cudaCreateSurfaceObject(&outputSurface, &desc));

  float cameraWidth = 2 * tanf(fov / 2);
  float gridSize = cameraWidth / width;
  glm::vec3 dx = gridSize * mat * glm::vec3{1, 0, 0};
  glm::vec3 dy = gridSize * mat * glm::vec3{0, 1, 0};
  glm::vec3 base = mat * glm::vec3{0, 0, 1};

  // Work
  runGridKernel<width, height>(trace, outputSurface, width, height, Ray{pos, base}, dx, dy, deviceSpheres, n);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDestroySurfaceObject(outputSurface));
  cudaCheck(cudaGraphicsUnmapResources(1, &outputTextureResource, nullptr));

  fullscreenTexture(outputTexture);
}

void setup() {
  // Allocate Render Texture
  glGenTextures(1, &outputTexture);
  glBindTexture(GL_TEXTURE_2D, outputTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  cudaCheck(cudaGraphicsGLRegisterImage(&outputTextureResource, outputTexture,
                                        GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

  // Allocate sphere buffer
  std::vector<Sphere> spheres{
    {{0, 0, 3}, 1},
    {{1, 1, 5}, 1},
    {{-1, -1, 2}, 0.25},
  };
  auto bytes = spheres.size() * sizeof(Sphere);

  cudaCheck(cudaMalloc(&deviceSpheres, bytes));
  cudaCheck(cudaMemcpy(deviceSpheres, spheres.data(), bytes, cudaMemcpyHostToDevice));
  n = spheres.size();
}

void teardown() {
  cudaFree(deviceSpheres);
  cudaGraphicsUnregisterResource(outputTextureResource);
}

int main() {
  GLManager glManager{width, height};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

  setup();

  glManager.renderLoop(display, 1./60. * std::chrono::duration<float, std::chrono::seconds::period>{1.});

  teardown();
}