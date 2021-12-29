#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// #define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

#include "camera.h"
#include "cuda_helpers.cuh"
#include "gl_manager.h"
#include "helper_math.h"
#include "intersector.cuh"
#include "primitives.cuh"
#include "random.cuh"

static const int width = 1200;
static const int height = 900;

GLuint outputTexture;
cudaGraphicsResource* outputTextureResource;
auto intersector = makeCUDAUnique<Intersector<Sphere>>();
Camera camera{glm::uvec2{width, height}};

template <typename... Primitives>
__global__ void trace(cudaSurfaceObject_t outputSurface, int width, int height, Ray base,
                      glm::vec3 dx, glm::vec3 dy,
                      const Intersector<Primitives...>* intersector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    glm::vec3 dir = glm::normalize(
        base.direction + static_cast<float>(static_cast<int>(x) - width / 2) * dx +
        static_cast<float>(static_cast<int>(y) - height / 2) * dy);
    Ray ray{base.origin, dir};

    auto [dist, isect, normal, _] = intersector->getIntersection(ray);
    bool intersected = dist > 0;

    uchar4 colour;
    unsigned char brightness = glm::dot(-ray.direction, normal) * 255;
    if (intersected) {
      colour = {brightness, 0, 0, 255};
    } else {
      colour = {0, 0, 0, 0};
    }
    surf2Dwrite(colour, outputSurface, x * 4, y, cudaBoundaryModeTrap);
  }
}

// (f, pdf)
template <typename T>
__host__ __device__ std::pair<glm::vec3, float> lambertianBRDF(const T& primitive) {
  return {primitive.colour * M_1_PIf32, 2 * M_PI};
}

// (dir, pdf)
__host__ __device__ std::pair<glm::vec3, float>
cosineSampleHemisphere(uint32_t& rngState) {
  // https://www.rorydriscoll.com/2009/01/07/better-sampling/
  float u0 = rand_uniform(rngState);
  float r = sqrtf(u0);
  float theta = 2 * M_PI * rand_uniform(rngState);
  return {{r * cosf(theta), r * sinf(theta), sqrtf(max(0., 1 - u0))}, cosf(theta) / M_PI};
}

template <typename... Primitives>
__global__ void directLighting(cudaSurfaceObject_t outputSurface, int width, int height,
                               Ray base, glm::vec3 dx, glm::vec3 dy,
                               const Intersector<Primitives...>* intersector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t rngState = x * y;
  if (x < width && y < height) {
    glm::vec3 dir = glm::normalize(
        base.direction + static_cast<float>(static_cast<int>(x) - width / 2) * dx +
        static_cast<float>(static_cast<int>(y) - height / 2) * dy);
    Ray ray{base.origin, dir};

    const size_t samples = 10;

    auto isect = intersector->getIntersection(ray);
    glm::vec3 l{0., 0., 0.};
    if (isect.intersected()) {
      glm::vec3 lt{0., 0., 0.};
      for (size_t i = 0; i < samples; ++i) {
        lt += isect.primitive->colour * isect.primitive->power;
        // choose random scene light
        // estimate direct(p, wi) = W sr^-1 m^-2 \sim nLights * direct(p, wi, light)
        // Estimate direct(p, wi, light) = int_{H^2} f(p, w0, wi) Ld(p, wi) |cos(theta_i)|
        // dwi:
        //   Sample Light:
        //     Sample a direction from light -> p, with pdf of that and incident radiance
        //     Compute BRDF for that wi and get |cos(theta_i)|
        //     Set to 0 if not visible
        //     Get MIS contribution with power heuristic, n = 1
        //
        //   Sample BRDF:
        //     Sample a direction from p, with pdf
        //     Compute BRDF and light pdf for direction if intersected
        //     Get MIS contribution with power heuristic, n = 1

        // IGNORE MIS FOR NOW
        intersector->sampleLight(rngState, [&](auto& l, size_t nLights) {
          auto [pointOnLight, lightpdf] = l.sample(rngState);
          if (lightpdf > 0.) {
            glm::vec3 wi = glm::normalize(pointOnLight - isect.intersection);
            auto [brdf, brdfpdf] = lambertianBRDF(*isect.primitive);
            brdf *= abs(glm::dot(wi, isect.normal));

            auto nearest = intersector->getIntersection({isect.intersection, wi});
            if (nearest.intersected()) {
              glm::vec3 li =
                  nearest.primitive->id == l.id ? l.power * l.colour : glm::vec3{};
              lt += static_cast<float>(nLights) * brdf * li / lightpdf;
            }
          }
        });
      }
      l += lt;
    }
    l /= static_cast<float>(samples);

    l = glm::clamp(l, 0.f, 1.f) * 255.f;
    uchar4 colour{static_cast<unsigned char>(l.x), static_cast<unsigned char>(l.y),
                  static_cast<unsigned char>(l.z), 255};
    surf2Dwrite(colour, outputSurface, x * 4, y, cudaBoundaryModeTrap);
  }
}

void display(GLFWwindow* window) {
  cudaCheck(cudaGraphicsMapResources(1, &outputTextureResource, nullptr));

  camera.update(window);

  cudaArray_t arr;
  cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, outputTextureResource, 0, 0));
  cudaResourceDesc desc{};
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr;
  cudaSurfaceObject_t outputSurface;
  cudaCheck(cudaCreateSurfaceObject(&outputSurface, &desc));

  // Work
  auto [ray, dx, dy] = camera.getState();
  runGridKernel<width, height>(directLighting<Sphere>, outputSurface, width, height, ray,
                               dx, dy, intersector.get());
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
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  cudaCheck(cudaGraphicsGLRegisterImage(
      &outputTextureResource, outputTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

  // Allocate sphere buffer
  intersector->addPrimitives(std::vector<Sphere>{
      {{0, 0, 3}, 1, {1, 1, 1}},
      {{-2, -2, 1}, 1, {1, 0, 0}},
      {{-1, -1, 2}, 0.25, {1, 1, 1}, 1},
  });
//  intersector->addPrimitives(std::vector<Triangle>{
//      {{}}
//  });
}

void teardown() { cudaGraphicsUnregisterResource(outputTextureResource); }

int main() {
  GLManager glManager{width, height};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

  setup();

  glManager.renderLoop(
      display, 1. / 60. * std::chrono::duration<float, std::chrono::seconds::period>{1.});

  teardown();
}