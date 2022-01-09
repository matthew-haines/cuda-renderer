#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <iomanip>

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
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
auto intersector = makeCUDAUnique<Intersector<Sphere, Triangle>>();
glm::vec3* accumulated;
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
  return {primitive.colour * M_1_PIf32, 1. / (2 * M_PI)};
}

// (dir, pdf)
__host__ __device__ std::pair<glm::vec3, float>
cosineSampleHemisphere(uint32_t& rngState) {
  // https://www.rorydriscoll.com/2009/01/07/better-sampling/
  float u0 = rand_uniform(rngState);
  float r = sqrtf(u0);
  float theta = 2 * M_PI * rand_uniform(rngState);
  float m = max(0., 1 - u0);
  return {{r * cosf(theta), r * sinf(theta), sqrtf(m)}, (1 / sqrtf(r * r + m)) * M_1_PI};
}

template <typename... Primitives>
__host__ __device__ glm::vec3
estimateDirect(const Intersector<Primitives...>& intersector,
               const Intersection& intersection, uint32_t& rngState) {
  glm::vec3 ld{};
  intersector.sampleLight(rngState, [&](auto& light, size_t nLights) {
    auto [pointOnLight, lightpdf] = light.sample(rngState);
    if (lightpdf > 0.) {
      glm::vec3 wi = glm::normalize(pointOnLight - intersection.intersection);
      auto [brdf, brdfpdf] = lambertianBRDF(*intersection.primitive);
      brdf *= abs(glm::dot(wi, intersection.normal));

      auto nearest = intersector.getIntersection(
          {intersection.intersection + 1e-5f * intersection.normal, wi});
      if (nearest.intersected()) {
        glm::vec3 li =
            nearest.primitive->id == light.id ? light.power * light.colour : glm::vec3{};
        ld += static_cast<float>(nLights) * brdf * li / lightpdf;
      }
    }
  });

  return ld;
}

__host__ __device__ float powerHeuristic(float pdfA, float pdfB) {
  return pdfA * pdfA / (pdfA * pdfA + pdfB * pdfB);
}

template <typename... Primitives>
__host__ __device__ glm::vec3
estimateDirectMIS(const Intersector<Primitives...>& intersector,
                  const Intersection& intersection, uint32_t& rngState) {
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

  glm::vec3 ld{};
  intersector.sampleLight(rngState, [&](auto& light, size_t nLights) {
    // Sample from light source
    auto [pointOnLight, lightpdf] = light.sample(rngState);
    if (lightpdf > 0.) {
      glm::vec3 wi = glm::normalize(pointOnLight - intersection.intersection);
      auto [brdf, brdfpdf] = lambertianBRDF(*intersection.primitive);
      brdf *= abs(glm::dot(wi, intersection.normal));

      auto nearest = intersector.getIntersection(
          {intersection.intersection + 1e-5f * intersection.normal, wi});
      if (nearest.intersected()) {
        glm::vec3 li =
            nearest.primitive->id == light.id ? light.power * light.colour : glm::vec3{};
        ld += static_cast<float>(nLights) * brdf * li * powerHeuristic(lightpdf, brdfpdf) / lightpdf;
      }
    }

    // Sample from BRDF
    auto [wi, brdfpdf] = cosineSampleHemisphere(rngState);
    if (brdfpdf > 0.) {
      auto [brdf, _] = lambertianBRDF(*intersection.primitive);
      brdf *= abs(glm::dot(wi, intersection.normal));

      auto nearest = intersector.getIntersection(
          {intersection.intersection + 1e-5f * intersection.normal, wi});
      if (nearest.intersected()) {
        glm::vec3 li =
            nearest.primitive->id == light.id ? light.power * light.colour : glm::vec3{};
        ld += static_cast<float>(nLights) * brdf * li * powerHeuristic(brdfpdf, lightpdf) / lightpdf; // LIGHTPDF
      }
    }
  });

  return ld;
}

template <typename... Primitives>
__global__ void
directLighting(cudaSurfaceObject_t outputSurface, glm::vec3* accumulated, size_t nSamples,
               int width, int height, Ray base, glm::vec3 dx, glm::vec3 dy,
               const Intersector<Primitives...>* intersector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t rngState = x * y * nSamples;
  if (x < width && y < height) {
    glm::vec3 dir = glm::normalize(
        base.direction + static_cast<float>(static_cast<int>(x) - width / 2) * dx +
        static_cast<float>(static_cast<int>(y) - height / 2) * dy);
    Ray ray{base.origin, dir};

    auto isect = intersector->getIntersection(ray);
    glm::vec3 l{0., 0., 0.};
    if (isect.intersected()) {
      l += isect.primitive->colour * isect.primitive->power + estimateDirect(*intersector, isect, rngState);
    }
    size_t idx = y * width + x;
    if (nSamples == 1)
      accumulated[idx] = glm::vec3{0., 0., 0.};
    accumulated[y * width + x] += l;

    l = glm::clamp(accumulated[y * width + x] / static_cast<float>(nSamples), 0.f, 1.f) *
        255.f;
    uchar4 colour{static_cast<unsigned char>(l.x), static_cast<unsigned char>(l.y),
                  static_cast<unsigned char>(l.z), 255};
    surf2Dwrite(colour, outputSurface, x * 4, y, cudaBoundaryModeTrap);
  }
}

template <typename... Primitives>
__global__ void
pathTrace(cudaSurfaceObject_t outputSurface, glm::vec3* accumulated, size_t nSamples,
               int width, int height, Ray base, glm::vec3 dx, glm::vec3 dy,
               const Intersector<Primitives...>* intersector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t rngState = x * y * nSamples;
  if (x < width && y < height) {
    glm::vec3 xOffset = (static_cast<float>(static_cast<int>(x) - width / 2) + rand_uniform(rngState) - 0.5f) * dx;
    glm::vec3 yOffset = (static_cast<float>(static_cast<int>(y) - height / 2) + rand_uniform(rngState) - 0.5f) * dy;
    Ray ray{base.origin, glm::normalize(base.direction + xOffset + yOffset)};
    glm::vec3 beta{1.};
    glm::vec3 L{0.};
    for (size_t i = 0; i < 4; ++i) {
      auto isect = intersector->getIntersection(ray);
      // If this is camera ray, then add light contribution if intersected
      if (i == 0 && isect.intersected())
        L += beta * isect.primitive->colour * isect.primitive->power;
      else if (!isect.intersected())
        break;

      L += beta * estimateDirect(*intersector, isect, rngState);

      auto [wi, brdfpdf] = cosineSampleHemisphere(rngState);
      wi = normalQuat(isect.normal) * wi;
      auto [brdf, _] = lambertianBRDF(*isect.primitive);
      if (brdfpdf == 0.)
        break;

      beta *= brdf * abs(glm::dot(wi, isect.normal)) / brdfpdf;
      ray = {isect.intersection + 1e-5f * isect.normal, wi};
    }

    size_t idx = y * width + x;
    if (nSamples == 1)
      accumulated[idx] = glm::vec3{0., 0., 0.};
    accumulated[y * width + x] += L;

    L = glm::clamp(accumulated[y * width + x] / static_cast<float>(nSamples), 0.f, 1.f) *
        255.f;
    uchar4 colour{static_cast<unsigned char>(L.x), static_cast<unsigned char>(L.y),
                  static_cast<unsigned char>(L.z), 255};
    surf2Dwrite(colour, outputSurface, x * 4, y, cudaBoundaryModeTrap);
  }
}

void display(GLFWwindow* window) {
  static size_t nSamples = 1;
  auto start = std::chrono::steady_clock::now();

  cudaCheck(cudaGraphicsMapResources(1, &outputTextureResource, nullptr));

  camera.update(window);
  if (camera.changed())
    nSamples = 1;

  cudaArray_t arr;
  cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, outputTextureResource, 0, 0));
  cudaResourceDesc desc{};
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = arr;
  cudaSurfaceObject_t outputSurface;
  cudaCheck(cudaCreateSurfaceObject(&outputSurface, &desc));

  // Work
  auto [ray, dx, dy] = camera.getState();
  runGridKernel<width, height>(
      pathTrace<Sphere, Triangle>, outputSurface, accumulated, nSamples, width,
      height, ray, dx, dy, intersector.get());
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaDestroySurfaceObject(outputSurface));
  cudaCheck(cudaGraphicsUnmapResources(1, &outputTextureResource, nullptr));

  fullscreenTexture(outputTexture);
  ++nSamples;

  auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::steady_clock::now() - start);
  auto ratio = std::chrono::duration<double, std::milli>{1000.} / elapsed;
  std::cout << "\rFPS: " << ratio << std::flush;
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
  cudaCheck(cudaMallocManaged(&accumulated, width * height * sizeof(glm::vec3)));

  // Allocate sphere buffer
  intersector->addPrimitives(std::vector<Sphere>{
      {{2.5, 5, 2.5}, 0.5, {1, 1, 1}, 1},
      {{2, 2, 2}, 1, {0, 0, 1}},
      {{4, 1.5, 3}, 0.5, {0, 1, 1}},
  });
  std::vector<Triangle> triangles;
  glm::vec3 dull{0.5, 0.5, 0.5};
  auto floor = makeQuad({0, 0, 0}, {0, 0, 5}, {5, 0, 5}, {5, 0, 0}, dull);
  auto backWall = makeQuad({0, 0, 5}, {0, 5, 5}, {5, 5, 5}, {5, 0, 5}, dull);
  auto frontWall = makeQuad({0, 0, 0}, {0, 5, 0}, {5, 5, 0}, {5, 0, 0}, dull);
  auto ceiling = makeQuad({0, 5, 0}, {0, 5, 5}, {5, 5, 5}, {5, 5, 0}, dull);
  auto leftWall = makeQuad({0, 0, 0}, {0, 5, 0}, {0, 5, 5}, {0, 0, 5}, {0.5, 0, 0});
  auto rightWall = makeQuad({5, 0, 0}, {5, 5, 0}, {5, 5, 5}, {5, 0, 5}, {0, 0.5, 0});
  triangles.insert(triangles.end(), floor.begin(), floor.end());
  triangles.insert(triangles.end(), backWall.begin(), backWall.end());
  triangles.insert(triangles.end(), frontWall.begin(), frontWall.end());
  triangles.insert(triangles.end(), ceiling.begin(), ceiling.end());
  triangles.insert(triangles.end(), leftWall.begin(), leftWall.end());
  triangles.insert(triangles.end(), rightWall.begin(), rightWall.end());
  intersector->addPrimitives(triangles);
}

void teardown() { cudaGraphicsUnregisterResource(outputTextureResource); }

int main() {
  std::cout << std::setprecision(2) << std::endl;

  GLManager glManager{width, height};
  camera.position = {2.5, 2.5, 1};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

  setup();

  glManager.renderLoop(display, std::chrono::duration<float>{0.});

  teardown();
}
