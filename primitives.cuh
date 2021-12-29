#ifndef RENDERER_PRIMITIVES_CUH
#define RENDERER_PRIMITIVES_CUH

#include <glm/glm.hpp>

#include "ray.h"

struct Primitive {
  Primitive(glm::vec3 colour = {}, float power = 0.);

  glm::vec3 colour;
  float power;
  size_t id;
};

struct Intersection {
  float distance;
  glm::vec3 intersection;
  glm::vec3 normal;
  const Primitive* primitive;

  __host__ __device__ bool intersected() const { return distance > 0.; }
};

struct Sphere : public Primitive {
  Sphere(glm::vec3 position, float radius, glm::vec3 colour, float power = 0);

  __host__ __device__ Intersection intersect(const Ray& ray) const;
  // (point, pdf)
  __host__ __device__ std::pair<glm::vec3, float> sample(uint32_t& rngState) const;
  __host__ __device__ std::pair<glm::vec3, float>
  sample(glm::vec3 point, uint32_t& rngState) const;

  glm::vec3 position;
  float radius;
};

struct Triangle : public Primitive {
  Triangle(glm::vec3 p0, glm::vec3 e1, glm::vec3 e2, glm::vec3 colour, float power = 0);

  __host__ __device__ glm::vec3 p1() const { return p0 + e1; }
  __host__ __device__ glm::vec3 p2() const { return p0 + e2; }

  __host__ __device__ Intersection intersect(const Ray& ray) const;
  // (point, pdf)
  __host__ __device__ std::pair<glm::vec3, float> sample(uint32_t& rngState) const;
  __host__ __device__ std::pair<glm::vec3, float>
  sample(glm::vec3 point, uint32_t& rngState) const;

  glm::vec3 p0;
  glm::vec3 e1;
  glm::vec3 e2;
};

#endif // RENDERER_PRIMITIVES_CUH
