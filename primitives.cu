#include "primitives.cuh"

#include <glm/gtx/intersect.hpp>

#include "random.cuh"

static size_t globalID = 0;

Primitive::Primitive(glm::vec3 colour, float power) : colour{colour}, power{power}, id{++globalID} {}

Sphere::Sphere(glm::vec3 position, float radius, glm::vec3 colour, float power) : Primitive{colour, power}, position{position}, radius{radius} {}

__host__ __device__ Intersection Sphere::intersect(const Ray& ray) const {
  glm::vec3 dist = position - ray.origin;
  float dist2 = glm::dot(dist, dist);
  float lengthToClosest = glm::dot(dist, ray.direction);
  float radius2 = radius * radius;
  if (lengthToClosest < 0)
    return {-1., {}, {}, {}};
  float halfChordDistance2 = radius2 - dist2 + lengthToClosest * lengthToClosest;
  if (halfChordDistance2 < 0)
    return {-1., {}, {}, {}};
  float intersectDistance;
  if (dist2 > radius2)
    intersectDistance = lengthToClosest - sqrtf(halfChordDistance2);
  else
    intersectDistance = lengthToClosest + sqrtf(halfChordDistance2);
  glm::vec3 isect = ray.direction * intersectDistance + ray.origin;
  return {intersectDistance, isect, (isect - position) / radius, this};
}

__host__ __device__ std::pair<glm::vec3, float> Sphere::sample(uint32_t& rngState) const {
  float z = 1 - 2 * rand_uniform(rngState);
  float r = sqrtf(max(0., 1 - z * z));
  float phi = 2 * M_PI * rand_uniform(rngState);
  glm::vec3 unit{r * cosf(phi), r * sinf(phi), z};
  return {radius * unit + position, 1. / (4. * M_PI * radius * radius)};
}

// PBRT
__host__ __device__ std::pair<glm::vec3, float>
Sphere::sample(glm::vec3 point, uint32_t& rngState) const {
  return sample(rngState);
// For now, ignore hemisphere sampling
//  auto dc = glm::distance(point, position);
//  if (dc < radius)
//    return sample(rngState);
//
//  float sinThetaMax2 = radius * radius / (dc * dc);
//  float cosThetaMax = sqrtf(max(0., 1 - sinThetaMax2));
//  float u0 = rand_uniform(rngState);
//  float u1 = rand_uniform(rngState);
//  float cosTheta = (1 - u0) + u0 * cosThetaMax;
//  float sinTheta = sqrtf(max(0., 1 - cosTheta * cosTheta));
//  float phi = u1 * 2 * M_PI;
//
//  float ds =
//      dc * cosTheta - sqrtf(max(0., radius * radius - dc * dc * sinTheta * sinTheta));
//  float cosAlpha = (dc * dc + radius * radius - ds * ds) / (2 * dc * radius);
//  float sinAlpha = sqrtf(max(0.0, 1 - cosAlpha * cosAlpha));
}

Triangle::Triangle(glm::vec3 p0, glm::vec3 e1, glm::vec3 e2, glm::vec3 colour, float power) : Primitive{colour, power}, p0{p0}, e1{e1}, e2{e2} {}

__host__ __device__ Intersection Triangle::intersect(const Ray& ray) const {
  // Muller Trombore
  glm::vec3 p = glm::cross(ray.direction, e2);
  float determinant = glm::dot(e1, p);
  if (std::abs(determinant) < 0)
    return {-1, {}, {}, {}};

  float invDeterminant = 1.f / determinant;
  glm::vec3 origin = ray.origin - p0;
  float u = invDeterminant * glm::dot(origin, p); // barycentric
  if (u < 0. || u > 1.) {
    return {-1, {}, {}, {}};
  }
  glm::vec3 q = glm::cross(origin, e1);
  float v = glm::dot(ray.direction, q) * invDeterminant;
  if (v < 0. || u + v > 1.) {
    return {-1, {}, {}, {}};
  }
  float distance = glm::dot(e2, q) * invDeterminant;
  glm::vec3 intersect, normal;
  if (distance > 0) {
    intersect = ray.origin + distance * ray.direction;
    normal = glm::normalize(glm::cross(e1, e2)); // might want to precompute
    normal = determinant > 0. ? normal : -normal;
    return {distance, intersect, normal, this};
  }
  return {-1, {}, {}, {}};
}
// Not implemented
__host__ __device__ std::pair<glm::vec3, float>
Triangle::sample(uint32_t& rngState) const {
  return {};
}
__host__ __device__ std::pair<glm::vec3, float>
Triangle::sample(glm::vec3 point, uint32_t& rngState) const {
  return {};
}

std::vector<Triangle> makeQuad(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 colour, float power) {
  return {
    {p0, p1 - p0, p2 - p0, colour, power},
    {p0, p2 - p0, p3 - p0, colour, power},
  };
}
