#ifndef RENDERER_RAY_H
#define RENDERER_RAY_H

#include <glm/glm.hpp>

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

#endif // RENDERER_RAY_H
