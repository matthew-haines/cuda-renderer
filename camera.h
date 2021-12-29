#ifndef RENDERER_CAMERA_H
#define RENDERER_CAMERA_H

#include <algorithm>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "ray.h"

// Frame: +z = in, +x = right, +y = up
class Camera {
public:
  Camera(glm::uvec2 dim, float fov = M_PI_2, glm::vec2 angles = {},
         glm::vec3 position = {}, glm::mat3 viewMatrix = glm::mat3{1});

  void update(GLFWwindow* window);

  // (Camera ray, dx, dy)
  std::tuple<Ray, glm::vec3, glm::vec3> getState() const;

private:
  glm::uvec2 dim;
  float fov;
  glm::vec2 angles; // (xrot, yrot)
  glm::vec3 position;
  glm::mat3 viewMatrix;
};

#endif // RENDERER_CAMERA_H
