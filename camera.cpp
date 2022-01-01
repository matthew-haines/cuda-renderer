#include "camera.h"

Camera::Camera(glm::uvec2 dim, float fov, glm::vec2 angles, glm::vec3 position,
               glm::mat3 viewMatrix)
    : dim{dim}, fov{fov}, angles{angles}, position{position}, viewMatrix{viewMatrix} {}

void Camera::update(GLFWwindow* window) {
  oldAngles = angles;
  oldPosition = position;

  if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
    angles.x -= 0.03;
  }
  if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
    angles.x += 0.03;
  }
  angles.x = std::clamp<float>(angles.x, -M_PI_2, M_PI_2);
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
  auto dir = glm::vec3{sinf(angles.y) * cosf(angles.x), sinf(angles.x),
                       cosf(angles.y) * cosf(angles.x)};
  auto posx = glm::normalize(glm::cross(dir, {0, 1, 0}));
  auto up = glm::cross(dir, posx);
  viewMatrix = glm::mat3{posx, up, dir};

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    position += viewMatrix * glm::vec3{0, 0, 0.1};
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    position -= viewMatrix * glm::vec3{0.1, 0, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    position -= viewMatrix * glm::vec3{0, 0, 0.1};
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    position += viewMatrix * glm::vec3{0.1, 0, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    position += glm::vec3{0, 0.1, 0};
  }
  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
    position -= glm::vec3{0, 0.1, 0};
  }
}

bool Camera::changed() const {
  return oldAngles != angles || oldPosition != position;
}

std::tuple<Ray, glm::vec3, glm::vec3> Camera::getState() const {
  float cameraWidth = 2 * tanf(fov / 2);
  float gridSize = cameraWidth / dim.x;
  glm::vec3 dx = gridSize * viewMatrix * glm::vec3{1, 0, 0};
  glm::vec3 dy = gridSize * viewMatrix * glm::vec3{0, 1, 0};
  glm::vec3 base = viewMatrix * glm::vec3{0, 0, 1};
  return {Ray{position, base}, dx, dy};
}
