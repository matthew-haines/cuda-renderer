#include "gl_manager.h"

#include <stdexcept>

GLManager::GLManager(int width, int height) : width{width}, height{height} {
  if (!glfwInit())
    throw std::runtime_error{"GLFW init failed"};
  window = glfwCreateWindow(width, height, "Renderer", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    throw std::runtime_error{"GLFW window creation failed"};
  }
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK)
    throw std::runtime_error{"GLEW init failed"};
}

GLManager::~GLManager() {
  glfwTerminate();
}