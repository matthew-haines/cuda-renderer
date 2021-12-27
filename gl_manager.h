#ifndef RENDERER_GL_MANAGER_H
#define RENDERER_GL_MANAGER_H

#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class GLManager {
public:
  GLManager(int width, int height);
  ~GLManager();
  template <typename Callback, typename Duration>
  void renderLoop(Callback callback, Duration duration);

private:
  int width;
  int height;
  GLFWwindow* window;
};

template <typename Callback, typename Duration>
void GLManager::renderLoop(Callback callback, Duration duration) {
  do {
    auto waitUntil = std::chrono::steady_clock::now() + duration;

    glClear(GL_COLOR_BUFFER_BIT);

    callback(window);

    glfwSwapBuffers(window);
    glfwPollEvents();

    std::this_thread::sleep_until(waitUntil);
  } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));
}

#endif//RENDERER_GL_MANAGER_H
