#include "cuda_helpers.h"

#include <stdexcept>

void cudaCheckImpl(cudaError err, int lineNumber, std::string_view fileName) {
  if (err != cudaSuccess) {
    std::string errorMessage{
      "CUDA Error " + std::to_string(err) + " " + cudaGetErrorString(err) +
      "\nLocation: " + std::string{fileName} + ":" + std::to_string(lineNumber)};
    throw std::runtime_error{errorMessage};
  }
}

void fullscreenTexture(GLuint tex) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  {
    glTexCoord2f(0, 1);
    glVertex3f(0, 0, 0);

    glTexCoord2f(0, 0);
    glVertex3f(0, 1, 0);

    glTexCoord2f(1, 0);
    glVertex3f(1, 1, 0);

    glTexCoord2f(1, 1);
    glVertex3f(1, 0, 0);
  }
  glEnd();
}
