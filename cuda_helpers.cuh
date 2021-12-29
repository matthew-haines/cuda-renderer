#ifndef RENDERER_CUDA_HELPERS_CUH
#define RENDERER_CUDA_HELPERS_CUH

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

void cudaCheckImpl(cudaError err, int lineNumber, std::string_view fileName);

#define cudaCheck(err) cudaCheckImpl(err, __LINE__, __FILE__)

void fullscreenTexture(GLuint tex);

template <typename T>
struct FormatSymbol {};

// Doesn't compile with char const[]
template <>
struct FormatSymbol<int> {
  static constexpr char * const symbol = "%d";
};

template <>
struct FormatSymbol<unsigned int> {
  static constexpr char * const symbol = "%u";
};

template <>
struct FormatSymbol<float> {
  static constexpr char * const symbol = "%f";
};

template <>
struct FormatSymbol<double> {
  static constexpr char * const symbol = "%lf";
};

template <int Length, typename T, glm::qualifier Q>
__host__ __device__ void printVector(const glm::vec<Length, T, Q>& v) {
  for (size_t i = 0; i < Length; ++i) {
    std::printf(FormatSymbol<T>::symbol, v[i]);
    std::printf(" ");
  }
  std::printf("\n");
}

template <int width, int height, typename T, typename... Args>
void runGridKernel(T kernel, Args&&... args) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernel<<<numBlocks, threadsPerBlock>>>(std::forward<Args&&>(args)...);
}

template <typename T>
struct CUDADelete {
  void operator()(T* ptr) const { cudaFree(ptr); }
};

template <typename T>
using CUDAUniquePtr = std::unique_ptr<T, CUDADelete<T>>;

template <typename T, typename... Args>
CUDAUniquePtr<T> makeCUDAUnique(Args&&... args) {
  T* ptr;
  cudaMallocManaged(&ptr, sizeof(T));
  new (ptr) T{std::forward<Args&&>(args)...};
  return CUDAUniquePtr<T>{ptr};
}

template <typename T>
T* deviceVector(const std::vector<T>& v) {
  auto bytes = v.size() * sizeof(T);
  T* devicePtr{};
  cudaCheck(cudaMalloc(&devicePtr, bytes));
  cudaCheck(cudaMemcpy(devicePtr, v.data(), bytes, cudaMemcpyHostToDevice));
  return devicePtr;
}

// rotates {0, 0, 1} is rotated to normal
inline __host__ __device__ glm::quat normalQuat(const glm::vec3& normal) {
  return glm::angleAxis(acosf(glm::dot({0, 0, 1}, normal)), glm::cross(normal, {0, 0, 1}));
}

#endif // RENDERER_CUDA_HELPERS_CUH
