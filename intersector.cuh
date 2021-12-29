#ifndef RENDERER_INTERSECTOR_CUH
#define RENDERER_INTERSECTOR_CUH

#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/tuple.hpp>
#include <variant>

#include "primitives.cuh"
#include "random.cuh"

template <typename... Types>
class Intersector {
public:
  Intersector() = default;
  Intersector(const Intersector& other) = delete;
  ~Intersector() {
    boost::fusion::for_each(arrays, [&](auto& x) { cudaFree(x.first); });
    boost::fusion::for_each(lights, [&](auto& x) { cudaFree(x.first); });
  }

  template <typename T>
  void addPrimitives(const std::vector<T>& v) {
    setArray(deviceVector(v), v.size(), arrays);
    nPrimitives += v.size();

    std::vector<T> areaLights;
    std::copy_if(v.begin(), v.end(), std::back_inserter(areaLights),
                 [](const T& x) { return x.power > 0.; });
    setArray(deviceVector(areaLights), areaLights.size(), lights);
    nLights += areaLights.size();
  }

  // Maybe make std::optional instead of -1?
  __host__ __device__ Intersection getIntersection(const Ray& ray) const {
    Intersection closest;
    closest.distance = 1e9;
    bool intersected = false;
    boost::fusion::for_each(arrays, [&](auto& arr) {
      for (size_t i = 0; i < arr.second; ++i) {
        auto intersection = arr.first[i].intersect(ray);
        if (intersection.distance >= 0 && intersection.distance < closest.distance) {
          closest = intersection;
          intersected = true;
        }
      }
    });

    if (!intersected) {
      closest.distance = -1;
    }
    return closest;
  }

  // (light, nLights)
  template <typename Visitor>
  __host__ __device__ void sampleLight(uint32_t& rngState, Visitor visitor) const {
    uint32_t i = rand_pcg(rngState) % nLights;
    boost::fusion::for_each(lights, [&](auto& arr) {
      if (i < arr.second)
        visitor(arr.first[i], nLights);
      i -= arr.second; // does cause loop around but shouldn't matter
    });
  }

private:
  template <typename T>
  void setArray(T* arr, size_t n,
                boost::fusion::tuple<std::pair<Types*, size_t>...>& tuple) {
    boost::fusion::for_each(tuple, [&](auto& x) {
      if constexpr (std::is_same_v<std::remove_reference_t<decltype(x)>,
                                   std::pair<T*, size_t>>)
        x = std::make_pair(arr, n);
    });
  }

  // Switch to CUDAUniquePtr later
  boost::fusion::tuple<std::pair<Types*, size_t>...> arrays;
  boost::fusion::tuple<std::pair<Types*, size_t>...> lights;
  size_t nPrimitives = 0;
  size_t nLights = 0;
};

#endif