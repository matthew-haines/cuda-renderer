#include "random.cuh"

__host__ __device__ uint32_t pcg_hash(uint32_t input) {
  uint32_t state = input * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__host__ __device__ uint32_t rand_pcg(uint32_t& state) {
  uint32_t s = state;
  state = s * 747796405u + 2891336453u;
  uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

// Probably fast enough
__host__ __device__ float rand_uniform(uint32_t& state) {
  return static_cast<float>(rand_pcg(state)) /
         static_cast<float>(std::numeric_limits<uint32_t>::max());
}