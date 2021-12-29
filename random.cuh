#ifndef RENDERER_RANDOM_CUH
#define RENDERER_RANDOM_CUH

#include <cstdint>

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__host__ __device__ uint32_t pcg_hash(uint32_t input);
__host__ __device__ uint32_t rand_pcg(uint32_t& state);
__host__ __device__ float rand_uniform(uint32_t& state);

#endif