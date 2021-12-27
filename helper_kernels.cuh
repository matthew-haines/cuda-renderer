#ifndef RENDERER_HELPER_KERNELS_CUH
#define RENDERER_HELPER_KERNELS_CUH

// H: [0, 360), S: [0, 1], L: [0, 1]
__host__ __device__ uchar4 HSLToRGB(float4 pixel);

#endif