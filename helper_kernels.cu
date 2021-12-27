#include "helper_kernels.cuh"
#include "helper_math.h"

__host__ __device__ uchar4 HSLToRGB(float4 pixel) {
  auto [h, s, l, _] = pixel;
  auto c = (1 - abs(2 * l - 1)) * s;
  auto x = c * (1 - abs(fmod(h / 60, (float) 2) - 1));
  auto m = l - c / 2;

  float4 rgbf;
  if (h < 60) rgbf = {c, x, 0, 0};
  else if (h < 120)
    rgbf = {x, c, 0, 0};
  else if (h < 180)
    rgbf = {0, c, x, 0};
  else if (h < 240)
    rgbf = {0, x, c, 0};
  else if (h < 300)
    rgbf = {x, 0, c, 0};
  else
    rgbf = {c, 0, x, 0};

  auto rgbff = (rgbf + float4{m, m, m, 0}) * 255;
  return make_uchar4(rgbff.x, rgbff.y, rgbff.z, 0);
}
