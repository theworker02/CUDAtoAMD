#pragma once

struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  constexpr dim3(unsigned int x_value = 1, unsigned int y_value = 1, unsigned int z_value = 1)
      : x(x_value), y(y_value), z(z_value) {}
};

/* Built-in launch variables require HIP/Clang device-language lowering. */
