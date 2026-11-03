#include <stdint.h>

#include "ca_kernels.h"

/*
 * Small target-side test for the formal Spike extension and X1 Conv2D path.
 * The original input offset is deliberately nonzero; X1 receives the
 * equivalent adjusted bias and an input offset of zero.
 */
static const int8_t input[16] __attribute__((aligned(8))) = {
  -8, 7, -6, 5, -4, 3, -2, 1,
  1, 2, 3, 4, 5, 6, 7, 8,
};

static const int8_t filter[32] __attribute__((aligned(8))) = {
  1, -2, 3, -4, 5, -6, 7, -8,
  -3, 1, 4, -1, 5, -9, 2, 6,
  8, 7, 6, 5, 4, 3, 2, 1,
  -1, -1, -1, -1, 1, 1, 1, 1,
};

static const int32_t original_bias[4] = {31, -47, 13, 101};
static const int32_t multiplier[4] = {
  1073741824, 1073741824, 1073741824, 1073741824,
};
static const int32_t shift[4] = {1, 1, 1, 1};

static const Conv2DParams x1_params = {
  1, 1, 2, 8,
  1, 1, 8,
  1, 2, 4,
  1, 1, 1, 1,
  0, 0,
  0, 0, -128, 127,
};

static int32_t clamp(int32_t value) {
  if (value < -128) return -128;
  if (value > 127) return 127;
  return value;
}

int main(void) {
  const int32_t original_input_offset = 11;
  int32_t adjusted_bias[4];
  int8_t expected[8];
  int8_t actual[8];
  int8_t unaligned_storage[17] __attribute__((aligned(8)));
  int8_t unaligned_actual[8];

  for (int output_c = 0; output_c < 4; ++output_c) {
    int32_t weight_sum = 0;
    for (int input_c = 0; input_c < 8; ++input_c) {
      weight_sum += filter[output_c * 8 + input_c];
    }
    adjusted_bias[output_c] = original_bias[output_c]
                              + original_input_offset * weight_sum;
  }

  for (int output_x = 0; output_x < 2; ++output_x) {
    for (int output_c = 0; output_c < 4; ++output_c) {
      int32_t acc = original_bias[output_c];
      for (int input_c = 0; input_c < 8; ++input_c) {
        acc += filter[output_c * 8 + input_c]
               * (input[output_x * 8 + input_c] + original_input_offset);
      }
      acc = output_scaling_round2(acc, multiplier[output_c], shift[output_c]);
      expected[output_x * 4 + output_c] = (int8_t)clamp(acc);
    }
  }

  conv2d_s8(input, filter, adjusted_bias, multiplier, shift,
            actual, &x1_params);

  /* An unaligned input must safely select the scalar fallback. */
  for (int i = 0; i < 16; ++i) unaligned_storage[i + 1] = input[i];
  conv2d_s8(unaligned_storage + 1, filter, adjusted_bias, multiplier, shift,
            unaligned_actual, &x1_params);

  for (int i = 0; i < 8; ++i) {
    if (actual[i] != expected[i] || unaligned_actual[i] != expected[i]) {
      return i + 1;
    }
  }
  return 0;
}
