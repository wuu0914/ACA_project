#include "ca_kernels.h"

#include <stdint.h>

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

/*
 * Math:
 *   output_q = clamp(round((input_q - input_zp) * input_scale/output_scale) + output_zp)
 */
void quantize_u8_to_s8(const uint8_t *input, 
                       int8_t *output, 
                       int size,
                       int input_zero_point, 
                       int output_zero_point,
                       int multiplier, 
                       int shift) {
  for (int i = 0; i < size; ++i) {
    int32_t value = (int32_t)input[i] - input_zero_point;
    value = output_scaling_round2(value, multiplier, shift);
    output[i] = (int8_t)clamp(value + output_zero_point, -128, 127);
  }
}

// Similar to quantize_u8_to_s8, but for int8 to uint8 conversion.
void quantize_s8_to_u8(const int8_t *input, 
                       uint8_t *output, 
                       int size,
                       int input_zero_point, 
                       int output_zero_point,
                       int multiplier, 
                       int shift) {
  for (int i = 0; i < size; ++i) {
    int32_t value = (int32_t)input[i] - input_zero_point;
    value = output_scaling_round2(value, multiplier, shift);
    output[i] = (uint8_t)clamp(value + output_zero_point, 0, 255);
  }
}
