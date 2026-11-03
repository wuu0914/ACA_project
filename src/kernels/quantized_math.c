#include "ca_kernels.h"

#include <limits.h>
#include <stdint.h>

/* Shared fixed-point primitives for kernels that need requantization. */
/* Defines how an int32 value is multiplied by a real scale in fixed-point arithmetic, 
   with the result rounded to the nearest. */

/* 
 * Math:
 *    Round((a * b) / 2^31)
 * 
 * Special case explanation: 
 *    When a * b = (-1) * (-1) in Q0,31, the binary result of (a * b) / 2^31 is 2^31, 
 *    which is overflowed.
 * 
 * About rounding - Round half up (a.k.a. Round to nearest, ties toward +∞):
 *    round(1.5) = 2
 *    round(1.499999) = 1
 *    round(-1.5) = -1
 *    round(-1.500001) = -2
 */
static int32_t ab_fxp_mul_round(int32_t a, int32_t b) {
  // Special case: saturation for (-1) * (-1) in Q0,31
  if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX; 

  // Normal case
  const int64_t product = (int64_t)a * (int64_t)b;                                    // a * b
  const int64_t nudge = product >= 0 ? (INT64_C(1) << 30) : 1 - (INT64_C(1) << 30);   // Prepare nudge for rounding
  return (int32_t)((product + nudge) / (INT64_C(1) << 31));
}

/* 
 * Math:
 *    Round(value / 2^exponent)
 * 
 * About rounding - Round to nearest, ties away from zero
 */
static int32_t right_shift_round(int32_t value, int exponent) {
  // Special case: right shift by 0 means no division
  if (exponent == 0) return value;

  // Normal case
  const uint32_t mask = (UINT32_C(1) << exponent) - 1;
  const int32_t remainder = (int32_t)((uint32_t)value & mask);      // remainder: the discarded bits during the right shift
  const int32_t threshold = (int32_t)(mask >> 1) + (value < 0);     // threshold: the value to compare with the remainder for rounding decision
  return (value >> exponent) + (remainder > threshold);
}

/*
 * In TFLite: MultiplyByQuantizedMultiplier
 *
 * Introduction:
 *    TFLite's default double-rounding path. 
 *    Just to do output scaling.
 * 
 * Math:
 *   (Output scale * shifted multiplier) >> shift
 */
int32_t output_scaling_round2(int32_t value,
                              int32_t multiplier,
                              int shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int32_t shifted = value * (1 << left_shift); // Shift value instead of multiplier, since multiplier usually occupies 32 bits.
  return right_shift_round(ab_fxp_mul_round(shifted, multiplier), right_shift);
}

/*
 * In TFLite: MultiplyByQuantizedMultiplier_single_round
 *
 * Introduction:
 *   TFLite's single-rounding path.
 *   Similar to MultiplyByQuantizedMultiplier.
 *   To match different quantized kernel arithmetic.
 */
int32_t output_scaling_round1(int32_t value, 
                              int32_t multiplier, 
                              int shift) {
  const int total_shift = 31 - shift;
  const int64_t round = INT64_C(1) << (total_shift - 1);
  return (int32_t)(((int64_t)value * multiplier + round) >> total_shift);
}
