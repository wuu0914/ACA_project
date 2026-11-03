#ifndef B3_REQUANTIZE_H_
#define B3_REQUANTIZE_H_

#include <stdint.h>

/*
 * B3 negative-shift signed-INT8 requantization
 *
 * The generator validates this contract once for every Conv2D and
 * DepthwiseConv2D operator:
 *   multiplier > 0
 *   -31 <= shift < 0
 *   activation range = [-128, 127]
 *
 * TFLite still performs two separate rounding steps.  B3 only rewrites them
 * into an equivalent form that GCC can inline without general-case branches.
 */
#if defined(B3_REQUANTIZE_NOINLINE)
/* B3 experiment: isolate output scaling from the kernel register allocator. */
#define B3_REQUANTIZE_FUNCTION static __attribute__((noinline))
#else
#define B3_REQUANTIZE_FUNCTION static inline
#endif

B3_REQUANTIZE_FUNCTION int8_t b3_requantize_s8(int32_t acc,
                                               int32_t multiplier,
                                               int shift,
                                               int32_t output_offset) {
  const int right_shift = -shift;

  /* Round((acc * multiplier) / 2^31), with ties toward positive infinity. */
  const int64_t product = (int64_t)acc * multiplier;
  const int32_t q31 = (int32_t)((product + (INT64_C(1) << 30)) >> 31);

  /* Round(q31 / 2^right_shift), with ties away from zero. */
  const int64_t half = INT64_C(1) << (right_shift - 1);
  const int32_t scaled = (int32_t)(((int64_t)q31 + half - (q31 < 0)) >> right_shift);

  int32_t output_value = scaled + output_offset;
  if (output_value < -128) output_value = -128;
  if (output_value > 127) output_value = 127;
  return (int8_t)output_value;
}

#undef B3_REQUANTIZE_FUNCTION

#endif
