#ifndef X3_REQUANTIZE_H_
#define X3_REQUANTIZE_H_

#include <stdint.h>

#include "xclip8.h"

/*
 * X3 negative-shift signed-INT8 requantization
 *
 * The two TFLite-compatible rounding steps are unchanged from B3.  X3 only
 * replaces the final two-sided clamp with one Xclip8 instruction.
 */
static inline int8_t x3_requantize_s8(int32_t acc,
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

  /* X3 implementation: add the zero point, then clamp with one instruction. */
  return xclip8(scaled + output_offset);
}

#endif
