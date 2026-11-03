#include "ca_kernels.h"

#include <limits.h>
#include <stdint.h>

/* 
 * Math:
 *    Round((a * b) / 2^31)
 * 
 * Special case explanation, ex: 
 *    When both a & b are Q0,31, and a * b = (-1) * (-1), the binary result of (a * b) / 2^31 is 2^31, 
 *    which is overflowed.
 * 
 * About rounding - Round half up (a.k.a. Round to nearest, ties toward +∞):
 *    round(1.5) = 2
 *    round(1.499999) = 1
 *    round(-1.5) = -1
 *    round(-1.500001) = -2
 */
static int32_t ab_fxp_mul_round(int32_t a, int32_t b) {
  // Special case
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

// Multiply by 2^shift and clamp the result to the int32_t range.
static int32_t left_shift_saturating(int32_t value, int shift) {
  const int64_t shifted = (int64_t)value << shift;
  if (shifted > INT32_MAX) return INT32_MAX;
  if (shifted < INT32_MIN) return INT32_MIN;
  return (int32_t)shifted;
}

/*
 * Math:
 *   result = round((a + b) / 2)
 * 
 * About rounding - Round to nearest, ties away from zero 
 */
static int32_t ab_avg_round(int32_t a, int32_t b) {
  const int64_t sum = (int64_t)a + b;
  return (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2);
}

/*
 * Approximate exp(a) for -1/4 <= a < 0. 
 *   input and output: Q0.31
 *
 * Math:
 *    x = a + 1/8 (Move the Taylor-expansion center from 0 to -1/8)
 *    exp(a) = exp(-1/8) * exp(x)
 *    exp(x) ~= 1 + x + x^2/2 + x^3/6 + x^4/24 (Use the fourth-order polynomial)
 */
static int32_t exp_approx_neg1_8(int32_t a) {
  const int32_t constant_term = 1895147668; // exp(-1/8) in Q0.31
  const int32_t one_third = 715827883;
  const int32_t x = a + (1 << 28);
  const int32_t x2 = ab_fxp_mul_round(x, x);
  const int32_t x3 = ab_fxp_mul_round(x2, x);
  const int32_t x4 = ab_fxp_mul_round(x2, x2);
  const int32_t x4_over_4 = right_shift_round(x4, 2);
  const int32_t higher_terms = right_shift_round(ab_fxp_mul_round(x4_over_4 + x3, one_third) + x2, 1);
  return constant_term + ab_fxp_mul_round(constant_term, x + higher_terms);
}

/*
 * Approximate exp(a) for a <= 0.
 *   input:  a in Q5.26
 *   output: exp(a) in Q0.31
 *
 * Math:
 *   Split a into: 
 *   a = a_mod_quarter - remainder, where a_mod_quarter is in [-1/4, 0). 
 *   First evaluate that small interval with exp_approx_neg1_8(), 
 *   then reconstruct exp(-remainder) from the precomputed exp(-1/4), exp(-1/2), ..., exp(-16).
 */
static int32_t exp_on_negative_values(int32_t a_q5_26) {
  // Isolate the part in [-1/4, 0)
  const int32_t one_quarter = 1 << 24;
  const int32_t a_mod_quarter = (a_q5_26 & (one_quarter - 1)) - one_quarter;   // a_mod_quarter is in [-1/4, 0).
  const int32_t remainder = a_mod_quarter - a_q5_26;
  int32_t result = exp_approx_neg1_8(left_shift_saturating(a_mod_quarter, 5)); // Q5.26 -> Q0.31, get exp(a_mod_quarter)

  // Precomputed exp(-1/4), exp(-1/2), ..., exp(-16) in Q0.31.
  static const struct {
    int exponent;
    int32_t multiplier;
  } factors[] = {
      {-2, 1672461947}, {-1, 1302514674}, {0, 790015084},
      {1, 290630308},   {2, 39332535},    {3, 720401}, {4, 242},
  };

  // exp(a) = exp(a_mod_quarter) * exp(-remainder).
  for (unsigned i = 0; i < sizeof(factors) / sizeof(factors[0]); ++i) {
    if ((uint32_t)remainder & (UINT32_C(1) << (26 + factors[i].exponent))) {  // Whether a bit in Q5,26 is 1
      result = ab_fxp_mul_round(result, factors[i].multiplier);
    }
  }

  // Q0.31 cannot encode +1 exactly, so exp(0) saturates at INT32_MAX.
  return a_q5_26 == 0 ? INT32_MAX : result;
}

/*
 * Approximate 1 / (1 + a), where a is Q0.31 and 0 <= a < 1.
 *
 * Math:
 *   Let d = (1 + a) / 2 in Q0.31,
 *   So 1 / (1 + a) = (1 / d) / 2
 *   To estimate x = 1 / d (x is Q2.29), first use a linear approximation:
 *   x = 48/17 - (32/17) * d      
 *   And refine x ~= 1/d three times using Newton-Raphson:
 *   x_next = x + x * (1 - d * x)
 *   Finally convert x / 2 = 1 / (1 + a) from Q2.29 to Q0.31.
 */
static int32_t one_over_one_plus_x(int32_t a) {
  const int32_t constant_48_over_17 = 1515870810;       // Q2.29
  const int32_t constant_neg_32_over_17 = -1010580540;  // Q2.29

  const int32_t d = ab_avg_round(a, INT32_MAX);         // Q0.31
  int32_t x = constant_48_over_17 + ab_fxp_mul_round(d, constant_neg_32_over_17); // x is Q2.29

  // Newton-Raphson: Three iterations reproduce gemmlowp's reciprocal approximation.
  for (int i = 0; i < 3; ++i) {
    const int32_t d_times_x = ab_fxp_mul_round(d, x);   // Q2.29
    const int32_t one_minus_product = (1 << 29) - d_times_x;
    const int32_t correction = left_shift_saturating(ab_fxp_mul_round(x, one_minus_product), 2);  // Q2.29
    x += correction;
  }
  return left_shift_saturating(x, 1);   // Q2.29 -> Q0.31 & (x / 2) at the same time
}

// Count leading zeros of 32 bits.
static int count_leading_zeros(uint32_t value) {
  if (value == 0) return 32;

  int count = 0;
  while ((value & (UINT32_C(1) << 31)) == 0) {
    value <<= 1;
    ++count;
  }
  return count;
}

/*
 * Convert the positive Q12.19 exponent sum into a reciprocal representation in Q0.31.
 *
 * Math:
 *   sum = normalized_sum * 2^num_bits_over_unit,  1 <= normalized_sum < 2
 *   a = normalized_sum - 1, 0 <= a < 1
 *   shifted_sum_minus_one stores a in Q0.31
 *   1 / (1 + a) = 1 / normalized_sum
 *   1 / sum = (1 / normalized_sum) * 2^(-num_bits_over_unit)
 * 
 * one_over_one_plus_x() returns 1 / normalized_sum. The separate exponent is
 * applied later through output_shift.
 */
static int32_t reciprocal(int32_t x, 
                          int *num_bits_over_unit) {
  const int headroom_plus_one = count_leading_zeros((uint32_t)x);
  *num_bits_over_unit = 12 - headroom_plus_one;
  const int32_t shifted_sum_minus_one = (int32_t)(((uint32_t)x << headroom_plus_one) - (UINT32_C(1) << 31));
  return one_over_one_plus_x(shifted_sum_minus_one);    // Q0.31, not times 2^(-num_bits_over_unit) yet
}

// Layout:
//   input_diff is integer
//   left_shift is integer
//   multiplier is Q0.31
//   scaled_diff is Q5.26 
//
// Math:
//   scaled_diff = round(input_diff * 2^26 * input_scale)
// In TFLite's PreprocessSoftmaxScaling(), it expects input_scale * 2^26 = multiplier * 2^left_shift.
static int32_t input_scaling(int32_t input_diff, 
                             int32_t multiplier,
                             int left_shift) {
  return ab_fxp_mul_round(input_diff * (1 << left_shift), multiplier);
}


/*
 * Layout:
 *   input  = [outer_size, channel]
 *   output = [outer_size, channel]
 *
 * Math for one output element:
 *   input_diff = input_q[outer_size][channel] - max(input_q[outer_size])
 *   probability = exp(input_diff * input_scale) / sum(exp(input_diff * input_scale))
 *   output_q = clamp(round(probability * 256) - 128, -128, 127)
 *
 * Notes:
 *   1. TFLite changes fixed-point formats by stage: scaled input difference is
 *      Q5.26, exp/reciprocal are Q0.31, and the exponent sum is Q12.19.
 *   2. exp(a): Split a into one part for LUT and the other for Taylor expansion.
 *   3. reciprocal: 
 *      normalize 1 / sum into 1 / (1.xxx * 2^N)
 *      -> approximate 1 / (1 + a) by transform d = (1 + a) / 2 and 
 *         find 1 / d by linear approximation + Newton-Raphson.
 *   4. Quantize output to int8_t.
 */
void softmax_s8(const int8_t *input, 
                int8_t *output, 
                int outer_size,
                int depth, 
                int input_multiplier, 
                int input_left_shift,
                int diff_min) {
  for (int row = 0; row < outer_size; ++row) {
    // Step 1: find the row maximum. 
    int8_t max_input = INT8_MIN;
    for (int channel = 0; channel < depth; ++channel) {
      const int input_index = row * depth + channel;
      if (input[input_index] > max_input) max_input = input[input_index];
    }

    // Step 2: compute sum(exp(input_diff * input_scale)).
    int32_t sum_of_exps = 0;
    for (int channel = 0; channel < depth; ++channel) {
      const int input_index = row * depth + channel;
      const int32_t input_diff = (int32_t)input[input_index] - max_input;

      // TFLite skips differences whose exponential is too small to affect an 8-bit result.
      if (input_diff >= diff_min) {
        // INT8 input_diff -> Q5.26 scaled_diff -> Q0.31 exp -> Q12.19 accumulator. 
        const int32_t scaled_diff = input_scaling(input_diff, input_multiplier, input_left_shift);  // scaled_diff is Q5.26
        sum_of_exps += right_shift_round(exp_on_negative_values(scaled_diff), 12);
      }
    }

    // Step 3: normalize sum_of_exps and approximate its reciprocal. 
    int num_bits_over_unit;   // carries the power-of-two removed by normalization
    const int32_t shifted_scale = reciprocal(sum_of_exps, &num_bits_over_unit);   // Q0.31

    // right shift 31 converts the Q0.31 product back to an integer 
    // left shift 8 preserves the eight probability bits needed by INT8 softmax
    // num_bits_over_unit restores the normalization exponent
    // => output_shift = num_bits + 31 - 8
    const int output_shift = num_bits_over_unit + 23;

    // Step 4: recompute each exponential and multiply it by the reciprocal.
    for (int channel = 0; channel < depth; ++channel) {
      const int input_index = row * depth + channel;
      const int32_t input_diff = (int32_t)input[input_index] - max_input;
      int32_t shifted_output = INT8_MIN;
      if (input_diff >= diff_min) {
        const int32_t scaled_diff = input_scaling(input_diff, input_multiplier, input_left_shift);
        const int32_t probability = right_shift_round(ab_fxp_mul_round(shifted_scale, exp_on_negative_values(scaled_diff)), output_shift);

        // probability is now round(real_probability * 256). 
        // INT8 softmax has scale = 1/256 and zero_point = -128
        // so add -128 before clamping.
        shifted_output = probability + INT8_MIN;
        if (shifted_output > INT8_MAX) shifted_output = INT8_MAX;
        if (shifted_output < INT8_MIN) shifted_output = INT8_MIN;
      }
      output[input_index] = (int8_t)shifted_output; // INT8 probability
    }
  }
}
