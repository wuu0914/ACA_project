#include "ca_kernels.h"

#include <stdint.h>

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

static int8_t requantize(int32_t acc, int output_c,
                         const int32_t *output_multipliers,
                         const int32_t *output_shifts,
                         const Conv2DParams *p) {
  acc = output_scaling_round2(acc, output_multipliers[output_c], output_shifts[output_c]);
  acc = clamp(acc + p->output_offset, p->activation_min, p->activation_max);
  return (int8_t)acc;
}

/*
 * B1 1x1 fast path
 *
 * Layout:
 *   input  = NHWC: [batch, input_y, input_x, input_channel]
 *   filter = OHWI: [output_channel, 1, 1, input_channel]
 *   output = NHWC: [batch, output_y, output_x, output_channel]
 *
 * Generated B1 data already contains:
 *   adjusted_bias[c] = bias[c] + input_offset * sum(filter[c])
 *   input_offset     = 0
 *
 * Four output channels reuse the same input value.  Each output channel still
 * owns an independent weight pointer and accumulator.
 */
static void conv2d_1x1_s8(const int8_t *input,
                          const int8_t *filter,
                          const int32_t *adjusted_bias,
                          const int32_t *output_multipliers,
                          const int32_t *output_shifts,
                          int8_t *output,
                          const Conv2DParams *p) {
  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_y = 0; output_y < p->output_H; ++output_y) {
      const int input_y = output_y * p->stride_H;

      for (int output_x = 0; output_x < p->output_W; ++output_x) {
        const int input_x = output_x * p->stride_W;
        const int input_index = ((batch * p->input_H + input_y) * p->input_W + input_x) * p->input_C;
        const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) * p->output_C;
        const int8_t *input_pixel = input + input_index;

        // Four output channels share one input load at each input channel.
        for (int output_c = 0; output_c < p->output_C; output_c += 4) {
          const int8_t *weight0 = filter + (output_c + 0) * p->filter_input_C;
          const int8_t *weight1 = filter + (output_c + 1) * p->filter_input_C;
          const int8_t *weight2 = filter + (output_c + 2) * p->filter_input_C;
          const int8_t *weight3 = filter + (output_c + 3) * p->filter_input_C;
          const int8_t *input_pointer = input_pixel;
          const int8_t *input_end = input_pixel + p->input_C;
          int32_t acc0 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 0];
          int32_t acc1 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 1];
          int32_t acc2 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 2];
          int32_t acc3 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 3];

          while (input_pointer != input_end) {
            const int32_t input_value = *input_pointer++;
            acc0 += (int32_t)*weight0++ * input_value;
            acc1 += (int32_t)*weight1++ * input_value;
            acc2 += (int32_t)*weight2++ * input_value;
            acc3 += (int32_t)*weight3++ * input_value;
          }

          output[output_index + output_c + 0] = requantize(acc0, output_c + 0, output_multipliers, output_shifts, p);
          output[output_index + output_c + 1] = requantize(acc1, output_c + 1, output_multipliers, output_shifts, p);
          output[output_index + output_c + 2] = requantize(acc2, output_c + 2, output_multipliers, output_shifts, p);
          output[output_index + output_c + 3] = requantize(acc3, output_c + 3, output_multipliers, output_shifts, p);
        }
      }
    }
  }
}

/*
 * B0 scalar fallback
 *
 * The first 3x3 Conv and any Conv that does not satisfy the B1 contract keep
 * the original dataflow, which is identical to B0.
 */
static void conv2d_scalar_s8(const int8_t *input, 
               const int8_t *filter,
               const int32_t *bias, 
               const int32_t *output_multipliers,
               const int32_t *output_shifts, 
               int8_t *output,
               const Conv2DParams *p) {
  // Note:
  //    groups = 1 for ordinary convolution.  
  //    A larger value divides input and output channels into independent groups.
  const int groups = p->input_C / p->filter_input_C;
  const int output_C_per_group = p->output_C / groups;

  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_y = 0; output_y < p->output_H; ++output_y) {
      const int input_y_origin = output_y * p->stride_H - p->pad_H;
      for (int output_x = 0; output_x < p->output_W; ++output_x) {
        const int input_x_origin = output_x * p->stride_W - p->pad_W;
        for (int output_c = 0; output_c < p->output_C; ++output_c) {
          const int group = output_c / output_C_per_group;
          int32_t acc = 0;

          for (int filter_y = 0; filter_y < p->filter_H; ++filter_y) {
            const int input_y = input_y_origin + filter_y * p->dilation_H;
            // SAME padding: pad input zero point.
            // Its centered quantized value (input_q - input_zp) is zero, 
            // so skip this out-of-image multiplication.
            if (input_y < 0 || input_y >= p->input_H) continue;

            for (int filter_x = 0; filter_x < p->filter_W; ++filter_x) {
              const int input_x = input_x_origin + filter_x * p->dilation_W;
              if (input_x < 0 || input_x >= p->input_W) continue;

              for (int filter_input_c = 0; filter_input_c < p->filter_input_C; ++filter_input_c) {
                const int input_c = group * p->filter_input_C + filter_input_c;
                const int input_index = ((batch * p->input_H + input_y) * p->input_W + input_x) 
                                        * p->input_C + input_c;
                const int filter_index = ((output_c * p->filter_H + filter_y) * p->filter_W + filter_x)
                                         * p->filter_input_C + filter_input_c;

                // Note: input_offset = -input_zero_point
                acc += (int32_t)filter[filter_index] * ((int32_t)input[input_index] + p->input_offset);
              }
            }
          }

          // bias[output_c] is already quantized in accumulator scale.
          if (bias != 0) acc += bias[output_c];

          // output_multipliers/output_shifts encode
          // input_scale * filter_scale[output_c] / output_scale.
          output[((batch * p->output_H + output_y) * p->output_W + output_x) * p->output_C + output_c] 
                = requantize(acc, output_c, output_multipliers, output_shifts, p);
        }
      }
    }
  }
}

void conv2d_s8(const int8_t *input,
               const int8_t *filter,
               const int32_t *bias,
               const int32_t *output_multipliers,
               const int32_t *output_shifts,
               int8_t *output,
               const Conv2DParams *p) {
  const int use_b1 = p->filter_H == 1
                     && p->filter_W == 1
                     && p->input_C == p->filter_input_C
                     && p->pad_H == 0
                     && p->pad_W == 0
                     && p->input_offset == 0
                     && ((p->output_C >> 2) << 2) == p->output_C;

  if (use_b1) {
    conv2d_1x1_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
    return;
  }

  conv2d_scalar_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
}
