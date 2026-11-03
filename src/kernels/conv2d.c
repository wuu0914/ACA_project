#include "ca_kernels.h"

#include <stdint.h>

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

/*
 * Layout:
 *   input  = NHWC: [batch, input_y, input_x, input_channel]
 *   filter = OHWI: [output_channel, filter_y, filter_x, input_channel]
 *   output = NHWC: [batch, output_y, output_x, output_channel]
 *
 * Math for one output element:
 *   acc = sum(filter_q * (input_q - input_zp)) + bias_q[output_c]
 *   output_multiplier[output_c] = input_scale * filter_scale[output_c] / output_scale
 *   output_q = clamp(round(acc * output_multiplier[output_c]) + output_zero_point,
 *                    activation_min, activation_max)
 *
 * Notes:
 *   1. INT8 filters are symmetric, so filter_zero_point = 0.
 *   2. bias_q is divided by input_scale * filter_scale[output_c],
 *      it can therefore be added before output scaling.
 *   3. multiplier/shift are per output channel because filter_scale is
 *      per output channel.
 */
void conv2d_s8(const int8_t *input, 
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
          acc = output_scaling_round2(acc, output_multipliers[output_c], output_shifts[output_c]);
          acc = clamp(acc + p->output_offset, p->activation_min, p->activation_max);

          const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) *
                                    p->output_C + output_c;
          output[output_index] = (int8_t)acc;
        }
      }
    }
  }
}
