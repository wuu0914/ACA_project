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
 *   filter = 1HWO: [1, filter_y, filter_x, output_channel]
 *   output = NHWC: [batch, output_y, output_x, output_channel]
 *
 * Channel mapping:
 *   
 *
 * Math for one output element:
 *   acc = sum(filter_q * (input_q - input_zp)) + bias_q[output_c]
 *   output_multiplier[output_c] =
 *       input_scale * filter_scale[output_c] / output_scale
 *   output_q = clamp(round(acc * output_multiplier[output_c])
 *                    + output_zero_point,
 *                    activation_min, activation_max)
 *
 * Notes:
 *   1. Unlike ordinary convolution, one output channel sums spatial taps from
 *      exactly one input channel.
 *   2. INT8 filters are symmetric, so filter_zero_point = 0.
 *   3. multiplier/shift and bias are indexed by output channel.
 *   4. output channel is related to input channel:
 *      output_C = input_C * depth_multiplier
 */
void depthwise_conv2d_s8(const int8_t *input,
                         const int8_t *filter,
                         const int32_t *bias,
                         const int32_t *output_multipliers,
                         const int32_t *output_shifts,
                         int8_t *output,
                         const DepthwiseConv2DParams *p) {
  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_y = 0; output_y < p->output_H; ++output_y) {
      const int input_y_origin = output_y * p->stride_H - p->pad_H;
      for (int output_x = 0; output_x < p->output_W; ++output_x) {
        const int input_x_origin = output_x * p->stride_W - p->pad_W;
        for (int input_c = 0; input_c < p->input_C; ++input_c) {
          for (int depth_multiplier_index = 0; depth_multiplier_index < p->depth_multiplier; ++depth_multiplier_index) {
            const int output_c = input_c * p->depth_multiplier + depth_multiplier_index;
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

                const int input_index = ((batch * p->input_H + input_y) * p->input_W + input_x) * p->input_C + input_c;
                const int filter_index = (filter_y * p->filter_W + filter_x) * p->output_C + output_c;

                // input_offset = -input_zero_point.
                acc += (int32_t)filter[filter_index] * ((int32_t)input[input_index] + p->input_offset);
              }
            }

            // bias[output_c] is already quantized in accumulator scale.
            if (bias != 0) acc += bias[output_c];

            acc = output_scaling_round2(acc, output_multipliers[output_c], output_shifts[output_c]);
            acc = clamp(acc + p->output_offset, p->activation_min, p->activation_max);

            const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) * p->output_C + output_c;
            output[output_index] = (int8_t)acc;
          }
        }
      }
    }
  }
}
