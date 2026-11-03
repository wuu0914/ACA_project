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
 *   output = NHWC: [batch, output_y, output_x, input_channel]
 *
 * Math for one output element:
 *   output_q = clamp(round(sum(valid input_q) / valid_count), activation_min, activation_max)
 *
 * Notes:
 *   1. clamp is for fused activation（ex: add ReLU/ReLU6 after pooling）
 *   2. -input_y_origin <= filter_y < input_H - input_y_origin &
 *      0 <= filter_y < filter_H
 *      so the valid filter_y range is max(0, -input_y_origin) <= filter_y < min(filter_H, input_H - input_y_origin)
 *   3. count / 2 adjustment is doing round to nearest, ties away from zero
 */
void average_pool2d_s8(const int8_t *input,
                       int8_t *output,
                       const AveragePool2DParams *p) {
  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_y = 0; output_y < p->output_H; ++output_y) {
      const int input_y_origin = output_y * p->stride_H - p->pad_H;
      const int valid_filter_y_start = input_y_origin < 0 ? -input_y_origin : 0;
      const int valid_filter_y_end = p->filter_H < p->input_H - input_y_origin ? p->filter_H : p->input_H - input_y_origin;
      for (int output_x = 0; output_x < p->output_W; ++output_x) {
        const int input_x_origin = output_x * p->stride_W - p->pad_W;
        const int valid_filter_x_start = input_x_origin < 0 ? -input_x_origin : 0;
        const int valid_filter_x_end = p->filter_W < p->input_W - input_x_origin ? p->filter_W : p->input_W - input_x_origin;
        const int valid_count = (valid_filter_y_end - valid_filter_y_start) * (valid_filter_x_end - valid_filter_x_start);

        for (int input_c = 0; input_c < p->input_C; ++input_c) {
          int32_t sum = 0;
          for (int filter_y = valid_filter_y_start; filter_y < valid_filter_y_end; ++filter_y) {
            const int input_y = input_y_origin + filter_y;
            for (int filter_x = valid_filter_x_start; filter_x < valid_filter_x_end; ++filter_x) {
              const int input_x = input_x_origin + filter_x;

              const int input_index = ((batch * p->input_H + input_y) * p->input_W + input_x) * p->input_C + input_c;
              sum += input[input_index];
            }
          }

          // Round to nearest, ties away from zero
          const int32_t average = sum > 0 ? (sum + valid_count / 2) / valid_count : (sum - valid_count / 2) / valid_count;

          const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) * p->input_C + input_c;
          output[output_index] = (int8_t)clamp(average, p->activation_min, p->activation_max);
        }
      }
    }
  }
}
