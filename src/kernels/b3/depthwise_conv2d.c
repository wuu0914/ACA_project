#include "ca_kernels.h"

#include <stdint.h>

#include "requantize.h"

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

/*
 * B3 implementation
 *
 * As with Conv2D, the generator validates the restricted requantization
 * contract once.  The B2 spatial and four-channel dataflow below is unchanged.
 */
static inline int8_t requantize(int32_t acc, int output_c,
                                const int32_t *output_multipliers,
                                const int32_t *output_shifts,
                                const DepthwiseConv2DParams *p) {
  return b3_requantize_s8(acc,
                          output_multipliers[output_c],
                          output_shifts[output_c],
                          p->output_offset);
}

/*
 * B2 DepthwiseConv2D fast path
 *
 * Layout:
 *   input  = NHWC: [batch, input_y, input_x, channel]
 *   filter = 1HWO: [1, filter_y, filter_x, channel]
 *   output = NHWC: [batch, output_y, output_x, channel]
 *
 * Every output pixel uses the same valid filter rectangle for all channels.
 * B2 calculates that rectangle before the channel loop, so padding checks are
 * not repeated for every channel and never appear in the MAC loop.
 *
 * Four independent accumulators process four adjacent channels.  They share
 * the spatial loops, two walking pointers and one inner-loop branch, but each
 * channel still loads and multiplies its own input/filter pair.
 */
static void depthwise_conv2d_4_s8(
    const int8_t *input,
    const int8_t *filter,
    const int32_t *bias,
    const int32_t *output_multipliers,
    const int32_t *output_shifts,
    int8_t *output,
    const DepthwiseConv2DParams *p) {
  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_y = 0; output_y < p->output_H; ++output_y) {
      const int input_y_origin = output_y * p->stride_H - p->pad_H;

      /* Dilation is one on this path, so clipping the filter index gives the
       * complete set of image rows that contribute to this output row. */
      const int filter_y_start = clamp(-input_y_origin, 0, p->filter_H);
      const int filter_y_end = clamp(p->input_H - input_y_origin, 0, p->filter_H);

      for (int output_x = 0; output_x < p->output_W; ++output_x) {
        const int input_x_origin = output_x * p->stride_W - p->pad_W;

        /* The horizontal range is also shared by every channel tile. */
        const int filter_x_start = clamp(-input_x_origin, 0, p->filter_W);
        const int filter_x_end = clamp(p->input_W - input_x_origin, 0, p->filter_W);
        const int valid_filter_width = filter_x_end - filter_x_start;
        const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) * p->output_C;

        for (int output_c = 0; output_c < p->output_C; output_c += 4) {
          int32_t acc0 = 0;
          int32_t acc1 = 0;
          int32_t acc2 = 0;
          int32_t acc3 = 0;

          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
            const int input_y = input_y_origin + filter_y;
            const int input_index = ((batch * p->input_H + input_y) * p->input_W
                                     + input_x_origin + filter_x_start) * p->input_C + output_c;
            const int filter_index = (filter_y * p->filter_W + filter_x_start) * p->output_C + output_c;

            const int8_t *input_pointer = input + input_index;
            const int8_t *filter_pointer = filter + filter_index;
            const int8_t *filter_end = filter_pointer + valid_filter_width * p->output_C;

            /* B2 pointer walking: moving to the next spatial tap skips one
             * complete NHWC/1HWO channel group. */
            while (filter_pointer != filter_end) {
              acc0 += (int32_t)filter_pointer[0] * ((int32_t)input_pointer[0] + p->input_offset);
              acc1 += (int32_t)filter_pointer[1] * ((int32_t)input_pointer[1] + p->input_offset);
              acc2 += (int32_t)filter_pointer[2] * ((int32_t)input_pointer[2] + p->input_offset);
              acc3 += (int32_t)filter_pointer[3] * ((int32_t)input_pointer[3] + p->input_offset);

              input_pointer += p->input_C;
              filter_pointer += p->output_C;
            }
          }

          if (bias != 0) {
            acc0 += bias[output_c + 0];
            acc1 += bias[output_c + 1];
            acc2 += bias[output_c + 2];
            acc3 += bias[output_c + 3];
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
 * Depth multiplier, dilation and channel-tail cases keep the original dataflow.
 * This path also documents the general mapping from one input channel to one
 * or more output channels.
 */
static void depthwise_conv2d_scalar_s8(
    const int8_t *input,
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
              if (input_y < 0 || input_y >= p->input_H) continue;

              for (int filter_x = 0; filter_x < p->filter_W; ++filter_x) {
                const int input_x = input_x_origin + filter_x * p->dilation_W;
                if (input_x < 0 || input_x >= p->input_W) continue;
                const int input_index = ((batch * p->input_H + input_y) * p->input_W + input_x) * p->input_C + input_c;
                const int filter_index = (filter_y * p->filter_W + filter_x) * p->output_C + output_c;

                acc += (int32_t)filter[filter_index] * ((int32_t)input[input_index] + p->input_offset);
              }
            }

            if (bias != 0) acc += bias[output_c];

            const int output_index = ((batch * p->output_H + output_y) * p->output_W + output_x) * p->output_C + output_c;
            output[output_index] = requantize(acc, output_c, output_multipliers, output_shifts, p);
          }
        }
      }
    }
  }
}

void depthwise_conv2d_s8(const int8_t *input,
                         const int8_t *filter,
                         const int32_t *bias,
                         const int32_t *output_multipliers,
                         const int32_t *output_shifts,
                         int8_t *output,
                         const DepthwiseConv2DParams *p) {
  /* B2 implementation: every model DepthwiseConv2D satisfies this contract. */
  const int use_b2 = p->depth_multiplier == 1
                     && p->dilation_H == 1
                     && p->dilation_W == 1
                     && p->output_C == p->input_C
                     && (p->output_C & 3) == 0;

  if (use_b2) {
    depthwise_conv2d_4_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
  } else {
    depthwise_conv2d_scalar_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
  }
}

