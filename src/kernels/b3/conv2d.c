#include "ca_kernels.h"

#include <stdint.h>

#include "requantize.h"
#include "xqmac8.h"

/*
 * B3 implementation
 *
 * The generator already verified this model uses positive multipliers,
 * negative shifts and the complete signed-INT8 activation range.  Therefore
 * Conv2D can use the smaller B3 requantization directly at every output.
 */
static inline int8_t requantize(int32_t acc, int output_c,
                                const int32_t *output_multipliers,
                                const int32_t *output_shifts,
                                const Conv2DParams *p) {
  return b3_requantize_s8(acc,
                          output_multipliers[output_c],
                          output_shifts[output_c],
                          p->output_offset);
}

/*
 * X1 1x1 fast path
 *
 * Generated B1/X1 data already contains:
 *   adjusted_bias[c] = bias[c] + input_offset * sum(filter[c])
 *   input_offset     = 0
 *
 * Four output channels reuse one packed input.  Each Xqmac8 consumes eight
 * signed input lanes, eight signed weight lanes and one INT32 accumulator.
 * EfficientNet-Lite0 has no input-channel or output-channel tail in this path.
 */
static void conv2d_1x1_xqmac8_s8(const int8_t *input,
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

        for (int output_c = 0; output_c < p->output_C; output_c += 4) {
          const int8_t *weight0 = filter + (output_c + 0) * p->filter_input_C;
          const int8_t *weight1 = filter + (output_c + 1) * p->filter_input_C;
          const int8_t *weight2 = filter + (output_c + 2) * p->filter_input_C;
          const int8_t *weight3 = filter + (output_c + 3) * p->filter_input_C;
          const int8_t *input_pointer = input_pixel;
          const int8_t *input_end = input_pixel + p->input_C;
          /* Keep RV64 register values sign-extended between Xqmac8 calls. */
          int64_t acc0 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 0];
          int64_t acc1 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 1];
          int64_t acc2 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 2];
          int64_t acc3 = adjusted_bias == 0 ? 0 : adjusted_bias[output_c + 3];

          /* X1 implementation: 8 input channels become four packed MACs. */
          while (input_pointer != input_end) {
            const uint64_t packed_input = xqmac8_load_aligned(input_pointer);
            acc0 = xqmac8(acc0, packed_input, xqmac8_load_aligned(weight0));
            acc1 = xqmac8(acc1, packed_input, xqmac8_load_aligned(weight1));
            acc2 = xqmac8(acc2, packed_input, xqmac8_load_aligned(weight2));
            acc3 = xqmac8(acc3, packed_input, xqmac8_load_aligned(weight3));
            input_pointer += 8;
            weight0 += 8;
            weight1 += 8;
            weight2 += 8;
            weight3 += 8;
          }

          output[output_index + output_c + 0] = requantize((int32_t)acc0, output_c + 0, output_multipliers, output_shifts, p);
          output[output_index + output_c + 1] = requantize((int32_t)acc1, output_c + 1, output_multipliers, output_shifts, p);
          output[output_index + output_c + 2] = requantize((int32_t)acc2, output_c + 2, output_multipliers, output_shifts, p);
          output[output_index + output_c + 3] = requantize((int32_t)acc3, output_c + 3, output_multipliers, output_shifts, p);
        }
      }
    }
  }
}

/* B0 scalar fallback for the first 3x3 Conv and any non-X1 input. */
static void conv2d_scalar_s8(const int8_t *input,
                             const int8_t *filter,
                             const int32_t *bias,
                             const int32_t *output_multipliers,
                             const int32_t *output_shifts,
                             int8_t *output,
                             const Conv2DParams *p) {
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
                acc += (int32_t)filter[filter_index] * ((int32_t)input[input_index] + p->input_offset);
              }
            }
          }

          if (bias != 0) acc += bias[output_c];
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
  /*
   * X1 implementation: alignment is checked once here.  An unaligned test
   * pointer and every unsupported shape safely use the scalar path.
   */
  const int use_x1 = p->filter_H == 1
                     && p->filter_W == 1
                     && p->input_C == p->filter_input_C
                     && p->pad_H == 0
                     && p->pad_W == 0
                     && p->input_offset == 0
                     && ((p->input_C >> 3) << 3) == p->input_C
                     && ((p->output_C >> 2) << 2) == p->output_C
                     && (((uintptr_t)input & 7u) == 0)
                     && (((uintptr_t)filter & 7u) == 0);

  if (use_x1) {
    conv2d_1x1_xqmac8_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
    return;
  }

  conv2d_scalar_s8(input, filter, bias, output_multipliers, output_shifts, output, p);
}
