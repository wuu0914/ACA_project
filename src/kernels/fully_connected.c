#include "ca_kernels.h"

#include <stdint.h>

static int32_t clamp(int32_t value, int32_t low, int32_t high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

/*
 * Layout:
 *   input   = [batch, input_channel]
 *   weights = [output_channel, input_channel]
 *   output  = [batch, output_channel]
 *
 * Math for one output element:
 *   acc = sum(weight_q * (input_q - input_zp)) + bias_q[output_c]
 *   output_multiplier = input_scale * weight_scale / output_scale
 *   output_q = clamp(round(acc * output_multiplier) + output_zero_point, activation_min, activation_max)
 *
 * Notes:
 *   1. INT8 weights are symmetric, so weight_zero_point = 0.
 *   2. The corresponding TFLite FullyConnected path uses single-rounding.
 */
void fully_connected_s8(const int8_t *input,
                        const int8_t *weights,
                        const int32_t *bias,
                        int output_multiplier,
                        int output_shift,
                        int8_t *output,
                        const FullyConnectedParams *p) {
  for (int batch = 0; batch < p->batches; ++batch) {
    for (int output_c = 0; output_c < p->output_C; ++output_c) {
      int32_t acc = 0;
      for (int input_c = 0; input_c < p->input_C; ++input_c) {
        const int input_index = batch * p->input_C + input_c;
        const int weight_index = output_c * p->input_C + input_c;

        // input_offset = -input_zero_point.
        acc += (int32_t)weights[weight_index] * ((int32_t)input[input_index] + p->input_offset);
      }

      // bias[output_c] is already quantized in accumulator scale.
      if (bias != 0) acc += bias[output_c];

      // Single-rounding
      acc = output_scaling_round1(acc, output_multiplier, output_shift);
      acc = clamp(acc + p->output_offset, p->activation_min, p->activation_max);

      const int output_index = batch * p->output_C + output_c;
      output[output_index] = (int8_t)acc;
    }
  }
}
