#ifndef CA_KERNELS_H_
#define CA_KERNELS_H_

#include <stdint.h>

/*
 * Minimal C interface for the Phase 1-3 kernels.
 */

/* CONV_2D uses OHWI filters: [out_channel, height, width, in_channel]. */
typedef struct {
  int batches;
  int input_H;
  int input_W;
  int input_C;
  int filter_H;
  int filter_W;
  int filter_input_C;
  int output_H;
  int output_W;
  int output_C;
  int stride_H;
  int stride_W;
  int dilation_H;
  int dilation_W;
  int pad_H;
  int pad_W;
  int input_offset;
  int output_offset;
  int activation_min;
  int activation_max;
} Conv2DParams;

/* DEPTHWISE_CONV_2D uses filters shaped [1, height, width, out_channel]. */
typedef struct {
  int batches;
  int input_H;
  int input_W;
  int input_C;
  int filter_H;
  int filter_W;
  int output_H;
  int output_W;
  int output_C;
  int depth_multiplier;
  int stride_H;
  int stride_W;
  int dilation_H;
  int dilation_W;
  int pad_H;
  int pad_W;
  int input_offset;
  int output_offset;
  int activation_min;
  int activation_max;
} DepthwiseConv2DParams;

/* ADD first maps both inputs to a shared high-precision intermediate scale. */
typedef struct {
  int size;
  int input1_offset;
  int input2_offset;
  int output_offset;
  int left_shift;
  int input1_multiplier;
  int input1_shift;
  int input2_multiplier;
  int input2_shift;
  int output_multiplier;
  int output_shift;
  int activation_min;
  int activation_max;
} AddParams;

typedef struct {
  int batches;
  int input_H;
  int input_W;
  int input_C;
  int output_H;
  int output_W;
  int filter_H;
  int filter_W;
  int stride_H;
  int stride_W;
  int pad_H;
  int pad_W;
  int activation_min;
  int activation_max;
} AveragePool2DParams;

typedef struct {
  int batches;
  int input_C;
  int output_C;
  int input_offset;
  int output_offset;
  int activation_min;
  int activation_max;
} FullyConnectedParams;

/*
 * Doing (Output scale * shifted multiplier) >> shift
 * Different rounding methods are used in TFLite.
 */
int32_t output_scaling_round2(int32_t value, int32_t multiplier, int shift);
int32_t output_scaling_round1(int32_t value, int32_t multiplier, int shift);

/* Requantize each element between uint8/int8 tensor representations. */
void quantize_u8_to_s8(const uint8_t *input, int8_t *output, int size,
                       int input_zero_point, int output_zero_point,
                       int multiplier, int shift);
void quantize_s8_to_u8(const int8_t *input, uint8_t *output, int size,
                       int input_zero_point, int output_zero_point,
                       int multiplier, int shift);

/* Main kernels */
void conv2d_s8(const int8_t *input, const int8_t *filter,
               const int32_t *bias, const int32_t *output_multipliers,
               const int32_t *output_shifts, int8_t *output,
               const Conv2DParams *params);

void depthwise_conv2d_s8(const int8_t *input, const int8_t *filter,
                         const int32_t *bias,
                         const int32_t *output_multipliers,
                         const int32_t *output_shifts, int8_t *output,
                         const DepthwiseConv2DParams *params);

void add_s8(const int8_t *input1, const int8_t *input2,
            const AddParams *params, int8_t *output);

void average_pool2d_s8(const int8_t *input, int8_t *output,
                       const AveragePool2DParams *params);

void fully_connected_s8(const int8_t *input, const int8_t *weights,
                        const int32_t *bias, int output_multiplier,
                        int output_shift, int8_t *output,
                        const FullyConnectedParams *params);

void softmax_s8(const int8_t *input, int8_t *output, int outer_size,
                int depth, int input_multiplier, int input_left_shift,
                int diff_min);

#endif
