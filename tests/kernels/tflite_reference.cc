#include "ca_kernels.h"

#include <cstdint>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"

/*
 * C wrapper around the official TFLite C++ reference functions.
 *
 * Python ctypes cannot call C++ templates or namespaced symbols directly, so
 * each function below only translates our plain C parameter structs
 * into TFLite structs and immediately calls the corresponding reference kernel. 
 */

namespace {
/* RuntimeShape stores dimensions dynamically; most image operators need NHWC. */
  tflite::RuntimeShape Shape4(int a, int b, int c, int d) {
    return tflite::RuntimeShape({a, b, c, d});
  }
}  

extern "C" {

/* Requantize owns the official zero-point, Q0.31 multiply, and clamp order. */
void tflite_ref_quantize_u8_to_s8(const uint8_t *input, 
                                  int8_t *output, 
                                  int size, 
                                  int input_zero_point,
                                  int output_zero_point, 
                                  int multiplier, 
                                  int shift) {
  tflite::reference_ops::Requantize(input, size, multiplier, shift, input_zero_point, output_zero_point, output);
}

void tflite_ref_quantize_s8_to_u8(const int8_t *input, 
                                  uint8_t *output, 
                                  int size, 
                                  int input_zero_point,
                                  int output_zero_point, 
                                  int multiplier, 
                                  int shift) {
  tflite::reference_ops::Requantize(input, size, multiplier, shift, input_zero_point, output_zero_point, output);
}

void tflite_ref_conv2d_s8(const int8_t *input, 
                          const int8_t *filter, 
                          const int32_t *bias,
                          const int32_t *output_multipliers, 
                          const int32_t *output_shifts,
                          int8_t *output, 
                          const Conv2DParams *p) {
  /* Keep this field mapping explicit so the C and TFLite contracts are visible. */
  tflite::ConvParams params{};
  params.stride_height = p->stride_H;
  params.stride_width = p->stride_W;
  params.dilation_height_factor = p->dilation_H;
  params.dilation_width_factor = p->dilation_W;
  params.padding_values.height = p->pad_H;
  params.padding_values.width = p->pad_W;
  params.input_offset = p->input_offset;
  params.output_offset = p->output_offset;
  params.quantized_activation_min = p->activation_min;
  params.quantized_activation_max = p->activation_max;

  // The shape below is for TFLite reference kernel input
  tflite::reference_integer_ops::ConvPerChannel(
      params, output_multipliers, output_shifts,
      Shape4(p->batches, p->input_H, p->input_W, p->input_C),           // input shape
      input,
      Shape4(p->output_C, p->filter_H, p->filter_W, p->filter_input_C), // filter shape
      filter, tflite::RuntimeShape({p->output_C}), bias,
      Shape4(p->batches, p->output_H, p->output_W, p->output_C),        // output shape
      output);
}

void tflite_ref_depthwise_conv2d_s8(const int8_t *input, 
                                    const int8_t *filter, 
                                    const int32_t *bias,
                                    const int32_t *output_multipliers, 
                                    const int32_t *output_shifts,
                                    int8_t *output, 
                                    const DepthwiseConv2DParams *p) {
  tflite::DepthwiseParams params{};
  params.stride_height = p->stride_H;
  params.stride_width = p->stride_W;
  params.dilation_height_factor = p->dilation_H;
  params.dilation_width_factor = p->dilation_W;
  params.padding_values.height = p->pad_H;
  params.padding_values.width = p->pad_W;
  params.depth_multiplier = p->depth_multiplier;
  params.input_offset = p->input_offset;
  params.output_offset = p->output_offset;
  params.quantized_activation_min = p->activation_min;
  params.quantized_activation_max = p->activation_max;

  /* TFLite depthwise filter layout is [1, H, W, output_channels]. */
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multipliers, output_shifts,
      Shape4(p->batches, p->input_H, p->input_W, p->input_C),
      input,
      Shape4(1, p->filter_H, p->filter_W, p->output_C), filter,
      tflite::RuntimeShape({p->output_C}), bias,
      Shape4(p->batches, p->output_H, p->output_W, p->output_C),
      output);
}

void tflite_ref_add_s8(const int8_t *input1, 
                       const int8_t *input2,
                       const AddParams *p, 
                       int8_t *output) {
  tflite::ArithmeticParams params{};
  params.input1_offset = p->input1_offset;
  params.input2_offset = p->input2_offset;
  params.output_offset = p->output_offset;
  params.left_shift = p->left_shift;
  params.input1_multiplier = p->input1_multiplier;
  params.input1_shift = p->input1_shift;
  params.input2_multiplier = p->input2_multiplier;
  params.input2_shift = p->input2_shift;
  params.output_multiplier = p->output_multiplier;
  params.output_shift = p->output_shift;
  params.quantized_activation_min = p->activation_min;
  params.quantized_activation_max = p->activation_max;
  /* A 1-D shape is sufficient because ADD is element-wise in these tests. */
  const tflite::RuntimeShape shape({p->size});
  tflite::reference_integer_ops::Add(params, shape, input1, shape, input2,
                                     shape, output);
}

void tflite_ref_average_pool2d_s8(const int8_t *input, 
                                  int8_t *output,
                                  const AveragePool2DParams *p) {
  tflite::PoolParams params{};
  params.stride_height = p->stride_H;
  params.stride_width = p->stride_W;
  params.filter_height = p->filter_H;
  params.filter_width = p->filter_W;
  params.padding_values.height = p->pad_H;
  params.padding_values.width = p->pad_W;
  params.quantized_activation_min = p->activation_min;
  params.quantized_activation_max = p->activation_max;
  tflite::reference_integer_ops::AveragePool(
      params, Shape4(p->batches, p->input_H, p->input_W, p->input_C), input,
      Shape4(p->batches, p->output_H, p->output_W, p->input_C), output);
}

void tflite_ref_fully_connected_s8(const int8_t *input, 
                                   const int8_t *weights, 
                                   const int32_t *bias,
                                   float input_scale, 
                                   float output_scale, 
                                   const float *weight_scales,
                                   int8_t *output, 
                                   const FullyConnectedParams *p) {
  tflite::FullyConnectedParams params{};
  params.input_offset = p->input_offset;
  params.output_offset = p->output_offset;
  params.quantized_activation_min = p->activation_min;
  params.quantized_activation_max = p->activation_max;
  /* This overload accepts real tensor scales and performs its own final scale. */
  tflite::reference_integer_ops::FullyConnectedPerChannel(
      params, tflite::RuntimeShape({p->batches, p->input_C}), input,
      tflite::RuntimeShape({p->output_C, p->input_C}), weights,
      tflite::RuntimeShape({p->output_C}), bias,
      tflite::RuntimeShape({p->batches, p->output_C}), input_scale,
      output_scale, weight_scales, output);
}

void tflite_ref_softmax_s8(const int8_t *input, 
                           int8_t *output,
                           int outer_size, 
                           int depth, 
                           int input_multiplier,
                           int input_left_shift, 
                           int diff_min) {
  tflite::SoftmaxParams params{};
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  params.diff_min = diff_min;
  /* Softmax treats the last dimension as the class/depth dimension. */
  const tflite::RuntimeShape shape({outer_size, depth});
  tflite::reference_ops::Softmax(params, shape, input, shape, output);
}

}  // extern "C"
