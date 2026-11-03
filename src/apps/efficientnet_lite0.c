#include "efficientnet_lite0.h"

#include <stddef.h>
#include <stdint.h>

#include "ca_kernels.h"
#include "efficientnet_lite0_data.h"

/*
 * efficientnet_lite0_data.h provides:
 *   1. efficientnet_lite0_tensors[]   - tensor address map
 *   2. efficientnet_lite0_operators[] - kernel I/O and parameters
*/

/* Return a tensor address of constant_data or arena */
static const void *read_tensor(int tensor_index, const uint8_t *arena) {
  const ModelTensor *tensor = &efficientnet_lite0_tensors[tensor_index];
  if (tensor->constant_data != 0) return tensor->constant_data;
  return arena + tensor->arena_offset;
}

/* Return a tensor address of arena. */
static void *write_tensor(int tensor_index, uint8_t *arena) {
  const ModelTensor *tensor = &efficientnet_lite0_tensors[tensor_index];
  return arena + tensor->arena_offset;
}

/* Copy bytes without depending on memcpy() in C library. */
static void copy_byte(void *destination, const void *source, size_t size) {
  uint8_t *output = destination;
  const uint8_t *input = source;

  for (size_t i = 0; i < size; ++i) output[i] = input[i];
}

/*
 * Dispatch one generated operator to its Phase 1-3 kernel.
 *
 * Common layout:
 *   input0              = activation from the preceding operator
 *   output              = activation written by this operator
 *   op->params.<kernel> = shape, offset, padding and activation limits
 *
 * Notes:
 *   Additional inputs have different meanings for each operator. They are
 *   documented beside each kernel call below.
 */
static int execute_operator(const ModelOperator *op, uint8_t *arena) {
  const void *input0 = read_tensor(op->input0, arena);    // input activation pointer
  void *output = write_tensor(op->output, arena);         // output activation pointer

  switch (op->type) {
    case MODEL_OP_QUANTIZE: {
      // EfficientNet-Lite0 uses uint8 -> int8 at input and int8 -> uint8 at output.
      const ModelQuantizeParams *p = &op->params.quantize;
      if (p->input_type == MODEL_TENSOR_UINT8 && p->output_type == MODEL_TENSOR_INT8) {
        quantize_u8_to_s8(input0, output, p->size, p->input_zero_point,
                          p->output_zero_point, p->multiplier, p->shift);
        return EFFICIENTNET_LITE0_OK;
      }
      if (p->input_type == MODEL_TENSOR_INT8 && p->output_type == MODEL_TENSOR_UINT8) {
        quantize_s8_to_u8(input0, output, p->size, p->input_zero_point,
                          p->output_zero_point, p->multiplier, p->shift);
        return EFFICIENTNET_LITE0_OK;
      }
      return EFFICIENTNET_LITE0_BAD_MODEL;
    }

    case MODEL_OP_CONV_2D:
      // Layout:
      //   input0 = input activation
      //   input1 = OHWI filter
      //   input2 = bias
      // multiplier/shift are per output channel.
      conv2d_s8(input0, read_tensor(op->input1, arena), read_tensor(op->input2, arena), 
                op->multipliers, op->shifts, output, &op->params.conv2d);
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_DEPTHWISE_CONV_2D:
      // Layout:
      //   input0 = input activation
      //   input1 = [1, filter_H, filter_W, output_C] filter
      //   input2 = bias
      // multiplier/shift are per output channel.
      depthwise_conv2d_s8(input0, read_tensor(op->input1, arena), read_tensor(op->input2, arena), 
                          op->multipliers, op->shifts, output, &op->params.depthwise_conv2d);
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_ADD:
      add_s8(input0, read_tensor(op->input1, arena), &op->params.add, output);
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_AVERAGE_POOL_2D:
      average_pool2d_s8(input0, output, &op->params.average_pool2d);
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_RESHAPE:
      // RESHAPE only changes the tensor view.
      if (read_tensor(op->input0, arena) != output) {return EFFICIENTNET_LITE0_BAD_MODEL;}
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_FULLY_CONNECTED:
      // Layout:
      //   input0 = flattened input activation
      //   input1 = [output_C, input_C] weights
      //   input2 = bias
      // EfficientNet-Lite0 uses one multiplier/shift for this final FC layer.
      fully_connected_s8(input0, read_tensor(op->input1, arena), read_tensor(op->input2, arena), 
                         op->multiplier, op->shift, output, &op->params.fully_connected);
      return EFFICIENTNET_LITE0_OK;

    case MODEL_OP_SOFTMAX: {
      const ModelSoftmaxParams *p = &op->params.softmax;
      softmax_s8(input0, output, p->outer_size, p->depth, p->input_multiplier, p->input_left_shift, p->diff_min);
      return EFFICIENTNET_LITE0_OK;
    }
  }

  return EFFICIENTNET_LITE0_BAD_MODEL;
}

/* Public model sizes let the caller allocate input, output and arena storage. */
size_t efficientnet_lite0_input_size(void) {return EFFICIENTNET_LITE0_INPUT_SIZE;}
size_t efficientnet_lite0_output_size(void) {return EFFICIENTNET_LITE0_OUTPUT_SIZE;}
size_t efficientnet_lite0_arena_size(void) {return EFFICIENTNET_LITE0_ARENA_SIZE;}


/*
 * EfficientNet-Lite0 executor:
 *
 * Flow:
 *   Copy model input into tensor arena
 *   Read one operator & Call the corresponding kernel
 *   Optionally report the operator output through trace()
 *   Copy the final output tensor from tensor arena to model output
 *
 * Notes:
 *   1. Weights and bias are read directly from efficientnet_lite0_data.c.
 *   2. Input, intermediate results and output are stored in tensor_arena.
 *   3. Activation tensors whose lifetimes do not overlap may share one offset.
 *   4. All shapes, padding and quantization parameters were prepared offline,
 *      so this executor does not need a TFLite parser or dynamic allocation.
 */

int efficientnet_lite0_invoke_with_trace(const uint8_t *input,
                                         uint8_t *output,
                                         void *tensor_arena,
                                         size_t arena_size,
                                         EfficientNetLite0Trace trace,  // function pointer of trace provided by tb
                                         void *user_data) {
  // Argument Check
  if (input == 0 || output == 0 || tensor_arena == 0) {
    return EFFICIENTNET_LITE0_BAD_ARGUMENT;
  }
  if (arena_size < EFFICIENTNET_LITE0_ARENA_SIZE) {
    return EFFICIENTNET_LITE0_ARENA_TOO_SMALL;
  }

  uint8_t *arena = tensor_arena;

  // Input Preparation
  // Copy caller-owned input into the arena assigned to the input tensor.
  copy_byte(write_tensor(EFFICIENTNET_LITE0_INPUT_TENSOR, arena), input,
             EFFICIENTNET_LITE0_INPUT_SIZE);

  // Run
  // Operator order is already topologically sorted by the original TFLite graph.
  for (int i = 0; i < EFFICIENTNET_LITE0_OPERATOR_COUNT; ++i) {
    const ModelOperator *op = &efficientnet_lite0_operators[i];
    const int status = execute_operator(op, arena);
    if (status != EFFICIENTNET_LITE0_OK) return status;

    // trace() runs immediately after this operator, before a later tensor can
    // reuse the same arena address. The callback must copy data it wants to keep.
    if (trace != 0) {
      const ModelTensor *tensor = &efficientnet_lite0_tensors[op->output];
      trace(i, op->output, read_tensor(op->output, arena), tensor->bytes, user_data);
    }
  }

  // Output
  // Copy the final tensor out so the caller does not depend on arena lifetime.
  copy_byte(output, read_tensor(EFFICIENTNET_LITE0_OUTPUT_TENSOR, arena),
             EFFICIENTNET_LITE0_OUTPUT_SIZE);
  return EFFICIENTNET_LITE0_OK;
}

int efficientnet_lite0_invoke(const uint8_t *input,
                              uint8_t *output,
                              void *tensor_arena,
                              size_t arena_size) {
  // Normal inference is the same executor with intermediate tracing disabled.
  return efficientnet_lite0_invoke_with_trace(input, output, tensor_arena, arena_size, 0, 0);
}
