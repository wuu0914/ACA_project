#ifndef EFFICIENTNET_LITE0_H_
#define EFFICIENTNET_LITE0_H_

#include <stddef.h>
#include <stdint.h>

// Complete EfficientNet-Lite0 INT8 model interface.

/*
 * Testbench
 *   |
 *   | provide input / output / arena memories
 *   v
 * C model
 *   |
 *   | copy input memory -> arena
 *   | run ModelOperator, use arena for intermediate activations
 *   | copy final result from arena -> output memory
 *   v
 * Testbench reads output memory
 */

enum {
  EFFICIENTNET_LITE0_OK = 0,
  EFFICIENTNET_LITE0_BAD_ARGUMENT = -1,
  EFFICIENTNET_LITE0_ARENA_TOO_SMALL = -2,
  EFFICIENTNET_LITE0_BAD_MODEL = -3,
};

/* Caller-provide function. Optional callback after each operator. Data points inside tensor_arena. */
typedef void (*EfficientNetLite0Trace)(int operator_index,
                                       int tensor_index,
                                       const void *data,
                                       size_t size,
                                       void *user_data);

/* Public model sizes let the caller allocate input, output and arena storage. */
size_t efficientnet_lite0_input_size(void);
size_t efficientnet_lite0_output_size(void);
size_t efficientnet_lite0_arena_size(void);

/* 
 * Run the complete model. 
 * Input and output are uint8 tensors. 
 * Caller provide intput image, output buffer, and tensor arena. 
 * The tensor arena must be at least arena_size bytes.
 */
int efficientnet_lite0_invoke(const uint8_t *input, 
                              uint8_t *output,      
                              void *tensor_arena,   
                              size_t arena_size);

/* Run the complete model and call trace after each operator. */
int efficientnet_lite0_invoke_with_trace(const uint8_t *input,
                                         uint8_t *output,
                                         void *tensor_arena,
                                         size_t arena_size,
                                         EfficientNetLite0Trace trace,
                                         void *user_data);

#endif
