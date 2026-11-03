#ifndef XQMAC8_H_
#define XQMAC8_H_

#include <stdint.h>

/*
 * Xqmac8 contract:
 *
 *   accumulator += sum(INT8(input[lane]) * INT8(weight[lane])), lane 0..7
 *
 * Lane 0 occupies the least-significant byte.  The accumulator wraps modulo
 * 2^32 and the result is interpreted as a signed INT32 value.
 */
static inline int64_t xqmac8(int64_t accumulator,
                             uint64_t packed_input,
                             uint64_t packed_weight) {
#if defined(__riscv) && __riscv_xlen == 64
  int64_t result = accumulator;

  /* X1 implementation: CUSTOM_0, funct3 = 0, funct7 = 0. */
  __asm__ volatile(
      ".insn r 0x0b, 0, 0, %0, %1, %2"
      : "+r"(result)
      : "r"(packed_input), "r"(packed_weight));

  return result;
#else
  /* Host builds use the same lane and INT32 wraparound semantics. */
  uint32_t result = (uint32_t)accumulator;

  for (int lane = 0; lane < 8; ++lane) {
    const int32_t input = (int8_t)(packed_input >> (lane * 8));
    const int32_t weight = (int8_t)(packed_weight >> (lane * 8));
    result += (uint32_t)(input * weight);
  }

  return (int64_t)(int32_t)result;
#endif
}

/*
 *  __builtin_assume_aligned tells GCC that
 *  one packed group can be emitted as a single RV64 ld instruction.
 */
typedef uint64_t Xqmac8Packed __attribute__((__may_alias__));

static inline uint64_t xqmac8_load_aligned(const int8_t *pointer) {
  const void *aligned = __builtin_assume_aligned(pointer, 8);
  return *(const Xqmac8Packed *)aligned;
}

#endif
