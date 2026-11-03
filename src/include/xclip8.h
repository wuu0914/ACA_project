#ifndef XCLIP8_H_
#define XCLIP8_H_

#include <stdint.h>

/*
 * Xclip8 contract:
 *
 *   rd = clamp(INT32(rs1), -128, 127)
 *
 * The result is sign-extended to XLEN.  Rs2 is fixed to x0 by the encoding.
 */
static inline int64_t xclip8(int32_t value) {
#if defined(__riscv) && __riscv_xlen == 64
  int64_t result = value;

  /* X3 implementation: CUSTOM_0, funct3 = 1, funct7 = 0, rs2 = x0. */
  __asm__ volatile(
      ".insn r 0x0b, 1, 0, %0, %0, x0"
      : "+r"(result));

  return result;
#else
  /* Host builds use the same signed INT32 clamp as the instruction. */
  if (value < -128) value = -128;
  if (value > 127) value = 127;
  return value;
#endif
}

#endif
