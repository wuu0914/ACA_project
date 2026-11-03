#include <riscv/extension.h>

#include <cstdint>
#include <vector>

/*
 * Formal Spike model for the Xclip8 instruction used by the X3 kernel.
 *
 * Encoding:
 *   opcode = CUSTOM_0 (0x0b), funct3 = 1, funct7 = 0, rs2 = x0
 *
 * Semantics:
 *   rd = sign_extend_64(clamp(INT32(rs1), -128, 127))
 */
static constexpr insn_bits_t MATCH_XCLIP8 = 0x0000100b;
static constexpr insn_bits_t MASK_XCLIP8 = 0xfff0707f;

static reg_t execute_xclip8(processor_t *processor,
                            insn_t instruction,
                            reg_t pc) {
  state_t *state = processor->get_state();
  const uint32_t value = static_cast<uint32_t>(state->XPR[instruction.rs1()]);

  /*
   * X3 hardware structure:
   *   value[31:7] == 0          -> 0 .. 127, keep the original value
   *   value[31:7] == all ones   -> -128 .. -1, keep the original value
   *   otherwise use value[31] to select 127 or -128
   */
  const uint32_t upper = value >> 7;
  const bool in_range = upper == 0 || upper == UINT32_C(0x01ffffff);
  int32_t clipped = static_cast<int32_t>(value);
  if (!in_range) clipped = (value >> 31) ? -128 : 127;
  const reg_t result = static_cast<reg_t>(static_cast<int64_t>(clipped));

  if (instruction.rd() != 0) {
    state->log_reg_write[instruction.rd() << 4] = {result, 0};
    state->XPR.write(instruction.rd(), result);
  }
  return pc + 4;
}

class Xclip8Extension final : public extension_t {
 public:
  const char *name() const override { return "xclip8"; }

  std::vector<insn_desc_t> get_instructions(const processor_t &) override {
    return {{MATCH_XCLIP8, MASK_XCLIP8,
             &::illegal_instruction, execute_xclip8,
             &::illegal_instruction, execute_xclip8,
             &::illegal_instruction, execute_xclip8,
             &::illegal_instruction, execute_xclip8}};
  }

  std::vector<disasm_insn_t *> get_disasms(const processor_t *) override {
    return {new disasm_insn_t("xclip8", MATCH_XCLIP8, MASK_XCLIP8, {})};
  }
};

REGISTER_EXTENSION(xclip8, []() {
  static Xclip8Extension extension;
  return &extension;
})
