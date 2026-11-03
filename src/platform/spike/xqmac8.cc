#include <riscv/extension.h>

#include <cstdint>
#include <vector>

/*
 * Formal Spike model for the Xqmac8 instruction used by the X1 kernel.
 *
 * Encoding:
 *   opcode = CUSTOM_0 (0x0b), funct3 = 0, funct7 = 0
 *
 * Semantics:
 *   rd = INT32(rd) + sum(INT8(rs1[lane]) * INT8(rs2[lane])), lane 0..7
 *
 * Lane 0 is the least-significant byte.  Accumulation is explicitly unsigned
 * so overflow follows the instruction contract's modulo-2^32 rule.
 */
static constexpr insn_bits_t MATCH_XQMAC8 = 0x0000000b;
static constexpr insn_bits_t MASK_XQMAC8 = 0xfe00707f;

static int32_t read_lane(reg_t packed, int lane) {
  return static_cast<int8_t>((packed >> (lane * 8)) & 0xff);
}

static reg_t execute_xqmac8(processor_t *processor,
                            insn_t instruction,
                            reg_t pc) {
  state_t *state = processor->get_state();
  const reg_t packed_input = state->XPR[instruction.rs1()];
  const reg_t packed_weight = state->XPR[instruction.rs2()];
  uint32_t accumulator = static_cast<uint32_t>(state->XPR[instruction.rd()]);

  for (int lane = 0; lane < 8; ++lane) {
    const int32_t product = read_lane(packed_input, lane) * read_lane(packed_weight, lane);
    accumulator += static_cast<uint32_t>(product);
  }

  const reg_t result = static_cast<reg_t>(static_cast<int32_t>(accumulator));
  if (instruction.rd() != 0) {
    state->log_reg_write[instruction.rd() << 4] = {result, 0};
    state->XPR.write(instruction.rd(), result);
  }
  return pc + 4;
}

class Xqmac8Extension final : public extension_t {
 public:
  const char *name() const override { return "xqmac8"; }

  std::vector<insn_desc_t> get_instructions(const processor_t &) override {
    return {{MATCH_XQMAC8, MASK_XQMAC8,
             &::illegal_instruction, execute_xqmac8,
             &::illegal_instruction, execute_xqmac8,
             &::illegal_instruction, execute_xqmac8,
             &::illegal_instruction, execute_xqmac8}};
  }

  std::vector<disasm_insn_t *> get_disasms(const processor_t *) override {
    return {new disasm_insn_t("xqmac8", MATCH_XQMAC8, MASK_XQMAC8, {})};
  }
};

REGISTER_EXTENSION(xqmac8, []() {
  static Xqmac8Extension extension;
  return &extension;
})
