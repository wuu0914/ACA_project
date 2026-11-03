#!/usr/bin/env python3
"""
Turn a Spike PC histogram into a readable EfficientNet-Lite0 baseline report.

Inputs:
    RV64IMC ELF                    symbol, instruction and section information
    Spike -g histogram            dynamic execution count for each PC
    phase2_estimated_macs.txt     estimated MAC count from model tensor shapes

Output:
    The measurements required by spec.md and each function's instruction share.
"""

import argparse
import bisect
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "deps/tools/riscv/bin"
DEFAULT_ELF = ROOT / "build/phase2/efficientnet_spike.elf"
DEFAULT_HISTOGRAM = ROOT / "build/phase2/spike.histogram"
DEFAULT_MACS = ROOT / "build/phase2/phase2_estimated_macs.txt"
DEFAULT_OUTPUT = ROOT / "build/phase2/phase2_spike_profile.txt"

HISTOGRAM_LINE = re.compile(r"^([0-9a-fA-F]+)\s+([0-9]+)$")
INSTRUCTION_LINE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([A-Za-z0-9_.]+)(?:\s+(.*))?$"
)
CUSTOM_WORD = re.compile(r"0x([0-9a-fA-F]{8})")
XQMAC8_MATCH = 0x0000000B
XQMAC8_MASK = 0xFE00707F
XCLIP8_MATCH = 0x0000100B
XCLIP8_MASK = 0xFFF0707F

LOADS = {
    "lb", "lbu", "lh", "lhu", "lw", "lwu", "ld",
    "c.lw", "c.lwsp", "c.ld", "c.ldsp",
}
STORES = {
    "sb", "sh", "sw", "sd",
    "c.sw", "c.swsp", "c.sd", "c.sdsp",
}
ADDS_SUBS = {
    "add", "addi", "addw", "addiw", "sub", "subw",
    "c.add", "c.addi", "c.addi4spn", "c.addi16sp",
    "c.addw", "c.addiw", "c.sub", "c.subw",
}
MULTIPLIES = {"mul", "mulh", "mulhsu", "mulhu", "mulw"}
DIVIDES = {"div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw"}
SHIFTS = {
    "sll", "slli", "sllw", "slliw",
    "srl", "srli", "srlw", "srliw",
    "sra", "srai", "sraw", "sraiw",
    "c.slli", "c.srli", "c.srai",
}
LOGIC = {
    "and", "andi", "or", "ori", "xor", "xori",
    "c.and", "c.andi", "c.or", "c.xor",
}
COMPARES = {"slt", "slti", "sltu", "sltiu"}
MOVES = {"c.mv"}
CONSTANTS = {"lui", "c.li"}
BRANCHES = {
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "jal", "jalr",
    "c.beqz", "c.bnez", "c.j", "c.jal", "c.jr", "c.jalr",
}
SYSTEM = {"ecall", "ebreak", "wfi", "fence", "fence.i"}
FUNCTION_CATEGORIES = [
    "load", "store", "add/sub", "multiply", "divide/rem", "shift",
    "logic", "compare", "move", "constant", "branch/jump", "xqmac8",
    "xclip8", "other",
]


def run_tool(name, *arguments):
    """Run the pinned RISC-V Binutils tool and return its text output."""
    command = [str(TOOL_DIR / name), *map(str, arguments)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout


def parse_histogram(path):
    """Read Spike's `PC count` lines; ignore the leading size label."""
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HISTOGRAM_LINE.fullmatch(line.strip())
        if match:
            counts[int(match.group(1), 16)] += int(match.group(2))
    if not counts:
        raise ValueError(f"no PC/count entries found in {path}")
    return counts


def disassemble(elf):
    """Map each ELF instruction address to the mnemonic printed by objdump."""
    output = run_tool(
        "riscv64-unknown-elf-objdump",
        "-d", "--no-show-raw-insn", "-M", "no-aliases", elf,
    )
    instructions = {}

    for line in output.splitlines():
        instruction = INSTRUCTION_LINE.match(line)
        if instruction:
            address = int(instruction.group(1), 16)
            mnemonic = instruction.group(2).lower()

            # X1 implementation: Binutils does not know the custom mnemonic,
            # but it preserves the 32-bit encoding in `.insn 4, 0x........`.
            if mnemonic == ".insn":
                words = CUSTOM_WORD.findall(instruction.group(3) or "")
                if words:
                    word = int(words[-1], 16)
                    if word & XCLIP8_MASK == XCLIP8_MATCH:
                        mnemonic = "xclip8"
                    elif word & XQMAC8_MASK == XQMAC8_MATCH:
                        mnemonic = "xqmac8"
            instructions[address] = mnemonic

    if not instructions:
        raise ValueError(f"no instructions found in {elf}")
    return instructions


def function_symbols(elf):
    """Read the start, size and name of every text symbol from nm."""
    output = run_tool(
        "riscv64-unknown-elf-nm", "-n", "-S", "--defined-only", elf
    )
    symbols = []
    for line in output.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4 or fields[2] not in {"T", "t"}:
            continue
        start = int(fields[0], 16)
        size = int(fields[1], 16)
        if size:
            symbols.append((start, start + size, fields[3]))
    if not symbols:
        raise ValueError(f"no text symbols found in {elf}")
    return symbols


def find_function(pc, symbols, starts):
    """Return the nm symbol whose half-open address range contains one PC."""
    index = bisect.bisect_right(starts, pc) - 1
    if index >= 0 and pc < symbols[index][1]:
        return symbols[index][2]
    return "<unknown>"


def classify_instruction(mnemonic):
    """Group RV64IMC instructions into the spec categories plus other."""
    if mnemonic == "xqmac8":
        return "xqmac8"
    if mnemonic == "xclip8":
        return "xclip8"
    if mnemonic in LOADS:
        return "load"
    if mnemonic in STORES:
        return "store"
    if mnemonic in MULTIPLIES:
        return "multiply"
    if mnemonic in BRANCHES:
        return "branch/jump"
    if mnemonic in DIVIDES or mnemonic in SYSTEM or mnemonic.startswith("csr"):
        return "other"
    return "integer ALU"


def classify_function_instruction(mnemonic):
    """Use the extra detail requested for each function's instruction mix."""
    if mnemonic == "xqmac8":
        return "xqmac8"
    if mnemonic == "xclip8":
        return "xclip8"
    if mnemonic in LOADS:
        return "load"
    if mnemonic in STORES:
        return "store"
    if mnemonic in ADDS_SUBS:
        return "add/sub"
    if mnemonic in MULTIPLIES:
        return "multiply"
    if mnemonic in DIVIDES:
        return "divide/rem"
    if mnemonic in SHIFTS:
        return "shift"
    if mnemonic in LOGIC:
        return "logic"
    if mnemonic in COMPARES:
        return "compare"
    if mnemonic in MOVES:
        return "move"
    if mnemonic in CONSTANTS:
        return "constant"
    if mnemonic in BRANCHES:
        return "branch/jump"
    return "other"


def text_size(elf):
    """Return the ELF .text size reported by RISC-V size."""
    output = run_tool("riscv64-unknown-elf-size", "-A", elf)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == ".text":
            return int(fields[1])
    raise ValueError(f"no .text section found in {elf}")


def read_estimated_macs(path):
    """Read per-function and total estimated MACs from cIO_prepare.py."""
    estimated_macs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in estimated_macs:
            raise ValueError(f"invalid estimated MAC line: {line!r}")
        estimated_macs[fields[0]] = int(fields[1])

    if "total" not in estimated_macs or estimated_macs["total"] <= 0:
        raise ValueError(f"missing positive estimated MAC total: {path}")
    function_total = sum(
        value for name, value in estimated_macs.items() if name != "total"
    )
    if function_total != estimated_macs["total"]:
        raise ValueError(f"per-function estimated MACs do not add up: {path}")
    return estimated_macs


def percentage(part, whole):
    return 100.0 * part / whole if whole else 0.0


def format_table(rows, headers):
    """Create a small aligned plain-text table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def render(row):
        return "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    separator = tuple("-" * width for width in widths)
    return "\n".join([render(headers), render(separator), *(render(row) for row in rows)])


def build_report(elf, histogram, estimated_macs_path):
    pc_counts = parse_histogram(histogram)
    instructions = disassemble(elf)
    symbols = function_symbols(elf)
    symbol_starts = [symbol[0] for symbol in symbols]
    estimated_macs = read_estimated_macs(estimated_macs_path)
    total_macs = estimated_macs["total"]

    category_counts = Counter()
    function_counts = Counter()
    function_category_counts = Counter()
    mapped_count = 0

    for pc, count in pc_counts.items():
        instruction = instructions.get(pc)
        if instruction is None:
            continue
        mnemonic = instruction
        function = find_function(pc, symbols, symbol_starts)
        mapped_count += count
        category_counts[classify_instruction(mnemonic)] += count
        function_counts[function] += count
        function_category = classify_function_instruction(mnemonic)
        function_category_counts[(function, function_category)] += count

    simulated_count = sum(pc_counts.values())
    outside_elf_count = simulated_count - mapped_count
    memory_count = category_counts["load"] + category_counts["store"]
    macs_per_instruction = total_macs / mapped_count if mapped_count else 0.0
    loads_per_mac = category_counts["load"] / total_macs
    stores_per_mac = category_counts["store"] / total_macs
    xqmac8_count = category_counts["xqmac8"]
    xclip8_count = category_counts["xclip8"]

    extensions = []
    if xqmac8_count:
        extensions.append("Xqmac8")
    if xclip8_count:
        extensions.append("Xclip8")
    title = "EfficientNet-Lite0 RV64IMC"
    if extensions:
        title += " + " + " + ".join(extensions)
    title += " Spike Profile"
    lines = [
        title,
        "=" * len(title),
        "",
        f"ELF: {elf}",
        f"Histogram: {histogram}",
        f"Estimated MACs: {estimated_macs_path}",
        "",
        "Required measurements",
        "---------------------",
        f"Dynamic instructions inside ELF:  {mapped_count:,}",
        f"Spike instructions outside ELF:   {outside_elf_count:,} (excluded)",
        f"Dynamic memory instructions:       {memory_count:,}",
        f"Estimated model MACs:               {total_macs:,}",
        f"Estimated MACs / retired instruction: {macs_per_instruction:.6f}",
        f"Load instructions / estimated MAC:   {loads_per_mac:.3f}",
        f"Store instructions / estimated MAC:  {stores_per_mac:.3f}",
        f"Code size (.text):                  {text_size(elf):,} bytes",
        f"Xqmac8 instructions:                {xqmac8_count:,}",
        f"MACs executed by Xqmac8:            {xqmac8_count * 8:,}",
        f"Xqmac8 / retired instruction:       {percentage(xqmac8_count, mapped_count):.2f}%",
        f"Xclip8 instructions:                {xclip8_count:,}",
        f"Xclip8 / retired instruction:       {percentage(xclip8_count, mapped_count):.2f}%",
        "",
        "Instruction mix",
        "---------------",
    ]

    category_order = [
        "load", "store", "integer ALU", "multiply", "xqmac8", "xclip8",
        "branch/jump", "other"
    ]
    category_rows = []
    for category in category_order:
        count = category_counts[category]
        category_rows.append((category, f"{count:,}", f"{percentage(count, mapped_count):.2f}%"))
    lines.append(format_table(category_rows, ("Category", "Dynamic count", "Share")))

    lines.extend(["", "Function profiling", "------------------"])
    function_rows = []
    for function, count in function_counts.most_common():
        function_macs = estimated_macs.get(function)
        function_rows.append((
            function,
            f"{count:,}",
            f"{percentage(count, mapped_count):.2f}%",
            f"{function_macs:,}" if function_macs is not None else "-",
            f"{percentage(function_macs, total_macs):.2f}%"
            if function_macs is not None else "-",
        ))
    lines.append(format_table(
        function_rows,
        ("Function", "Dynamic instructions", "Inst %", "Estimated MACs", "MAC %"),
    ))

    lines.extend(["", "Instructions inside each function", "---------------------------------"])
    function_mix_rows = []
    for function, total in function_counts.most_common():
        counts = [
            function_category_counts[(function, category)]
            for category in FUNCTION_CATEGORIES
        ]
        if sum(counts) != total:
            raise ValueError(f"instruction categories do not add up for {function}")
        function_mix_rows.append((
            function,
            *(f"{count:,}" for count in counts),
        ))
    lines.append(format_table(
        function_mix_rows,
        ("Function", *FUNCTION_CATEGORIES),
    ))

    lines.extend([
        "",
        "Notes",
        "-----",
        "- Instruction-mix other contains divide/remainder and system instructions.",
        "- Per-function other contains remaining address and system instructions.",
        "- Estimated MACs count Conv2D, DepthwiseConv2D and FullyConnected work.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--histogram", type=Path, default=DEFAULT_HISTOGRAM)
    parser.add_argument("--estimated-macs", type=Path, default=DEFAULT_MACS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(args.elf, args.histogram, args.estimated_macs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
