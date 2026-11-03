#!/usr/bin/env python3
"""Collect the six Spike profile files into one small Markdown table."""

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    ("B0", ROOT / "build/phase2/phase2_spike_profile.txt"),
    ("B1", ROOT / "build/b1/phase3_2_spike_profile.txt"),
    ("X1", ROOT / "build/x1/phase3_3_spike_profile.txt"),
    ("B2", ROOT / "build/b2/phase4_2_spike_profile.txt"),
    ("B3", ROOT / "build/b3/phase5_1_spike_profile.txt"),
    ("X3", ROOT / "build/x3/phase5_2_spike_profile.txt"),
)


def metric(text, label, value_pattern):
    """Read one named value from the Required measurements section."""
    match = re.search(rf"^{re.escape(label)}\s+({value_pattern})\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing metric: {label}")
    return match.group(1)


def optional_count(text, label):
    """Return zero when a configuration does not implement the instruction."""
    match = re.search(rf"^{re.escape(label)}\s+([0-9,]+)\s*$", text, re.MULTILINE)
    return int(match.group(1).replace(",", "")) if match else 0


def read_profile(name, path):
    """Extract only the values needed for the cross-version comparison."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing; run make spike-{name.lower()}")

    text = path.read_text(encoding="utf-8")
    return {
        "name": name,
        "instructions": int(metric(
            text, "Dynamic instructions inside ELF:", r"[0-9,]+"
        ).replace(",", "")),
        "macs_per_instruction": metric(
            text, "Estimated MACs / retired instruction:", r"[0-9.]+"
        ),
        "text_bytes": int(metric(text, "Code size (.text):", r"[0-9,]+ bytes").split()[0].replace(",", "")),
        "xqmac8": optional_count(text, "Xqmac8 instructions:"),
        "xclip8": optional_count(text, "Xclip8 instructions:"),
    }


def render(rows):
    """Render one table with every value measured from the profile files."""
    baseline = rows[0]["instructions"]
    lines = [
        "# Spike Profile Summary",
        "",
        "| Version | Retired instructions | Reduction vs B0 | MACs / instruction | .text bytes | Xqmac8 | Xclip8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        reduction = 100.0 * (baseline - row["instructions"]) / baseline
        lines.append(
            f"| {row['name']} | {row['instructions']:,} | {reduction:.2f}% | "
            f"{row['macs_per_instruction']} | {row['text_bytes']:,} | "
            f"{row['xqmac8']:,} | {row['xclip8']:,} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/spike_profile_summary.md")
    args = parser.parse_args()

    rows = [read_profile(name, path) for name, path in PROFILES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
