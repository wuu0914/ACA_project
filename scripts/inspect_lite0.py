#!/usr/bin/env python3
"""List the TFLite operators used by the EfficientNet-Lite0 INT8 model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
OPERATOR_TYPES = (
    "QUANTIZE",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "ADD",
    "AVERAGE_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
)


def interpreter_for(model_path: Path):
    """Load a model with LiteRT builtin kernels and no default delegate."""
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    return Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )


def operator_names(model_path: Path) -> list[str]:
    interpreter = interpreter_for(model_path)
    interpreter.allocate_tensors()

    # LiteRT has no public Python method that lists every operator. This private
    # inspection helper is stable here because ai-edge-litert is version-pinned.
    return [operator["op_name"] for operator in interpreter._get_ops_details()]


def print_inventory(names: list[str]) -> None:
    counts = Counter(names)
    print(f"EfficientNet-Lite0 INT8: {len(names)} operator instances")
    print(f"Operator types: {len(counts)}")
    for name in OPERATOR_TYPES:
        print(f"  {name:20s} {counts[name]:2d}")

    unexpected = set(counts) - set(OPERATOR_TYPES)
    missing = set(OPERATOR_TYPES) - set(counts)
    if unexpected or missing:
        raise RuntimeError(
            f"model operator set changed; unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print_inventory(operator_names(args.model))


if __name__ == "__main__":
    main()
