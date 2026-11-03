#!/usr/bin/env python3
"""
Prepare one fixed EfficientNet-Lite0 inference for bare-metal Spike.

Flow:
    Preprocess and quantize panda.jpg
    Run the TFLite reference model once
    Write model input and golden output as C arrays
    Estimate model MACs from the TFLite tensor shapes
"""

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
DEFAULT_IMAGE = ROOT / "models/testdata/panda.jpg"
DEFAULT_DATA = ROOT / "build/phase2/phase2_test_data.c"
DEFAULT_MACS = ROOT / "build/phase2/phase2_estimated_macs.txt"

OPERATOR_FUNCTION = {
    "CONV_2D": "conv2d_s8",
    "DEPTHWISE_CONV_2D": "depthwise_conv2d_s8",
    "FULLY_CONNECTED": "fully_connected_s8",
}


def preprocess_image(path):
    """
    Reproduce the official EfficientNet-Lite image preprocessing.

    Flow:
        RGB image
        -> center crop 87.5% of the shorter side
        -> bicubic resize to 224 x 224
        -> float32 normalization: (pixel - 127) / 128
        -> add batch dimension: [224, 224, 3] -> [1, 224, 224, 3]
    """
    image = Image.open(path).convert("RGB")
    crop_size = int(0.875 * min(image.size))
    left = ((image.width - crop_size) + 1) // 2
    top = ((image.height - crop_size) + 1) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32)
    return ((pixels - 127.0) / 128.0)[None, ...]


def quantize_input(normalized, input_detail):
    """
    Convert real input into the uint8 tensor expected by the TFLite model.

    Math:
        real = scale * (quantized - zero_point)
        quantized = clamp(round(real / scale) + zero_point, 0, 255)
    """
    scale, zero_point = input_detail["quantization"]
    limits = np.iinfo(input_detail["dtype"])
    return np.ascontiguousarray(
        np.clip(np.rint(normalized / scale + zero_point), limits.min, limits.max)
        .astype(input_detail["dtype"])
    )


def array_product(shape):
    """Multiply a tensor shape without returning a NumPy integer."""
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def estimate_operator_macs(name, input_shapes, output_shape):
    """
    Estimate conventional neural-network MACs from tensor shapes.

    CONV_2D filter layout:             [output_C, filter_H, filter_W, input_C]
    DEPTHWISE_CONV_2D filter layout:   [1, filter_H, filter_W, output_C]
    FULLY_CONNECTED weight layout:     [output_C, input_C]

    ADD, pooling, quantize and softmax still execute integer instructions, but
    they are not counted as multiply-accumulate operations in the model MAC total.
    """
    if name == "CONV_2D":
        filter_shape = input_shapes[1]
        return array_product(output_shape) * array_product(filter_shape[1:])
    if name == "DEPTHWISE_CONV_2D":
        filter_shape = input_shapes[1]
        return array_product(output_shape) * int(filter_shape[1]) * int(filter_shape[2])
    if name == "FULLY_CONNECTED":
        weight_shape = input_shapes[1]
        return array_product(output_shape) * int(weight_shape[-1])
    return 0


def tensor_shape(tensor_details, tensor_index):
    """Return one tensor shape as ordinary Python integers."""
    if tensor_index < 0:
        return None
    return [int(value) for value in tensor_details[tensor_index]["shape"]]


def estimate_model_macs(interpreter):
    """Add estimated MACs by the C kernel function that executes each operator."""
    tensor_details = {
        int(detail["index"]): detail for detail in interpreter.get_tensor_details()
    }
    estimated_macs = {function: 0 for function in OPERATOR_FUNCTION.values()}

    for operation in interpreter._get_ops_details():
        name = operation["op_name"]
        function = OPERATOR_FUNCTION.get(name)
        if function is None:
            continue

        input_indices = [int(value) for value in operation["inputs"]]
        output_indices = [int(value) for value in operation["outputs"]]
        input_shapes = [tensor_shape(tensor_details, index) for index in input_indices]
        output_shape = tensor_shape(tensor_details, output_indices[0])
        estimated_macs[function] += estimate_operator_macs(
            name, input_shapes, output_shape
        )

    return estimated_macs


def write_estimated_macs(path, estimated_macs):
    """Write per-function and total estimated MACs as a small text file."""
    total = sum(estimated_macs.values())
    lines = [
        *(f"{function} {macs}" for function, macs in estimated_macs.items()),
        f"total {total}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_u8_array(name, array):
    """Render one flat uint8 NumPy array as a readable C definition."""
    values = [str(int(value)) for value in array.reshape(-1)]
    lines = []
    for start in range(0, len(values), 16):
        lines.append("  " + ", ".join(values[start:start + 16]) + ",")
    return (
        f"const uint8_t {name}[{len(values)}] = {{\n"
        + "\n".join(lines)
        + "\n};\n"
    )


def write_test_data(path, model_input, golden_output):
    """Write the only input and golden arrays required by the target testbench."""
    text = f"""/* Generated by scripts/cIO_prepare.py. Do not edit. */
#include <stdint.h>

{format_u8_array("phase2_model_input", model_input)}
{format_u8_array("phase2_golden_output", golden_output)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-macs", type=Path, default=DEFAULT_MACS)
    args = parser.parse_args()

    # Prepare golden model
    interpreter = Interpreter(
        model_path=str(args.model),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    # Preprocess model input
    normalized = preprocess_image(args.image)
    model_input = quantize_input(normalized, input_detail)

    # Get golden output
    interpreter.set_tensor(input_detail["index"], model_input)
    interpreter.invoke()
    golden_output = np.ascontiguousarray(
        interpreter.get_tensor(output_detail["index"])
    ).view(np.uint8).reshape(-1)

    write_test_data(args.output_data, model_input, golden_output)

    # Get estimate mac for profiling
    estimated_macs = estimate_model_macs(interpreter)
    write_estimated_macs(args.output_macs, estimated_macs)

    print(f"input:          {model_input.size} uint8 values")
    print(f"golden output:  {golden_output.size} uint8 values")
    print(f"estimated MACs: {sum(estimated_macs.values()):,}")
    print(f"data:           {args.output_data}")
    print(f"estimated MACs: {args.output_macs}")


if __name__ == "__main__":
    main()
