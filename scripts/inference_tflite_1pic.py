#!/usr/bin/env python3
"""Run one image through the EfficientNet-Lite0 INT8 TFLite model."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
IMAGE = ROOT / "models/testdata/panda.jpg"
LABELS = ROOT / "models/testdata/labels_map.txt"


def main() -> None:
    # 1. Load the complete .tflite model and allocate its tensor memory.
    interpreter = Interpreter(model_path=str(MODEL), num_threads=1)
    interpreter.allocate_tensors()

    # Do not guess a model's input/output index, shape, dtype, or quantization.
    # Read them from the model because another .tflite may use different values.
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    # 2. Apply the preprocessing published with EfficientNet-Lite0:
    #    center-crop 87.5%, resize to 224x224, then normalize RGB pixels.
    image = Image.open(IMAGE).convert("RGB")
    crop_size = int(0.875 * min(image.size))
    left = ((image.width - crop_size) + 1) // 2
    top = ((image.height - crop_size) + 1) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32)
    normalized = ((pixels - 127.0) / 128.0)[None, ...]  # NHWC batch

    # 3. Quantize real-valued input x with q = round(x / scale) + zero_point.
    input_scale, input_zero_point = input_info["quantization"]
    limits = np.iinfo(input_info["dtype"])
    model_input = np.clip(
        np.rint(normalized / input_scale + input_zero_point),
        limits.min,
        limits.max,
    ).astype(input_info["dtype"])

    # 4. Copy the input into the Interpreter and execute the whole graph.
    interpreter.set_tensor(input_info["index"], model_input)
    interpreter.invoke()

    # 5. Read the quantized output and convert it back to probabilities with
    #    x = (q - zero_point) * scale.
    quantized_output = interpreter.get_tensor(output_info["index"])
    output_scale, output_zero_point = output_info["quantization"]
    probabilities = (
        quantized_output.astype(np.float32) - output_zero_point
    ) * output_scale
    probabilities = probabilities[0]

    labels = ast.literal_eval(LABELS.read_text(encoding="utf-8"))
    top5 = np.argsort(probabilities)[-5:][::-1]

    print(f"model: {MODEL.relative_to(ROOT)}")
    print(f"image: {IMAGE.relative_to(ROOT)}")
    print(
        f"input: shape={input_info['shape'].tolist()} "
        f"dtype={input_info['dtype'].__name__} "
        f"scale={input_scale:.8g} zero_point={input_zero_point}"
    )
    print(
        f"output: shape={output_info['shape'].tolist()} "
        f"dtype={output_info['dtype'].__name__} "
        f"scale={output_scale:.8g} zero_point={output_zero_point}"
    )
    print("top-5 predictions:")
    for rank, class_index in enumerate(top5, start=1):
        print(
            f"  {rank}. class={class_index:3d} "
            f"probability={probabilities[class_index]:.6f} "
            f"label={labels[int(class_index)]}"
        )


if __name__ == "__main__":
    main()
