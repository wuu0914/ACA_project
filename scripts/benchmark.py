#!/usr/bin/env python3
"""Benchmark and compare EfficientNet-Lite0 reference models.

The subcommands intentionally run in separate uv environments:

* ``tflite`` uses ai-edge-litert.
* ``pytorch`` uses torch + timm.
* ``compare`` consumes their JSON outputs and writes the experiment report.

All backends share the preprocessing code in this file so model differences are
not confused with resize, crop, layout, or normalization differences.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = ROOT / "models/testdata/panda.jpg"
DEFAULT_LABELS = ROOT / "models/testdata/labels_map.txt"
DEFAULT_FP32 = ROOT / "models/upstream/efficientnet-lite0-fp32.tflite"
DEFAULT_INT8 = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
DEFAULT_WEIGHTS = ROOT / "models/upstream/tf_efficientnet_lite0-0aa007d2.pth"
DEFAULT_TFLITE_RESULTS = ROOT / "build/phase1_1_tflite.json"
DEFAULT_PYTORCH_RESULTS = ROOT / "build/phase1_1_pytorch.json"
DEFAULT_REPORT = ROOT / "build/phase1_1_report.md"
DEFAULT_IMAGENET_ROOT = ROOT / "models/dataset/imagenet"
DEFAULT_EVAL_TFLITE = ROOT / "build/phase1_1_eval_tflite.json"
DEFAULT_EVAL_PYTORCH = ROOT / "build/phase1_1_eval_pytorch.json"
DEFAULT_EVAL_REPORT = ROOT / "build/phase1_1_eval_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path) -> dict[int, str]:
    labels = ast.literal_eval(path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or set(labels) != set(range(1000)):
        raise ValueError(f"expected labels 0..999 in {path}")
    return labels


def preprocess_image(path: Path, include_fingerprint: bool = True) -> tuple[Any, dict[str, Any]]:
    """Return normalized NHWC float32 input and a preprocessing record."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    original_size = list(image.size)
    crop_fraction = 0.875
    crop_size = int(crop_fraction * min(image.size))
    left = ((image.width - crop_size) + 1) // 2
    top = ((image.height - crop_size) + 1) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32)
    normalized = ((pixels - 127.0) / 128.0)[None, ...]
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    record = {
        "image": display_path,
        "original_size_wh": original_size,
        "center_crop_fraction": crop_fraction,
        "source_crop_size_hw": [crop_size, crop_size],
        "interpolation": "Pillow bicubic",
        "resize_target_hw": [224, 224],
        "normalization": "float32((pixel - 127.0) / 128.0)",
    }
    if include_fingerprint:
        record["image_sha256"] = sha256(path)
        record["normalized_nhwc_sha256"] = hashlib.sha256(normalized.tobytes()).hexdigest()
    return normalized, record


def latency_stats(function: Callable[[], Any], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return summarize_latencies(samples, warmup)


def summarize_latencies(samples: list[float], warmup: int = 0) -> dict[str, Any]:
    ordered = sorted(samples)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "scope": "model invocation only; preprocessing and input copy excluded",
        "warmup_runs": warmup,
        "measured_runs": len(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": p95,
    }


def top_k(probabilities: Any, labels: dict[int, str], count: int = 5) -> list[dict[str, Any]]:
    import numpy as np

    indices = np.argsort(probabilities)[-count:][::-1]
    return [
        {"index": int(index), "label": labels[int(index)], "probability": float(probabilities[index])}
        for index in indices
    ]


def load_imagenet_sample(dataset_root: Path, limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic nested subset of the 50,000-image validation set."""
    val_file = dataset_root / "aux/val.txt"
    synsets_file = dataset_root / "aux/synsets.txt"
    image_root = dataset_root / "val"
    if not val_file.is_file() or not synsets_file.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"{dataset_root} must contain aux/val.txt, aux/synsets.txt, and val/"
        )

    synsets = [line.strip() for line in synsets_file.read_text(encoding="utf-8").splitlines()]
    if len(synsets) != 1000:
        raise ValueError(f"expected 1000 synsets, got {len(synsets)}")
    entries = []
    for line in val_file.read_text(encoding="utf-8").splitlines():
        filename, label_text = line.split()
        label = int(label_text)
        wnid = synsets[label]
        path = image_root / wnid / filename
        entries.append(
            {
                "filename": filename,
                "label": label,
                "wnid": wnid,
                "relative_path": f"{wnid}/{filename}",
                "path": path,
            }
        )
    if len(entries) != 50_000:
        raise ValueError(f"expected 50000 validation entries, got {len(entries)}")
    if not 1 <= limit <= len(entries):
        raise ValueError(f"--limit must be in [1, {len(entries)}]")

    random.Random(seed).shuffle(entries)
    selected = entries[:limit]
    missing = [str(entry["path"]) for entry in selected if not entry["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"missing ImageNet image: {missing[0]}")
    sample_text = "".join(
        f"{entry['relative_path']}\t{entry['label']}\n" for entry in selected
    ).encode()
    metadata = {
        "root": str(dataset_root),
        "validation_entries": len(entries),
        "sample_count": limit,
        "selection": "shuffle all val.txt entries with Python random.Random(seed), then take prefix",
        "seed": seed,
        "sample_manifest_sha256": hashlib.sha256(sample_text).hexdigest(),
        "val_txt_sha256": sha256(val_file),
        "synsets_txt_sha256": sha256(synsets_file),
        "preprocessing": {
            "center_crop_fraction": 0.875,
            "interpolation": "Pillow bicubic",
            "resize_target_hw": [224, 224],
            "normalization": "float32((pixel - 127.0) / 128.0)",
        },
    }
    return selected, metadata


def accuracy_record(correct1: int, correct5: int, count: int) -> dict[str, Any]:
    top1 = 100.0 * correct1 / count
    top5 = 100.0 * correct5 / count
    # A simple normal-approximation interval is included to make the uncertainty
    # of 100- and 1000-image subsets visible; it is not an official ImageNet score.
    p = correct1 / count
    margin = 100.0 * 1.96 * math.sqrt(p * (1.0 - p) / count)
    return {
        "sample_count": count,
        "top1_correct": correct1,
        "top1_percent": top1,
        "top1_approx_95_percent_margin_pp": margin,
        "top5_correct": correct5,
        "top5_percent": top5,
    }


def prediction_record(entry: dict[str, Any], probabilities: Any) -> dict[str, Any]:
    import numpy as np

    top5 = np.argsort(probabilities)[-5:][::-1]
    return {
        "image": entry["relative_path"],
        "target": entry["label"],
        "top5": [int(index) for index in top5],
        "top1_probability": float(probabilities[top5[0]]),
    }


def tensor_record(detail: dict[str, Any]) -> dict[str, Any]:
    quantization = detail["quantization_parameters"]
    return {
        "name": detail["name"],
        "shape": detail["shape"].tolist(),
        "dtype": detail["dtype"].__name__,
        "scales": quantization["scales"].tolist(),
        "zero_points": quantization["zero_points"].tolist(),
        "quantized_dimension": int(quantization["quantized_dimension"]),
    }


def inspect_tflite(interpreter: Any) -> dict[str, Any]:
    import numpy as np

    tensors = interpreter.get_tensor_details()
    tensor_by_index = {detail["index"]: detail for detail in tensors}
    operations = interpreter._get_ops_details()  # LiteRT exposes no public operator-list API.
    operator_counts = Counter(operation["op_name"] for operation in operations)
    dtype_counts = Counter(detail["dtype"].__name__ for detail in tensors)
    float_tensors = [
        detail["name"] for detail in tensors if np.issubdtype(detail["dtype"], np.floating)
    ]

    weight_checks = []
    bias_checks = []
    for operation in operations:
        if operation["op_name"] not in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
            continue
        weight_index = int(operation["inputs"][1])
        weight = tensor_by_index[weight_index]
        zero_points = weight["quantization_parameters"]["zero_points"]
        weight_checks.append(
            weight["dtype"] == np.int8
            and zero_points.size > 0
            and bool(np.all(zero_points == 0))
        )
        bias_index = int(operation["inputs"][2])
        if bias_index >= 0:
            bias_checks.append(tensor_by_index[bias_index]["dtype"] == np.int32)

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    integer_io = np.issubdtype(input_detail["dtype"], np.integer) and np.issubdtype(
        output_detail["dtype"], np.integer
    )
    return {
        "input": tensor_record(input_detail),
        "output": tensor_record(output_detail),
        "tensor_count": len(tensors),
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "operator_count": len(operations),
        "operator_counts": dict(sorted(operator_counts.items())),
        "float_tensor_count": len(float_tensors),
        "quantized_compute_weight_checks": len(weight_checks),
        "quantized_compute_bias_checks": len(bias_checks),
        "all_compute_weights_int8_symmetric": bool(weight_checks) and all(weight_checks),
        "all_compute_biases_int32": bool(bias_checks) and all(bias_checks),
        "integer_only_verified": (
            integer_io
            and not float_tensors
            and bool(weight_checks)
            and all(weight_checks)
            and bool(bias_checks)
            and all(bias_checks)
        ),
    }


def run_tflite_model(
    model_path: Path,
    normalized_input: Any,
    labels: dict[int, str],
    threads: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    inspection = inspect_tflite(interpreter)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] == np.float32:
        model_input = normalized_input
    else:
        scale, zero_point = input_detail["quantization"]
        limits = np.iinfo(input_detail["dtype"])
        model_input = np.clip(
            np.rint(normalized_input / scale + zero_point), limits.min, limits.max
        ).astype(input_detail["dtype"])

    interpreter.set_tensor(input_detail["index"], model_input)
    latency = latency_stats(interpreter.invoke, warmup, repeats)
    raw_output = interpreter.get_tensor(output_detail["index"])
    if output_detail["dtype"] == np.float32:
        probabilities = raw_output.astype(np.float32)[0]
    else:
        scale, zero_point = output_detail["quantization"]
        probabilities = ((raw_output.astype(np.float32) - zero_point) * scale)[0]

    return {
        "path": str(model_path.relative_to(ROOT)),
        "sha256": sha256(model_path),
        "size_bytes": model_path.stat().st_size,
        "inspection": inspection,
        "latency": latency,
        "probability_sum": float(probabilities.sum()),
        "top5": top_k(probabilities, labels),
        "probabilities": probabilities.tolist(),
    }


def command_tflite(args: argparse.Namespace) -> None:
    from ai_edge_litert import __version__ as litert_version

    normalized_input, preprocessing = preprocess_image(args.image)
    labels = load_labels(args.labels)
    result = {
        "schema_version": 1,
        "backend": "LiteRT",
        "environment": {
            "python": platform.python_version(),
            "ai_edge_litert": litert_version,
            "threads": args.threads,
        },
        "preprocessing": preprocessing,
        "models": {
            "tflite_fp32": run_tflite_model(
                args.fp32, normalized_input, labels, args.threads, args.warmup, args.repeats
            ),
            "tflite_int8": run_tflite_model(
                args.int8, normalized_input, labels, args.threads, args.warmup, args.repeats
            ),
        },
    }
    write_json(args.output, result)


def command_pytorch(args: argparse.Namespace) -> None:
    import numpy as np
    import timm
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    normalized_input, preprocessing = preprocess_image(args.image)
    labels = load_labels(args.labels)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized_input.transpose(0, 3, 1, 2)))
    model = timm.create_model("tf_efficientnet_lite0", pretrained=False)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    def invoke() -> Any:
        with torch.inference_mode():
            return model(tensor)

    latency = latency_stats(invoke, args.warmup, args.repeats)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()

    result = {
        "schema_version": 1,
        "backend": "PyTorch/timm",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "threads": args.threads,
        },
        "preprocessing": preprocessing,
        "models": {
            "pytorch_fp32": {
                "path": str(args.weights.relative_to(ROOT)),
                "sha256": sha256(args.weights),
                "size_bytes": args.weights.stat().st_size,
                "architecture": "timm:tf_efficientnet_lite0",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "input_layout": "NCHW",
                "output": "logits followed by softmax in this script",
                "latency": latency,
                "probability_sum": float(probabilities.sum()),
                "top5": top_k(probabilities, labels),
                "probabilities": probabilities.tolist(),
            }
        },
    }
    write_json(args.output, result)


def command_evaluate_tflite(args: argparse.Namespace) -> None:
    import numpy as np
    from ai_edge_litert import __version__ as litert_version
    from ai_edge_litert.interpreter import Interpreter

    entries, dataset = load_imagenet_sample(args.dataset_root, args.limit, args.seed)
    interpreters = {
        "tflite_fp32": Interpreter(model_path=str(args.fp32), num_threads=args.threads),
        "tflite_int8": Interpreter(model_path=str(args.int8), num_threads=args.threads),
    }
    states = {}
    for name, interpreter in interpreters.items():
        inspection = inspect_tflite(interpreter)
        interpreter.allocate_tensors()
        states[name] = {
            "interpreter": interpreter,
            "input": interpreter.get_input_details()[0],
            "output": interpreter.get_output_details()[0],
            "inspection": inspection,
            "latencies": [],
            "predictions": [],
            "correct1": 0,
            "correct5": 0,
        }

    first_input, _ = preprocess_image(entries[0]["path"])
    for state in states.values():
        detail = state["input"]
        if detail["dtype"] == np.float32:
            model_input = first_input
        else:
            scale, zero_point = detail["quantization"]
            limits = np.iinfo(detail["dtype"])
            model_input = np.clip(
                np.rint(first_input / scale + zero_point), limits.min, limits.max
            ).astype(detail["dtype"])
        state["interpreter"].set_tensor(detail["index"], model_input)
        for _ in range(args.warmup):
            state["interpreter"].invoke()

    for number, entry in enumerate(entries, start=1):
        normalized, _ = preprocess_image(entry["path"], include_fingerprint=False)
        for state in states.values():
            detail = state["input"]
            if detail["dtype"] == np.float32:
                model_input = normalized
            else:
                scale, zero_point = detail["quantization"]
                limits = np.iinfo(detail["dtype"])
                model_input = np.clip(
                    np.rint(normalized / scale + zero_point), limits.min, limits.max
                ).astype(detail["dtype"])
            interpreter = state["interpreter"]
            interpreter.set_tensor(detail["index"], model_input)
            start = time.perf_counter_ns()
            interpreter.invoke()
            state["latencies"].append((time.perf_counter_ns() - start) / 1_000_000)
            output_detail = state["output"]
            raw = interpreter.get_tensor(output_detail["index"])
            if output_detail["dtype"] == np.float32:
                probabilities = raw.astype(np.float32)[0]
            else:
                scale, zero_point = output_detail["quantization"]
                probabilities = ((raw.astype(np.float32) - zero_point) * scale)[0]
            prediction = prediction_record(entry, probabilities)
            state["predictions"].append(prediction)
            state["correct1"] += prediction["top5"][0] == entry["label"]
            state["correct5"] += entry["label"] in prediction["top5"]
        if number % max(100, args.limit // 10) == 0 or number == args.limit:
            print(f"LiteRT evaluated {number}/{args.limit} images")

    models = {}
    for name, state in states.items():
        path = args.fp32 if name == "tflite_fp32" else args.int8
        models[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "inspection": state["inspection"],
            "accuracy": accuracy_record(state["correct1"], state["correct5"], args.limit),
            "latency": summarize_latencies(state["latencies"], args.warmup),
            "predictions": state["predictions"],
        }
    write_json(
        args.output,
        {
            "schema_version": 1,
            "backend": "LiteRT",
            "environment": {
                "python": platform.python_version(),
                "ai_edge_litert": litert_version,
                "threads": args.threads,
            },
            "dataset": dataset,
            "models": models,
        },
    )


def command_evaluate_pytorch(args: argparse.Namespace) -> None:
    import numpy as np
    import timm
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    entries, dataset = load_imagenet_sample(args.dataset_root, args.limit, args.seed)
    model = timm.create_model("tf_efficientnet_lite0", pretrained=False)
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()

    first_input, _ = preprocess_image(entries[0]["path"])
    first_tensor = torch.from_numpy(np.ascontiguousarray(first_input.transpose(0, 3, 1, 2)))
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(first_tensor)

    latencies = []
    predictions = []
    correct1 = 0
    correct5 = 0
    for number, entry in enumerate(entries, start=1):
        normalized, _ = preprocess_image(entry["path"], include_fingerprint=False)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2)))
        start = time.perf_counter_ns()
        with torch.inference_mode():
            logits = model(tensor)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
        prediction = prediction_record(entry, probabilities)
        predictions.append(prediction)
        correct1 += prediction["top5"][0] == entry["label"]
        correct5 += entry["label"] in prediction["top5"]
        if number % max(100, args.limit // 10) == 0 or number == args.limit:
            print(f"PyTorch evaluated {number}/{args.limit} images")

    write_json(
        args.output,
        {
            "schema_version": 1,
            "backend": "PyTorch/timm",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
                "threads": args.threads,
            },
            "dataset": dataset,
            "models": {
                "pytorch_fp32": {
                    "path": str(args.weights.relative_to(ROOT)),
                    "sha256": sha256(args.weights),
                    "architecture": "timm:tf_efficientnet_lite0",
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "accuracy": accuracy_record(correct1, correct5, args.limit),
                    "latency": summarize_latencies(latencies, args.warmup),
                    "predictions": predictions,
                }
            },
        },
    )


def compare_classification_predictions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_predictions = left["predictions"]
    right_predictions = right["predictions"]
    if len(left_predictions) != len(right_predictions):
        raise ValueError("prediction counts differ")
    top1_agree = 0
    both_correct = 0
    left_only_correct = 0
    right_only_correct = 0
    top5_overlap_total = 0
    for a, b in zip(left_predictions, right_predictions):
        if a["image"] != b["image"] or a["target"] != b["target"]:
            raise ValueError("prediction sample order differs")
        target = a["target"]
        a_correct = a["top5"][0] == target
        b_correct = b["top5"][0] == target
        top1_agree += a["top5"][0] == b["top5"][0]
        both_correct += a_correct and b_correct
        left_only_correct += a_correct and not b_correct
        right_only_correct += b_correct and not a_correct
        top5_overlap_total += len(set(a["top5"]) & set(b["top5"]))
    count = len(left_predictions)
    discordant = left_only_correct + right_only_correct
    if discordant:
        smaller = min(left_only_correct, right_only_correct)
        # Evaluate the exact binomial tail in log space.  Multiplying
        # comb(n, k) by 0.5**n directly overflows once the full 50k-image
        # validation set produces thousands of discordant pairs.
        log_largest_term = (
            math.lgamma(discordant + 1)
            - math.lgamma(smaller + 1)
            - math.lgamma(discordant - smaller + 1)
            - discordant * math.log(2.0)
        )
        scaled_tail = 1.0
        scaled_term = 1.0
        for k in range(smaller, 0, -1):
            scaled_term *= k / (discordant - k + 1)
            scaled_tail += scaled_term
        mcnemar_p = min(1.0, 2.0 * math.exp(log_largest_term) * scaled_tail)
    else:
        mcnemar_p = 1.0
    return {
        "sample_count": count,
        "top1_agree": top1_agree,
        "top1_agreement_percent": 100.0 * top1_agree / count,
        "mean_top5_overlap": top5_overlap_total / count,
        "both_top1_correct": both_correct,
        "left_only_top1_correct": left_only_correct,
        "right_only_top1_correct": right_only_correct,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "top1_accuracy_delta_right_minus_left_pp": (
            right["accuracy"]["top1_percent"] - left["accuracy"]["top1_percent"]
        ),
        "top5_accuracy_delta_right_minus_left_pp": (
            right["accuracy"]["top5_percent"] - left["accuracy"]["top5_percent"]
        ),
    }


def render_imagenet_report(
    dataset: dict[str, Any], models: dict[str, Any], comparisons: dict[str, Any]
) -> str:
    count = dataset["sample_count"]
    is_full_validation = count == dataset["validation_entries"]
    lines = [
        f"# Phase 1-1 ImageNet validation ({count} images)",
        "",
        (
            "This evaluates the complete 50,000-image ILSVRC2012 validation set."
            if is_full_validation
            else "This is a deterministic subset measurement, not the official 50,000-image score."
        ),
        "",
        "## Dataset and method",
        "",
        f"- Dataset root: `{dataset['root']}`",
        f"- Selection seed: `{dataset['seed']}`",
        f"- Sample manifest SHA256: `{dataset['sample_manifest_sha256']}`",
        "- Batch size: 1; one CPU thread",
        "- Shared preprocessing: center-crop 87.5% of the shorter side, bicubic resize to 224, then `(pixel - 127) / 128`",
        "",
        (
            "All validation images are included; the seeded order only controls traversal."
            if is_full_validation
            else "The 100-image sample is the prefix of the 1000-image sample because both are"
        ),
        *( [] if is_full_validation else ["taken from the same seeded shuffle. This makes the smoke test directly traceable."] ),
        "",
        "## Accuracy",
        "",
        "| Model | Top-1 | Top-5 | Approx. Top-1 95% margin | Median inference |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("pytorch_fp32", "tflite_fp32", "tflite_int8"):
        accuracy = models[name]["accuracy"]
        latency = models[name]["latency"]
        lines.append(
            f"| {name} | {accuracy['top1_correct']}/{count} ({accuracy['top1_percent']:.2f}%) | "
            f"{accuracy['top5_correct']}/{count} ({accuracy['top5_percent']:.2f}%) | "
            f"±{accuracy['top1_approx_95_percent_margin_pp']:.2f} pp | {latency['median_ms']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            "## Pairwise answers",
            "",
            "| Question | Top-1 agreement | Mean top-5 overlap | Top-1 accuracy delta | Left-only / right-only correct | McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "pytorch_fp32_vs_tflite_fp32": "Does conversion change results? (PyTorch → TFLite FP32)",
        "tflite_fp32_vs_tflite_int8": "Does quantization reduce classification? (FP32 → INT8)",
    }
    for name, metrics in comparisons.items():
        lines.append(
            f"| {labels[name]} | {metrics['top1_agree']}/{count} ({metrics['top1_agreement_percent']:.2f}%) | "
            f"{metrics['mean_top5_overlap']:.3f}/5 | {metrics['top1_accuracy_delta_right_minus_left_pp']:+.2f} pp | "
            f"{metrics['left_only_top1_correct']} / {metrics['right_only_top1_correct']} | "
            f"{metrics['mcnemar_exact_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "This is the full validation set; compare the result with Google's published score while keeping preprocessing/runtime versions explicit."
                if is_full_validation
                else "A small subset has sampling uncertainty: read the accuracy delta together with the agreement and left-only/right-only counts."
            ),
            "Host latency is only a runtime",
            "sanity measurement and is not an RV64IMC architecture result.",
            "",
        ]
    )
    return "\n".join(lines)


def command_compare_evaluation(args: argparse.Namespace) -> None:
    tflite = json.loads(args.tflite_results.read_text(encoding="utf-8"))
    pytorch = json.loads(args.pytorch_results.read_text(encoding="utf-8"))
    if tflite["dataset"] != pytorch["dataset"]:
        raise ValueError("dataset metadata differs; refusing an invalid comparison")
    models = {**pytorch["models"], **tflite["models"]}
    comparisons = {
        "pytorch_fp32_vs_tflite_fp32": compare_classification_predictions(
            models["pytorch_fp32"], models["tflite_fp32"]
        ),
        "tflite_fp32_vs_tflite_int8": compare_classification_predictions(
            models["tflite_fp32"], models["tflite_int8"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_imagenet_report(tflite["dataset"], models, comparisons), encoding="utf-8"
    )
    print(f"Wrote {args.output.relative_to(ROOT)}")


def comparison(a: list[float], b: list[float]) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    difference = np.abs(left - right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    left_top5 = set(np.argsort(left)[-5:].tolist())
    right_top5 = set(np.argsort(right)[-5:].tolist())
    return {
        "same_top1": bool(left.argmax() == right.argmax()),
        "top5_overlap": len(left_top5 & right_top5),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "cosine_similarity": float(np.dot(left, right) / denominator),
    }


def render_report(all_models: dict[str, Any], comparisons: dict[str, Any], preprocessing: dict[str, Any]) -> str:
    fp32_audit = all_models["tflite_fp32"]["inspection"]
    int8_audit = all_models["tflite_int8"]["inspection"]
    lines = [
        "# Phase 1-1 measurement report",
        "",
        "This file is generated by `make phase1-1-run`. Latency is a host-side sanity",
        "measurement, not a Spike or CV-Wally performance result.",
        "",
        "## Reproducibility",
        "",
        f"- Input: `{preprocessing['image']}` (`{preprocessing['image_sha256']}`)",
        f"- Crop: centered {preprocessing['center_crop_fraction'] * 100:.1f}% of the shorter side ({preprocessing['source_crop_size_hw'][0]} x {preprocessing['source_crop_size_hw'][1]} for this image)",
        "- Resize: cropped square to 224 x 224 with bicubic interpolation",
        "- Normalization: `(pixel - 127.0) / 128.0`",
        f"- Processed input fingerprint: `{preprocessing['normalized_nhwc_sha256']}`",
        "- Threads: 1 for every runtime",
        "",
        "## Model audit",
        "",
        "| Model | Size | Parameters/tensors | Ops | Float tensors | Integer-only |",
        "|---|---:|---:|---:|---:|:---:|",
        f"| PyTorch FP32 | {all_models['pytorch_fp32']['size_bytes'] / 1_000_000:.2f} MB | {all_models['pytorch_fp32']['parameter_count']:,} params | — | — | No |",
        f"| TFLite FP32 | {all_models['tflite_fp32']['size_bytes'] / 1_000_000:.2f} MB | {fp32_audit['tensor_count']} tensors | {fp32_audit['operator_count']} | {fp32_audit['float_tensor_count']} | No |",
        f"| TFLite INT8 | {all_models['tflite_int8']['size_bytes'] / 1_000_000:.2f} MB | {int8_audit['tensor_count']} tensors | {int8_audit['operator_count']} | {int8_audit['float_tensor_count']} | {'Yes' if int8_audit['integer_only_verified'] else 'NO'} |",
        "",
        f"The INT8 audit checked {int8_audit['quantized_compute_weight_checks']} Conv/DepthwiseConv/FullyConnected weights and {int8_audit['quantized_compute_bias_checks']} biases: weights are signed INT8 with zero-point 0 and biases are INT32. No FLOAT tensor exists; TFLite's quantized kernels use INT32 accumulation.",
        "",
        "## Prediction on the fixed panda image",
        "",
        "| Model | Top-1 class | Probability | Probability sum |",
        "|---|---|---:|---:|",
    ]
    for name in ("pytorch_fp32", "tflite_fp32", "tflite_int8"):
        model = all_models[name]
        winner = model["top5"][0]
        lines.append(
            f"| {name} | {winner['index']}: {winner['label']} | {winner['probability']:.8f} | {model['probability_sum']:.8f} |"
        )

    lines.extend(
        [
            "",
            "The UINT8 softmax uses steps of 1/256. Small probabilities round to zero,",
            "so its dequantized values need not sum to exactly 1.0.",
            "",
            "## Numerical comparison",
            "",
            "| Pair | Same top-1 | Top-5 overlap | Mean abs. error | Max abs. error | Cosine similarity |",
            "|---|:---:|---:|---:|---:|---:|",
        ]
    )
    for pair_name, metrics in comparisons.items():
        lines.append(
            f"| {pair_name.replace('_vs_', ' vs ')} | {'Yes' if metrics['same_top1'] else 'No'} | {metrics['top5_overlap']}/5 | {metrics['mean_absolute_error']:.3e} | {metrics['max_absolute_error']:.3e} | {metrics['cosine_similarity']:.9f} |"
        )

    lines.extend(
        [
            "",
            "## Host latency (single thread)",
            "",
            "| Model | Median | Mean | P95 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("pytorch_fp32", "tflite_fp32", "tflite_int8"):
        latency = all_models[name]["latency"]
        lines.append(
            f"| {name} | {latency['median_ms']:.3f} ms | {latency['mean_ms']:.3f} ms | {latency['p95_ms']:.3f} ms |"
        )
    lines.extend(
        [
            "",
            "Do not use this latency table as the architecture speedup result: PyTorch and",
            "LiteRT use different host kernels, while the project target is bare-metal RV64IMC.",
            "The table only confirms that all three runnable references are healthy.",
            "",
        ]
    )
    return "\n".join(lines)


def command_compare(args: argparse.Namespace) -> None:
    tflite = json.loads(args.tflite_results.read_text(encoding="utf-8"))
    pytorch = json.loads(args.pytorch_results.read_text(encoding="utf-8"))
    if tflite["preprocessing"] != pytorch["preprocessing"]:
        raise ValueError("backend preprocessing records differ; refusing an invalid comparison")
    all_models = {**pytorch["models"], **tflite["models"]}
    comparisons = {
        "pytorch_fp32_vs_tflite_fp32": comparison(
            all_models["pytorch_fp32"]["probabilities"],
            all_models["tflite_fp32"]["probabilities"],
        ),
        "tflite_fp32_vs_tflite_int8": comparison(
            all_models["tflite_fp32"]["probabilities"],
            all_models["tflite_int8"]["probabilities"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(all_models, comparisons, tflite["preprocessing"]), encoding="utf-8"
    )
    print(f"Wrote {args.output.relative_to(ROOT)}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def path_argument(value: str) -> Path:
    return Path(value).resolve()


def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", type=path_argument, default=DEFAULT_IMAGE)
    parser.add_argument("--labels", type=path_argument, default=DEFAULT_LABELS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=path_argument, default=DEFAULT_IMAGENET_ROOT)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tflite = subparsers.add_parser("tflite", help="audit and run both TFLite models")
    add_common_run_arguments(tflite)
    tflite.add_argument("--fp32", type=path_argument, default=DEFAULT_FP32)
    tflite.add_argument("--int8", type=path_argument, default=DEFAULT_INT8)
    tflite.add_argument("--output", type=path_argument, default=DEFAULT_TFLITE_RESULTS)
    tflite.set_defaults(function=command_tflite)

    pytorch = subparsers.add_parser("pytorch", help="run the timm PyTorch port")
    add_common_run_arguments(pytorch)
    pytorch.add_argument("--weights", type=path_argument, default=DEFAULT_WEIGHTS)
    pytorch.add_argument("--output", type=path_argument, default=DEFAULT_PYTORCH_RESULTS)
    pytorch.set_defaults(function=command_pytorch)

    compare = subparsers.add_parser("compare", help="compare JSON results and write Markdown")
    compare.add_argument("--tflite-results", type=path_argument, default=DEFAULT_TFLITE_RESULTS)
    compare.add_argument("--pytorch-results", type=path_argument, default=DEFAULT_PYTORCH_RESULTS)
    compare.add_argument("--output", type=path_argument, default=DEFAULT_REPORT)
    compare.set_defaults(function=command_compare)

    eval_tflite = subparsers.add_parser(
        "evaluate-tflite", help="measure TFLite accuracy on an ImageNet validation subset"
    )
    add_evaluation_arguments(eval_tflite)
    eval_tflite.add_argument("--fp32", type=path_argument, default=DEFAULT_FP32)
    eval_tflite.add_argument("--int8", type=path_argument, default=DEFAULT_INT8)
    eval_tflite.add_argument("--output", type=path_argument, default=DEFAULT_EVAL_TFLITE)
    eval_tflite.set_defaults(function=command_evaluate_tflite)

    eval_pytorch = subparsers.add_parser(
        "evaluate-pytorch", help="measure PyTorch accuracy on an ImageNet validation subset"
    )
    add_evaluation_arguments(eval_pytorch)
    eval_pytorch.add_argument("--weights", type=path_argument, default=DEFAULT_WEIGHTS)
    eval_pytorch.add_argument("--output", type=path_argument, default=DEFAULT_EVAL_PYTORCH)
    eval_pytorch.set_defaults(function=command_evaluate_pytorch)

    compare_eval = subparsers.add_parser(
        "compare-evaluation", help="compare ImageNet subset evaluation JSON files"
    )
    compare_eval.add_argument(
        "--tflite-results", type=path_argument, default=DEFAULT_EVAL_TFLITE
    )
    compare_eval.add_argument(
        "--pytorch-results", type=path_argument, default=DEFAULT_EVAL_PYTORCH
    )
    compare_eval.add_argument("--output", type=path_argument, default=DEFAULT_EVAL_REPORT)
    compare_eval.set_defaults(function=command_compare_evaluation)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "threads") and args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if hasattr(args, "warmup") and args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if hasattr(args, "repeats") and (args.warmup < 0 or args.repeats < 1):
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
