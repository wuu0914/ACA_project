#!/usr/bin/env python3
"""
    A testbench for the complete C model against TFLite.
    Flow:
        Center-crop, resize, normalize and quantize the image
        Run the TFLite reference model as golden
        Run the C model with exactly the same uint8 input
        Compare the final output byte-for-byte

    Trace mode:
        Preserve every TFLite intermediate tensor
        Receive each C operator output through a callback
        Compare the two outputs before tensor arena memory is reused
"""

import argparse
import concurrent.futures
import ctypes as C
import hashlib
import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models/upstream/efficientnet-lite0-int8.tflite"
DEFAULT_LIBRARY = ROOT / "build/libefficientnet_lite0.so"
DEFAULT_IMAGE = ROOT / "models/testdata/panda.jpg"


def sha256(path):
    """Calculate a file fingerprint without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_image(path):
    """
    Prepare the NHWC float32 input expected before TFLite quantization.

    Flow:
        Convert image to RGB
        -> Center-crop 87.5% of the shorter side
        -> Resize the crop to 224 x 224 with bicubic interpolation
        -> Normalize each pixel with (pixel - 127) / 128, where -1.0 < normalized <= 1.0
        -> Add the batch dimension: [224, 224, 3] -> [1, 224, 224, 3]

    Notes:
        This is copied into Phase 1-4 so the testbench does not import Phase 1-1.
        Keeping the exact crop and normalization is important: a different input
        would make the two models agree with each other but test a different image.
    """
    image = Image.open(path).convert("RGB")
    crop_size = int(0.875 * min(image.size))
    left = ((image.width - crop_size) + 1) // 2
    top = ((image.height - crop_size) + 1) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32)
    return ((pixels - 127.0) / 128.0)[None, ...]


def load_imagenet_sample(dataset_root, limit, seed):
    """
    Select a reproducible subset from the 50,000 ImageNet validation images.

    Dataset layout:
        dataset_root/aux/val.txt       - filename and class index
        dataset_root/aux/synsets.txt   - class index to WordNet ID
        dataset_root/val/<wnid>/*.JPEG - validation images

    Selection:
        Shuffle all entries with random.Random(seed), then take the first limit.
        The same seed therefore keeps smaller and larger tests nested.
    """
    val_file = dataset_root / "aux/val.txt"
    synsets_file = dataset_root / "aux/synsets.txt"
    image_root = dataset_root / "val"
    if not val_file.is_file() or not synsets_file.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"{dataset_root} must contain aux/val.txt, aux/synsets.txt, and val/"
        )

    synsets = [
        line.strip()
        for line in synsets_file.read_text(encoding="utf-8").splitlines()
    ]
    if len(synsets) != 1000:
        raise ValueError(f"expected 1000 synsets, got {len(synsets)}")

    entries = []
    for line in val_file.read_text(encoding="utf-8").splitlines():
        filename, label_text = line.split()
        label = int(label_text)
        wnid = synsets[label]
        entries.append({
            "filename": filename,
            "label": label,
            "wnid": wnid,
            "relative_path": f"{wnid}/{filename}",
            "path": image_root / wnid / filename,
        })

    if len(entries) != 50_000:
        raise ValueError(f"expected 50000 validation entries, got {len(entries)}")
    if not 1 <= limit <= len(entries):
        raise ValueError(f"--limit must be in [1, {len(entries)}]")

    random.Random(seed).shuffle(entries)
    selected = entries[:limit]
    missing = [str(entry["path"]) for entry in selected if not entry["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"missing ImageNet image: {missing[0]}")

    # The report fingerprints both the dataset metadata and exact selected subset.
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


# C callback layout from efficientnet_lite0.h:
#   trace(operator_index, tensor_index, data, size, user_data)
# `data` points to this operator's output inside the C tensor arena.
TRACE_CALLBACK = C.CFUNCTYPE(
    None, C.c_int, C.c_int, C.c_void_p, C.c_size_t, C.c_void_p
)

# ProcessPool workers cannot share a TFLite interpreter or writable tensor arena.
# initialize_worker() gives every process its own state before it receives images.
WORKER_STATE = {}


def pointer(array, c_type):
    """Pass a contiguous NumPy array to C without copying its data."""
    return array.ctypes.data_as(C.POINTER(c_type))


def load_c_model(path):
    """
    Load the compiled C model and describe its function signatures to ctypes.

    Notes:
        ctypes does not read C headers. argtypes/restype are the Python copy of
        efficientnet_lite0.h and prevent pointers or size_t values from being
        interpreted as ordinary 32-bit integers.
    """
    library = C.CDLL(str(path))
    library.efficientnet_lite0_input_size.restype = C.c_size_t
    library.efficientnet_lite0_output_size.restype = C.c_size_t
    library.efficientnet_lite0_arena_size.restype = C.c_size_t
    library.efficientnet_lite0_invoke.argtypes = (
        C.POINTER(C.c_uint8),
        C.POINTER(C.c_uint8),
        C.c_void_p,
        C.c_size_t,
    )
    library.efficientnet_lite0_invoke.restype = C.c_int
    library.efficientnet_lite0_invoke_with_trace.argtypes = (
        C.POINTER(C.c_uint8),
        C.POINTER(C.c_uint8),
        C.c_void_p,
        C.c_size_t,
        TRACE_CALLBACK,
        C.c_void_p,
    )
    library.efficientnet_lite0_invoke_with_trace.restype = C.c_int
    return library


def aligned_arena(size):
    """
    Allocate `size` writable bytes whose first address is 16-byte aligned.

    Output
        storage: the truely allocated memory
        arena: the aligned 'size' byte memory for C program
    """
    # At most 15 leading bytes are skipped, 
    # so allocating size + 15 guarantees that a complete aligned slice is available.
    storage = np.empty(size + 15, dtype=np.uint8)
    offset = (-storage.ctypes.data) % 16
    arena = storage[offset:offset + size]
    assert arena.ctypes.data % 16 == 0
    return storage, arena


def quantize_input(normalized, input_detail):
    """
    Convert real float32 input into the TFLite model's integer input tensor.

    Math:
        real = scale * (quantized - zero_point)
        quantized = clamp(round(real / scale) + zero_point, qmin, qmax)
    """
    scale, zero_point = input_detail["quantization"]
    limits = np.iinfo(input_detail["dtype"])
    return np.ascontiguousarray(
        np.clip(np.rint(normalized / scale + zero_point), limits.min, limits.max)
        .astype(input_detail["dtype"])
    )


def first_difference(ours, tflite):
    """Describe the mismatch count and first different byte."""
    different = np.flatnonzero(ours.reshape(-1) != tflite.reshape(-1))
    first = int(different[0])
    return (
        f"{len(different)} byte(s) differ; first flat index {first}: "
        f"C={int(ours.reshape(-1)[first])}, "
        f"TFLite={int(tflite.reshape(-1)[first])}"
    )


def preserved_outputs(interpreter):
    """
    Copy every TFLite operator output after one complete inference.

    Notes:
        _get_ops_details() is LiteRT's private operator-list API.
        experimental_preserve_all_tensors=True keeps intermediate tensors valid.
        copy() makes each golden output independent from LiteRT tensor storage.
    """
    operations = interpreter._get_ops_details()
    outputs = []
    for op_index, operation in enumerate(operations):
        tensor_indices = [int(value) for value in operation["outputs"]]
        if len(tensor_indices) != 1:
            raise RuntimeError(f"operator {op_index} has {len(tensor_indices)} outputs")
        tensor = np.ascontiguousarray(interpreter.get_tensor(tensor_indices[0]))
        outputs.append(tensor.view(np.uint8).reshape(-1).copy())
    return operations, outputs


def invoke_c(library, model_input, arena, output_size, trace_tflite):
    """
    Run one complete C inference, optionally comparing every operator output.

    Without trace:
        Call efficientnet_lite0_invoke() and return the final uint8[1000].

    With trace:
        C calls trace() immediately after every operator. Python views that C
        memory without copying and compares it with the preserved TFLite output
        before a later operator is allowed to reuse the same arena address.
    """
    output = np.empty(output_size, dtype=np.uint8)
    trace_error = []

    if trace_tflite is not None:
        operations, tflite_outputs = trace_tflite

        @TRACE_CALLBACK
        def trace(operator_index, tensor_index, data, size, _user_data):
            # Keep only the first error, but let C finish safely.
            if trace_error:
                return

            # Both generated models must refer to the same output tensor index.
            operation = operations[operator_index]
            tflite_tensor = int(operation["outputs"][0])
            if tensor_index != tflite_tensor:
                trace_error.append(
                    f"operator {operator_index}: C output tensor {tensor_index}, "
                    f"TFLite output tensor {tflite_tensor}"
                )
                return
            # Create a temporary NumPy view over C memory; no bytes are copied.
            ours = np.ctypeslib.as_array(
                (C.c_uint8 * size).from_address(int(data))
            )
            tflite = tflite_outputs[operator_index]
            if ours.size != tflite.size:
                trace_error.append(
                    f"operator {operator_index} {operation['op_name']}: "
                    f"C has {ours.size} bytes, TFLite has {tflite.size}"
                )
            elif not np.array_equal(ours, tflite):
                trace_error.append(
                    f"operator {operator_index} {operation['op_name']} tensor "
                    f"{tensor_index}: {first_difference(ours, tflite)}"
                )

        start = time.perf_counter_ns()
        # Run C model with intermediate output trace
        status = library.efficientnet_lite0_invoke_with_trace(
            pointer(model_input, C.c_uint8), pointer(output, C.c_uint8),
            C.c_void_p(arena.ctypes.data), arena.size, trace, None,
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    else:
        start = time.perf_counter_ns()
        # Run complete C model
        status = library.efficientnet_lite0_invoke(
            pointer(model_input, C.c_uint8), pointer(output, C.c_uint8),
            C.c_void_p(arena.ctypes.data), arena.size,
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    if status != 0:
        raise RuntimeError(f"C model returned status {status}")
    if trace_error:
        raise AssertionError(trace_error[0])
    return output, elapsed_ms


def image_entries(args):
    """Use the fixed panda image or a deterministic ImageNet subset."""
    if args.dataset_root is None:
        return ([{"path": args.image, "relative_path": str(args.image)}],
                {"kind": "fixed image", "count": 1})
    entries, metadata = load_imagenet_sample(args.dataset_root, args.limit, args.seed)
    return entries, metadata


def write_report(path, report):
    """Write optional machine-readable evidence from this comparison run."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {path}")


def initialize_worker(model_path, library_path):
    """
    Prepare one independent TFLite golden and C arena in a worker process.

    The ProcessPool calls this once per process, not once per image.
    """
    interpreter = Interpreter(
        model_path=model_path,
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    library = load_c_model(Path(library_path))
    arena_size = int(library.efficientnet_lite0_arena_size())
    arena_storage, arena = aligned_arena(arena_size)
    WORKER_STATE.update({
        "interpreter": interpreter,
        "input": interpreter.get_input_details()[0],
        "output": interpreter.get_output_details()[0],
        "library": library,
        "arena_storage": arena_storage,
        "arena": arena,
        "output_size": int(library.efficientnet_lite0_output_size()),
    })


def compare_image_worker(task):
    """Compare one image inside an already initialized worker process."""
    image_path, display_path = task
    state = WORKER_STATE

    # Input Preparation
    normalized = preprocess_image(Path(image_path))
    model_input = quantize_input(normalized, state["input"])

    # Run TFLite
    interpreter = state["interpreter"]
    interpreter.set_tensor(state["input"]["index"], model_input)
    start = time.perf_counter_ns()
    interpreter.invoke()
    tflite_ms = (time.perf_counter_ns() - start) / 1_000_000
    tflite = np.ascontiguousarray(
        interpreter.get_tensor(state["output"]["index"])
    ).view(np.uint8).reshape(-1)

    # Run C Model
    ours, c_ms = invoke_c(
        state["library"], model_input.reshape(-1), state["arena"],
        state["output_size"], None,
    )
    if not np.array_equal(ours, tflite):
        raise AssertionError(
            f"{display_path}: {first_difference(ours, tflite)}"
        )
    return tflite_ms, c_ms


def main():
    # Test Selection
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)    # single image
    parser.add_argument("--dataset-root", type=Path)                    
    parser.add_argument("--limit", type=int, default=100)               # select images in dataset
    parser.add_argument("--seed", type=int, default=20261008)           # how to select detaset
    parser.add_argument("--trace", action="store_true", help="compare every operator output (single image only)")   # False if not specified
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes for dataset comparison")            # # of processes
    parser.add_argument("--report", type=Path)                          # JSON report
    args = parser.parse_args()

    entries, dataset_metadata = image_entries(args)

    # Check: if the above parameters are legal
    if args.trace and len(entries) != 1:
        parser.error("--trace requires exactly one image")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.trace and args.jobs != 1:
        parser.error("--trace requires --jobs 1")

    # TFLite Golden Preparation
    # BUILTIN_REF: selects TFLite reference kernels instead of optimized kernels.
    # Trace mode: asks LiteRT not to release intermediate tensors after invoke().
    interpreter = Interpreter(
        model_path=str(args.model),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=args.trace,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    # C Model Preparation & Allocate arena
    library = load_c_model(args.library)
    input_size = int(library.efficientnet_lite0_input_size())
    output_size = int(library.efficientnet_lite0_output_size())
    arena_size = int(library.efficientnet_lite0_arena_size())
    _arena_storage, arena = aligned_arena(arena_size)

    # Check: I/O of the models
    if input_size != int(np.prod(input_detail["shape"])):
        raise RuntimeError("C and TFLite input sizes differ")
    if output_size != int(np.prod(output_detail["shape"])):
        raise RuntimeError("C and TFLite output sizes differ")

    tflite_times = []
    c_times = []
    progress_step = max(1, len(entries) // 10)
    comparison_start = time.perf_counter_ns()

    # Single Process Comparison
    if args.jobs == 1:
        # One image is fully processed before moving to the next image.
        for number, entry in enumerate(entries, start=1):
            # Input Preparation
            # Quantize float into uint8, then send to both models later.
            normalized = preprocess_image(entry["path"])
            model_input = quantize_input(normalized, input_detail)

            # Run TFLite
            # Only interpreter.invoke() is timed; image preprocessing is excluded.
            interpreter.set_tensor(input_detail["index"], model_input)
            start = time.perf_counter_ns()
            interpreter.invoke()
            tflite_times.append((time.perf_counter_ns() - start) / 1_000_000)
            tflite = np.ascontiguousarray(interpreter.get_tensor(output_detail["index"])
                     ).view(np.uint8).reshape(-1)
            trace_tflite = preserved_outputs(interpreter) if args.trace else None   # without trace, None

            # Run C Model
            # compares each C operator as it executes. 
            # Without --trace, only the final uint8[1000] output is compared below.
            ours, c_time = invoke_c(library, model_input.reshape(-1), arena, output_size, trace_tflite)
            c_times.append(c_time)
            if not np.array_equal(ours, tflite):
                raise AssertionError(
                    f"{entry['relative_path']}: {first_difference(ours, tflite)}"
                )

            if number % progress_step == 0 or number == len(entries):
                print(f"PASS  {number}/{len(entries)} images byte-exact", flush=True)

    # Multiple Process Comparison, no trace
    else:
        # Each worker owns one TFLite interpreter and one C tensor arena.
        # Workers process different images, never parts of one inference.
        tasks = [(str(entry["path"]), entry["relative_path"]) for entry in entries] # seperate images

        context = multiprocessing.get_context("spawn")                                                    
        with concurrent.futures.ProcessPoolExecutor(        # Generate process pool with args.jobs workers.
            max_workers=args.jobs,
            mp_context=context,
            initializer=initialize_worker,                  # initialize workers by giving independent TFLite Interpreter, C library, arena, etc. 
            initargs=(str(args.model), str(args.library)),
        ) as executor:
            results = executor.map(compare_image_worker, tasks, chunksize=1)    # Seperate tasks for workers to compare C and TFLite outputs.
            for number, (tflite_ms, c_ms) in enumerate(results, start=1):
                tflite_times.append(tflite_ms)
                c_times.append(c_ms)
                if number % progress_step == 0 or number == len(entries):
                    print(f"PASS  {number}/{len(entries)} images byte-exact", flush=True)
    wall_seconds = (time.perf_counter_ns() - comparison_start) / 1_000_000_000

    # Report
    # Per-model means measure inference calls; wall time also includes image
    # preprocessing, byte comparison and multiprocessing overhead.
    report = {
        "schema_version": 1,
        "model": str(args.model),
        "library": str(args.library),
        "dataset": dataset_metadata,
        "images_compared": len(entries),
        "bytes_compared_per_image": output_size,
        "all_outputs_byte_exact": True,
        "trace_all_operator_outputs": bool(args.trace),
        "worker_processes": args.jobs,
        "tensor_arena_bytes": arena_size,
        "comparison_wall_seconds": wall_seconds,
        "mean_tflite_reference_ms": float(np.mean(tflite_times)),
        "mean_c_model_ms": float(np.mean(c_times)),
    }
    write_report(args.report, report)
    print(
        f"All {len(entries)} image(s) match the complete TFLite reference "
        f"model byte-for-byte."
    )


if __name__ == "__main__":
    main()
