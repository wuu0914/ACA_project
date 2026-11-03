#!/usr/bin/env python3
"""Byte-compare the B1 Conv2D fast path and scalar fallback with TFLite."""

import ctypes as C
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
B1 = C.CDLL(str(ROOT / "build/b1/libb1_conv2d.so"))
TFLITE = C.CDLL(str(ROOT / "build/libtflite_reference.so"))


class Conv2DParams(C.Structure):
    _fields_ = [
        (name, C.c_int)
        for name in (
            "batches input_H input_W input_C filter_H filter_W filter_input_C "
            "output_H output_W output_C stride_H stride_W dilation_H dilation_W "
            "pad_H pad_W input_offset output_offset activation_min activation_max"
        ).split()
    ]


def pointer(array, c_type):
    return array.ctypes.data_as(C.POINTER(c_type))


def quantize_multiplier(real_multiplier):
    """real_multiplier ~= multiplier / 2^31 * 2^shift."""
    significand, shift = math.frexp(real_multiplier)
    multiplier = math.floor(significand * (1 << 31) + 0.5)
    if multiplier == 1 << 31:
        multiplier //= 2
        shift += 1
    return multiplier, shift


def run(kernel, source, weights, bias, multipliers, shifts, params):
    output = np.empty(
        params.batches * params.output_H * params.output_W * params.output_C,
        dtype=np.int8,
    )
    kernel(
        pointer(source, C.c_int8),
        pointer(weights, C.c_int8),
        pointer(bias, C.c_int32),
        pointer(multipliers, C.c_int32),
        pointer(shifts, C.c_int32),
        pointer(output, C.c_int8),
        C.byref(params),
    )
    return output


def compare(name, ours, golden):
    if np.array_equal(ours, golden):
        print(f"PASS  {name}")
        return
    different = np.flatnonzero(ours != golden)
    first = int(different[0])
    raise AssertionError(
        f"{name}: {different.size} byte(s) differ; first index {first}: "
        f"B1={ours[first]}, TFLite={golden[first]}"
    )


def test_1x1_fast_path():
    """Exercise two complete output-channel tiles."""
    original_offset = 11
    original = Conv2DParams(
        1, 3, 4, 5, 1, 1, 5, 3, 4, 8,
        1, 1, 1, 1, 0, 0, original_offset, -3, -110, 107,
    )
    b1 = Conv2DParams(*[
        0 if name == "input_offset" else getattr(original, name)
        for name, _ctype in Conv2DParams._fields_
    ])

    source = np.resize(
        np.array([-128, -127, -12, -1, 0, 1, 37, 126, 127], dtype=np.int8),
        original.batches * original.input_H * original.input_W * original.input_C,
    )
    weights = np.resize(
        np.array([127, -128, 31, -17, 0, 1, -1], dtype=np.int8),
        original.output_C * original.filter_input_C,
    )
    bias = np.arange(original.output_C, dtype=np.int32) * 101 - 253
    weight_rows = weights.astype(np.int64).reshape(original.output_C, -1)
    adjusted_bias = (
        bias.astype(np.int64) + original_offset * weight_rows.sum(axis=1)
    ).astype(np.int32)
    scales = [quantize_multiplier(value)
              for value in np.linspace(0.013, 0.43, original.output_C)]
    multipliers = np.array([value[0] for value in scales], dtype=np.int32)
    shifts = np.array([value[1] for value in scales], dtype=np.int32)

    ours = run(B1.conv2d_s8, source, weights, adjusted_bias,
               multipliers, shifts, b1)
    golden = run(TFLITE.tflite_ref_conv2d_s8, source, weights, bias,
                 multipliers, shifts, original)
    compare("B1 1x1 adjusted-bias tile-4", ours, golden)


def test_3x3_fallback():
    """Keep padding and the original nonzero input offset on the B0 path."""
    params = Conv2DParams(
        1, 4, 5, 2, 3, 3, 2, 2, 3, 3,
        2, 2, 1, 1, 0, 1, 11, -7, -100, 100,
    )
    source = np.resize(
        np.array([-128, -127, -12, -1, 0, 1, 37, 126, 127], dtype=np.int8),
        params.batches * params.input_H * params.input_W * params.input_C,
    )
    weights = np.resize(
        np.array([127, -128, 31, -17, 0, 1, -1], dtype=np.int8),
        params.output_C * params.filter_H * params.filter_W
        * params.filter_input_C,
    )
    bias = np.arange(params.output_C, dtype=np.int32) * 97 - 101
    scales = [quantize_multiplier(value)
              for value in np.linspace(0.017, 0.41, params.output_C)]
    multipliers = np.array([value[0] for value in scales], dtype=np.int32)
    shifts = np.array([value[1] for value in scales], dtype=np.int32)

    ours = run(B1.conv2d_s8, source, weights, bias,
               multipliers, shifts, params)
    golden = run(TFLITE.tflite_ref_conv2d_s8, source, weights, bias,
                 multipliers, shifts, params)
    compare("B1 scalar fallback 3x3 stride-2 SAME", ours, golden)


if __name__ == "__main__":
    test_1x1_fast_path()
    test_3x3_fallback()
