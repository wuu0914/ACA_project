#!/usr/bin/env python3
"""Byte-compare the B2 DepthwiseConv2D fast path and fallback with TFLite."""

import ctypes as C
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
B2 = C.CDLL(str(ROOT / "build/b2/libb2_depthwise_conv2d.so"))
TFLITE = C.CDLL(str(ROOT / "build/libtflite_reference.so"))


class DepthwiseConv2DParams(C.Structure):
    _fields_ = [
        (name, C.c_int)
        for name in (
            "batches input_H input_W input_C filter_H filter_W output_H output_W "
            "output_C depth_multiplier stride_H stride_W dilation_H dilation_W "
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


def corner(size, reverse=False):
    pattern = np.array(
        [-128, -127, -17, -1, 0, 1, 31, 126, 127], dtype=np.int8
    )
    if reverse:
        pattern = pattern[::-1]
    return np.ascontiguousarray(np.resize(pattern, size))


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
        f"B2={ours[first]}, TFLite={golden[first]}"
    )


def check(name, params):
    """Send one complete DepthwiseConv2D case to B2 and TFLite."""
    source = corner(
        params.batches * params.input_H * params.input_W * params.input_C
    )
    weights = corner(
        params.filter_H * params.filter_W * params.output_C,
        reverse=True,
    )
    bias = np.arange(params.output_C, dtype=np.int32) * 53 - 271
    scales = [
        quantize_multiplier(value)
        for value in np.linspace(0.013, 0.37, params.output_C)
    ]
    multipliers = np.array([value[0] for value in scales], dtype=np.int32)
    shifts = np.array([value[1] for value in scales], dtype=np.int32)

    ours = run(B2.depthwise_conv2d_s8, source, weights, bias,
               multipliers, shifts, params)
    golden = run(TFLITE.tflite_ref_depthwise_conv2d_s8,
                 source, weights, bias, multipliers, shifts, params)
    compare(name, ours, golden)


def test_fast_paths():
    """Cover borders, interiors, both model filter sizes and both strides."""
    check(
        "B2 tile-4 3x3 stride-1 SAME",
        DepthwiseConv2DParams(
            1, 4, 5, 4, 3, 3, 4, 5, 4, 1,
            1, 1, 1, 1, 1, 1, 11, -5, -117, 109,
        ),
    )
    check(
        "B2 tile-4 5x5 stride-2 SAME",
        DepthwiseConv2DParams(
            1, 5, 6, 8, 5, 5, 3, 3, 8, 1,
            2, 2, 1, 1, 2, 2, 128, 7, -128, 127,
        ),
    )


def test_scalar_fallback():
    """Depth multiplier two is intentionally outside the B2 contract."""
    check(
        "B2 scalar fallback depth-multiplier-2",
        DepthwiseConv2DParams(
            1, 4, 4, 4, 3, 3, 4, 4, 8, 2,
            1, 1, 1, 1, 1, 1, -7, 3, -105, 113,
        ),
    )


if __name__ == "__main__":
    test_fast_paths()
    test_scalar_fallback()
