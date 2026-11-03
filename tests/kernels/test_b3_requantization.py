#!/usr/bin/env python3
"""Byte-compare the B3 Conv2D and DepthwiseConv2D paths with TFLite."""

import ctypes as C
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
B3 = C.CDLL(str(ROOT / "build/b3/libb3_kernels.so"))
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


def values(size, reverse=False):
    pattern = np.array(
        [-128, -127, -17, -1, 0, 1, 31, 126, 127], dtype=np.int8
    )
    if reverse:
        pattern = pattern[::-1]
    return np.ascontiguousarray(np.resize(pattern, size))


def scaling(output_c):
    scales = [
        quantize_multiplier(value)
        for value in np.linspace(0.013, 0.43, output_c)
    ]
    multipliers = np.array([value[0] for value in scales], dtype=np.int32)
    shifts = np.array([value[1] for value in scales], dtype=np.int32)
    assert np.all(multipliers > 0) and np.all(shifts < 0)
    return multipliers, shifts


def run(kernel, source, weights, bias, multipliers, shifts, params):
    output = np.empty(
        params.batches * params.output_H * params.output_W * params.output_C,
        dtype=np.int8,
    )
    kernel(
        pointer(source, C.c_int8), pointer(weights, C.c_int8),
        pointer(bias, C.c_int32), pointer(multipliers, C.c_int32),
        pointer(shifts, C.c_int32), pointer(output, C.c_int8),
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
        f"B3={ours[first]}, TFLite={golden[first]}"
    )


def test_conv2d_1x1():
    """Exercise X1 MACs followed by B3 requantization."""
    original_offset = 11
    original = Conv2DParams(
        1, 3, 4, 8, 1, 1, 8, 3, 4, 8,
        1, 1, 1, 1, 0, 0, original_offset, -3, -128, 127,
    )
    b3 = Conv2DParams(*[
        0 if name == "input_offset" else getattr(original, name)
        for name, _ctype in Conv2DParams._fields_
    ])
    source = values(original.input_H * original.input_W * original.input_C)
    weights = values(original.output_C * original.filter_input_C, reverse=True)
    bias = np.arange(original.output_C, dtype=np.int32) * 101 - 253
    weight_rows = weights.astype(np.int64).reshape(original.output_C, -1)
    adjusted_bias = (
        bias.astype(np.int64) + original_offset * weight_rows.sum(axis=1)
    ).astype(np.int32)
    multipliers, shifts = scaling(original.output_C)

    assert source.ctypes.data % 8 == 0 and weights.ctypes.data % 8 == 0
    ours = run(B3.conv2d_s8, source, weights, adjusted_bias,
               multipliers, shifts, b3)
    golden = run(TFLITE.tflite_ref_conv2d_s8, source, weights, bias,
                 multipliers, shifts, original)
    compare("B3 X1 Conv2D", ours, golden)


def test_conv2d_3x3():
    """Exercise the first-layer-style scalar MAC followed by B3."""
    params = Conv2DParams(
        1, 5, 6, 3, 3, 3, 3, 3, 3, 4,
        2, 2, 1, 1, 1, 1, 11, -7, -128, 127,
    )
    source = values(params.input_H * params.input_W * params.input_C)
    weights = values(
        params.output_C * params.filter_H * params.filter_W
        * params.filter_input_C,
        reverse=True,
    )
    bias = np.arange(params.output_C, dtype=np.int32) * 97 - 101
    multipliers, shifts = scaling(params.output_C)
    ours = run(B3.conv2d_s8, source, weights, bias,
               multipliers, shifts, params)
    golden = run(TFLITE.tflite_ref_conv2d_s8, source, weights, bias,
                 multipliers, shifts, params)
    compare("B3 scalar Conv2D", ours, golden)


def test_depthwise_conv2d():
    """Exercise B2 pointer walking followed by B3 requantization."""
    params = DepthwiseConv2DParams(
        1, 5, 6, 8, 5, 5, 3, 3, 8, 1,
        2, 2, 1, 1, 2, 2, 128, 7, -128, 127,
    )
    source = values(params.input_H * params.input_W * params.input_C)
    weights = values(params.filter_H * params.filter_W * params.output_C,
                     reverse=True)
    bias = np.arange(params.output_C, dtype=np.int32) * 53 - 271
    multipliers, shifts = scaling(params.output_C)
    ours = run(B3.depthwise_conv2d_s8, source, weights, bias,
               multipliers, shifts, params)
    golden = run(TFLITE.tflite_ref_depthwise_conv2d_s8,
                 source, weights, bias, multipliers, shifts, params)
    compare("B3 B2 DepthwiseConv2D", ours, golden)


if __name__ == "__main__":
    test_conv2d_1x1()
    test_conv2d_3x3()
    test_depthwise_conv2d()
